from aimotion_f1tenth_simulator.classes.controller_base import ControllerBase
from aimotion_f1tenth_simulator.classes.mpcc_util.base_mpcc import Base_MPCC_Controller
from aimotion_f1tenth_simulator.classes.mpcc_util.casadi_mpcc import Casadi_MPCC
from aimotion_f1tenth_simulator.classes.mpcc_util.theta_opt import Theta_opt
from aimotion_f1tenth_simulator.classes.traj_classes import Spline_2D
import numpy as np
from scipy.interpolate import splrep, splprep, splev
import matplotlib.pyplot as plt
#MPCC includes
import casadi as cs
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
from scipy.interpolate import splev
from scipy.integrate import ode
r = 1


class CarMPCCController(Base_MPCC_Controller):
    def __init__(self, vehicle_params: dict, MPCC_params: dict, mute = False, lin_tire = False, index = 0):
        """
        Init controller parameters
        :param vehicle_params: dict
        :param MPCC_params: dict
        """
        self.muted = mute
        self.vehicle_params = vehicle_params
        self.MPCC_params = MPCC_params
        self.trajectory = None
        self.index = index
        self.theta = 0.0
        self.theta_dot = 0.0
        self.s_start = 0.0
        self.x0 = np.zeros(6)
        self.c_t = 0
        self.input = np.array([self.MPCC_params["d_max"],0])
       
        self.ocp_solver = None #acados solver to compute control

        self.casadi_solver = None #casadi_solver for initial guess

        self.errors = {"contouring" : 0, "longitudinal": 0, "progress": 0, "c_t": 0}
        self.finished = False
    
        self.prev_phi = 0

        self.reverse = False #Reverse mode flag


        self.lin_tire = lin_tire #Flag for using linear tire force model
        self.alpha = 1
        self.prev_state = np.zeros(6)


    def load_parameters(self):
        """
        Load self parameters from the dict-s
        """
        self.parameters = cs.types.SimpleNamespace()

        self.casadi_MPCC_params = self.MPCC_params.copy()
        
        m = float(self.vehicle_params["m"])
        l_f = float(self.vehicle_params["l_f"])
        l_r = float(self.vehicle_params["l_r"])
        I_z = float(self.vehicle_params["I_z"])

        C_m1 = float(self.vehicle_params["C_m1"])
        C_m2 = float(self.vehicle_params["C_m2"])
        C_m3 = float(self.vehicle_params["C_m3"])

        C_f = float(self.vehicle_params["C_f"])
        C_r = float(self.vehicle_params["C_r"])

        if self.reverse == True:
            C_f = -1*C_f
            C_r = -1*C_r

        self.parameters.m = m
        self.parameters.l_f = l_f
        self.parameters.l_r = l_r
        self.parameters.I_z = I_z
        self.parameters.C_m1 = C_m1
        self.parameters.C_m2 = C_m2
        self.parameters.C_m3 = C_m3
        self.parameters.C_f = C_f
        self.parameters.C_r = C_r
        #incremental input constraints:

        self.parameters.ddot_max= float(self.MPCC_params["ddot_max"])
        self.parameters.deltadot_max = float(self.MPCC_params["deltadot_max"])
        self.parameters.thetahatdot_min = float(self.MPCC_params["thetahatdot_min"])
        self.parameters.thetahatdot_max = float(self.MPCC_params["thetahatdot_max"])

        #input constraints:
        delta_max = float(self.MPCC_params["delta_max"])


        if self.reverse: #If reverse mode is activated change the min and max value of d 
            d_max = -float(self.MPCC_params["d_min"])
            d_min = -float(self.MPCC_params["d_max"])

            self.casadi_MPCC_params["d_max"] = -float(self.MPCC_params["d_min"])

            self.casadi_MPCC_params["d_min"] = -float(self.MPCC_params["d_max"])

        else:
            d_max = float(self.MPCC_params["d_max"])
            d_min = float(self.MPCC_params["d_min"])
        

        #ocp parameters:
        self.parameters.N = int(self.MPCC_params["N"])
        self.parameters.Tf = float(self.MPCC_params["Tf"])
        self.parameters.opt_tol = float(self.MPCC_params["opt_tol"])
        self.parameters.max_QP_iter = int(self.MPCC_params["max_QP_iter"])
        #cost weights:
        self.parameters.delta_max = delta_max
        self.parameters.d_max = d_max
        self.parameters.d_min = d_min
        self.parameters.q_con = float(self.MPCC_params["q_con"])
        self.parameters.q_lat = float(self.MPCC_params["q_long"])
        self.parameters.q_theta = float(self.MPCC_params["q_theta"])
        self.parameters.q_d = float(self.MPCC_params["q_d"])
        self.parameters.q_delta = float(self.MPCC_params["q_delta"])

    def compute_control(self, x0, setpoint,time, **kwargs):
        """
        Calculating the optimal inputs
        :param x0 (1xNx array)
        :setpoint = None
        :return u_opt (optimal input vector)
        :return errors (contouring error, heading angle, longitudinal errors, theta, v_xi)
        :return finished (trajectory execution finished)
        """

       
        x0 = np.array([x0["pos_x"], x0["pos_y"], x0["head_angle"], x0["long_vel"], x0["lat_vel"], x0["yaw_rate"]])

        while x0[2]-self.prev_phi > np.pi:
            x0[2] = x0[2]-2*np.pi
        while x0[2]-self.prev_phi < -np.pi:
            x0[2] = x0[2]+2*np.pi

        self.prev_phi = x0[2]

        self.prev_state = x0


        if self.theta >= self.trajectory.L-0.05:
            errors = np.array([0.0, 0.0,0.0,float(self.theta), 0.0])
            u_opt = np.array([0,0])
            self.input = np.array([0,0])
            self.errors = {"contouring" : 0, "longitudinal": 0, "progress": 0, "c_t": 0}
            self.finished = True
            return u_opt
        

        if x0[3]*cs.sign(self.parameters.d_max) <0.5:
            x0[3] = 0.5*cs.sign(self.parameters.d_max)

        x0 = np.concatenate((x0, np.array([self.theta]), self.input))

        self.state_vector = np.reshape(x0, (-1,1))

        self.ocp_solver.set(0, 'lbx', x0)
        self.ocp_solver.set(0, 'ubx', x0)
        self.ocp_solver.set(0, 'x', x0)


        tol = self.parameters.opt_tol
        t = 0
        for i in range(self.parameters.max_QP_iter):
            self.ocp_solver.solve()
            res = self.ocp_solver.get_residuals()

            t += self.ocp_solver.get_stats("time_tot")
            num_iter = i+1
            if max(res) < tol:
                break #Tolerance limit reached
            if t > 1/150: #If the controller frequency is below 60 Hz break
                if self.muted == False:
                    #print("\nTime limit reached: 150Hz\n")
                    pass

        x_opt = np.reshape(self.ocp_solver.get(1, "x"),(-1,1)) #Full predictied optimal state vector (x,y,phi, vxi, veta, omega, thetahat, d, delta)
        self.input = x_opt[7:, 0]
        self.theta = np.reshape(self.ocp_solver.get(1, "x"),(-1,1))[6,0]

        u_opt = np.reshape(self.ocp_solver.get(1, "x"),(-1,1))[7:,0]
        if (1/t < self.MPCC_params["freq_limit"]) or (max(res) > self.MPCC_params["res_limit"]):
            raise Exception(f"Slow computing, emergency shut down, current freq: {1/t}, residuals: {res}")
        if self.muted == False:
            print(f"\rFrequency: {(1/(t)):4f}, solver time: {t:.5f}, QP iterations: {num_iter:2}, progress: {self.theta/self.trajectory.L*100:.2f}%, input: {u_opt}, residuals: {res}               \r", end = '', flush=True)
        
        for i in range(self.parameters.N-1):
            x_i = self.ocp_solver.get(i+1, "x")
            self.ocp_solver.set(i, "x", x_i)

            self.state_vector = np.append(self.state_vector, np.reshape(x_i, (-1,1)), axis = 1)

        for i in range(self.parameters.N-2):
            self.ocp_solver.set(i, "u", self.ocp_solver.get(i+1, "u"))


        e_con = self.trajectory.e_c(x_opt[:2,0], self.theta)[0][0]

        e_long = self.trajectory.e_l(x_opt[:2,0],self.theta)[0][0]

        

        self.errors = {"contouring" : float(e_con), "longitudinal": float(e_long), "progress": self.theta, "c_t": t}
        
        self.c_t = t

        self.theta_dot = self.ocp_solver.get(0, "u")[0]
        

        #u_opt = np.array([-0.1, -0.3])
        return u_opt

        
    def get_errors(self) -> dict:
        """Returns the current error values

        Returns:
            dict: Dict containing the current error values
        """
        return self.errors
    
    def get_inputs(self) -> dict:
        """Returns the current input values

        Returns:
            dict: Dict containing the current input values
        """

        return {"d": self.input[0], "delta": self.input[1]}
    
    

    def _generate_model(self):
        """
        Class method for creating the AcadosModel. 
        Sets self.parameters used by the casadi solver.
        """
        self.load_parameters() #Make sure that the SimpleNameSpace variables are updated to the current MPCC parameters

        m= self.parameters.m
        l_f=self.parameters.l_f 
        l_r=self.parameters.l_r 
        I_z=self.parameters.I_z 
        C_m1=self.parameters.C_m1
        C_m2=self.parameters.C_m2
        C_m3=self.parameters.C_m3
        C_f=self.parameters.C_f 
        C_r=self.parameters.C_r 

        model = AcadosModel()

        model.name = "f1tenth_bicycle_model"

        """ Creating the state vector: [x,y,vx,vy,thetahat, thetahat, d, delta]' """
        x = cs.MX.sym('x')
        y = cs.MX.sym('y')
        phi = cs.MX.sym("phi")
        vxi = cs.MX.sym("vxi")
        veta = cs.MX.sym("veta")
        omega = cs.MX.sym("omega")
        thetahat = cs.MX.sym("thetahat")
        d = cs.MX.sym("d")
        delta = cs.MX.sym("delta")


        model.x = cs.vertcat(x,y,phi,vxi, veta, omega,thetahat, d, delta)
       
       
       
        #Defining the slip angles
        if not self.lin_tire:
            alpha_r = cs.arctan2((-veta+l_r*omega),(vxi+0.0001* cs.sign(self.parameters.d_max))) #In the documentation arctan2 is used but vxi can't be < 0
            alpha_f = delta- cs.arctan2((veta+l_f*omega),(vxi+0.0001* cs.sign(self.parameters.d_max)))
        else:
            alpha_r = (-veta+l_r*omega)/(vxi+0.0001* cs.sign(self.parameters.d_max))
            alpha_f = delta-(veta+l_f*omega)/(vxi+0.0001* cs.sign(self.parameters.d_max))


        #Wheel forces

        Fxi = C_m1*d-C_m2*vxi-cs.sign(vxi)*C_m3

        Freta = C_r * alpha_r
        Ffeta = C_f*alpha_f
        
        #State derivates:
        xdot = cs.MX.sym("xdot")
        ydot =  cs.MX.sym("ydot")
        phidot =  cs.MX.sym("phidot")
        vxidot =  cs.MX.sym("vxidot")
        vetadot =  cs.MX.sym("vetadot")
        omegadot =  cs.MX.sym("omegadot")
        thetahatdot =  cs.MX.sym("thetadot")
        ddot =  cs.MX.sym("ddot")
        deltadot =  cs.MX.sym("deltadot")

        model.xdot = cs.vertcat(xdot,ydot, phidot, vxidot, vetadot,omegadot, thetahatdot, ddot, deltadot)

        """Input vector: [ddot, deltadot, thetahatdot]' """
        model.u = cs.vertcat(thetahatdot, ddot, deltadot)

        #Tire-force ellipse:
        F_rear = Fxi**2/self.MPCC_params["mu_xi"]**2+Freta**2/self.MPCC_params["mu_eta"]**2
        F_front = Fxi**2/self.MPCC_params["mu_xi"]**2+Ffeta**2/self.MPCC_params["mu_eta"]**2

        model.con_h_expr = cs.vertcat(F_front, F_rear)

        """Explicit expression:"""

        model.f_expl_expr = cs.vertcat(
            vxi*cs.cos(phi)-veta*cs.sin(phi), #xdot
            vxi*cs.sin(phi)+veta*cs.cos(phi), #ydot
            omega, #phidot
            (1/m)*(Fxi+Fxi*cs.cos(delta)-Ffeta*cs.sin(delta)+m*veta*omega), #vxidot
            (1/m)*(Freta+Fxi*cs.sin(delta)+Ffeta*cs.cos(delta)-m*vxi*omega), #vetadot
            (1/I_z)*(Ffeta*l_f*cs.cos(delta)+Fxi*l_f*cs.sin(delta)-Freta*l_r), #omegadot
            thetahatdot,
            ddot,
            deltadot,
        )

        #Current position
        if self.parameters.d_max < 0:
            point = cs.vertcat(x- self.parameters.l_r* cs.cos(phi),y- self.parameters.l_r*cs.sin(phi)) 
        else:
            point = cs.vertcat(x,y)


        model.cost_expr_ext_cost = self._cost_expr(point, theta=thetahat,thetahatdot=thetahatdot, ddot = ddot, deltadot = deltadot)

        return model


    def _cost_expr(self, point,theta,thetahatdot, ddot, deltadot):
        """
        Definition of the cost expression
        :param point: array containing x and y coordinates
        :param theta: path parameter
        :return: cost value (scalar)
        """

        e_c = self._cost_e_c(point,theta)
        e_l = self._cost_e_l(point,theta)
        cost = e_c**2*self.parameters.q_con+e_l**2*self.parameters.q_lat-thetahatdot*self.parameters.q_theta+self.parameters.q_d*cs.fabs(ddot)+self.parameters.q_delta*cs.fabs(deltadot)
        
        
        return cost


    def _cost_e_c(self, point ,theta):
        """
        Contouring error function
        :param point: array containing x and y coordinates
        :param theta: path parameter(s)
        :return: contouring error
        """

        point_r, v =  self.trajectory.get_path_parameters(theta, self.s_start) #point: vertical, v: horizontal
        n = cs.hcat((v[:, 1], -v[:, 0])) #Creating a perpendicular vector
        e_c = cs.dot(n.T,(point_r-point))
        return e_c


    def _cost_e_l(self, point, theta):
        """
        Lag error function
        :param point: array containing x and y coordinates
        :param theta: path parameter(s)
        :return: lag error
        """
        point_r, v = self.trajectory.get_path_parameters(theta, self.s_start)
        e_l = cs.dot(v.T,(point_r-point))
        return e_l
    
    def reset(self):
        """Reset acados model and ocp solver"""
        self.ocp_solver = None
        self.finished = False

    def train_GP_controllers(self, *args, **kwargs):
        raise NotImplementedError

    def _generate_ocp_solver(self, model: AcadosModel):
        """
        Creates the acados ocp solver
        :param model: AcadosModel, generated by the class method
        :return ocp_solver: AcadosOcpSolver
        """
        ocp = AcadosOcp()
        ocp.model = model

        ocp.dims.N = self.parameters.N

        ocp.solver_options.tf = self.parameters.Tf

        ocp.cost.cost_type = "EXTERNAL"

        ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'#'PARTIAL_CONDENSING_HPIPM' # FULL_CONDENSING_QPOASES
        ocp.solver_options.integrator_type = 'ERK'
        ocp.solver_options.nlp_solver_type = "SQP_RTI"  # SQP_RTI
        ocp.solver_options.nlp_solver_max_iter = 3000
        ocp.solver_options.nlp_solver_tol_stat = 1e-3
        ocp.solver_options.levenberg_marquardt = 10.0
        ocp.solver_options.print_level = 0
        ocp.solver_options.qp_solver_iter_max = 1000
        ocp.code_export_directory = f"acados_c_generated/c_generated_code_{self.index}"
        ocp.solver_options.hessian_approx = 'EXACT'

        lbx = np.array((0,self.parameters.d_min, -self.parameters.delta_max))
        ubx = np.array((self.trajectory.L*1.05,self.parameters.d_max, self.parameters.delta_max)) #TODO: max value of theta 

        ocp.constraints.lbx = lbx
        ocp.constraints.ubx = ubx
        ocp.constraints.idxbx = np.array((6,7,8)) #d and delta

        lbu = np.array((self.parameters.thetahatdot_min,
                        -self.parameters.ddot_max, 
                        -self.parameters.deltadot_max
                        ))
        
        ubu = np.array((self.parameters.thetahatdot_max,
                        self.parameters.ddot_max, 
                        self.parameters.deltadot_max))
        

        ocp.constraints.ubu = ubu
        ocp.constraints.lbu = lbu
        ocp.constraints.idxbu = np.arange(3)
        phi0 = float(self.trajectory.get_path_parameters_ang(self.s_start)[2])


        """using non-linear constraints:"""
        ocp.constraints.lh = np.array([-1,-1])
        ocp.constraints.uh = np.array([1,1])

        ocp.cost.zl = np.array([50000,50000])  # lower slack penalty
        ocp.cost.zu = np.array([50000,50000])  # upper slack penalty
        ocp.cost.Zl = np.array([100000,100000])  # lower slack weight
        ocp.cost.Zu = np.array([100000,100000])  # upper slack weight

        ## Initialize slack variables for lower and upper bounds
        ocp.constraints.lsh = np.zeros(2)
        ocp.constraints.ush = np.zeros(2)
        ocp.constraints.idxsh = np.arange(2)
        #x0 = np.array((float(self.trajectory.spl_sx(self.s_start)), #x
        #            float(self.trajectory.spl_sy(self.s_start)), #y
        #            phi0,#phi
        #            0.001, #vxi
        #            0, #veta
        #            0, # omega
        #            self.s_start+0.05,
        #            0.02, #d
        #            0, #delta
        #))
        x0 = np.concatenate((self.x0, np.array([self.theta]), self.input))

        ocp.constraints.x0 = x0 #Set in the set_trajectory function


        ocp_solver = AcadosOcpSolver(ocp, json_file = f"acados_c_generated/acados_ocp_{self.index}.json")
        return ocp_solver
    

    def set_trajectory(self, pos_tck, evol_tck, generate_solver = True):
        """
        Evaluete the reference spline from the given spline tck, and convert it into a Spline2D instance.
        Args:
            pos_tck(np.array): position spline
            evol_tck(np.array): reference progress spline
            generate_solver(bool): generate ocp solver
        """

        #Transform ref trajectory
        t_end = evol_tck[0][-1]
        t_eval=np.linspace(0, t_end, 10000)
        s=splev(t_eval, evol_tck)
        (x,y) = splev(s, pos_tck)

        points_list = []

        for i in range(len(x)):
            points_list.append([i, x[i], y[i]])

        self.trajectory = Spline_2D(np.array([[0,0,0],[1,1,1],[2,2,2]]))

        self.trajectory.spl_sx = cs.interpolant("traj", "bspline", [s], x)
        self.trajectory.spl_sy = cs.interpolant("traj", "bspline", [s], y)
        self.trajectory.L = s[-1]

        if generate_solver:
            self.generate_solver()

    def generate_solver(self):
        """
        Generate Acados and Casadi solvers. Previously generated solvers're  **deleted**
        """
        self.reset()
        self.load_parameters()
        self.input = np.array([self.parameters.d_max, 0])
        self.ocp_solver = self._generate_ocp_solver(self._generate_model())
        self.casadi_solver = Casadi_MPCC(MPCC_params=self.casadi_MPCC_params,
                                                vehicle_param=self.vehicle_params,
                                                dt = self.parameters.Tf/self.parameters.N,
                                                q_c = self.parameters.q_con,
                                                q_l= self.parameters.q_lat,
                                                q_t = self.parameters.q_theta,
                                                q_delta=self.parameters.q_delta,
                                                q_d = self.parameters.q_d,
                                                theta_0=self.theta,
                                                input_0 = self.input,
                                                trajectory=self.trajectory,
                                                N = self.parameters.N,
                                                x_0 = self.x0)

    def init_controller(self, x0):
        """
        Initialize the Acados SQP solver by solving the optimization problem with Casadi IPOPT
        Args:
            x0(np.array): Initial position of CoM"""


        theta_finder = Theta_opt(point= x0,
                              window= np.array([0,5]),
                              trajectory= self.trajectory)
        theta_start = theta_finder.solve() #Find current theta in the defined window



        self.theta = theta_start
        self.s_start = theta_start
        self.theta_dot = 0.0

        self.x0 = x0
        self.prev_phi = self.x0[2]
        if self.x0[3] < 0.3:
            self.x0[3] = 0.3

        
        self.input = np.array([self.MPCC_params["d_max"],0])

        if self.trajectory == None:
            raise Exception("No trajectory defined")
        
        if self.ocp_solver == None: #If the solver hasn't been generated generate it
            self.generate_solver()

        X, v_U,U, theta, dtheta = self.casadi_solver.opti_step(self.x0) #Call casadi solver for optimal initial guess

        x_0 = np.concatenate((self.x0, np.array([self.theta]), v_U[:,0]))
        self.ocp_solver.set(0, "x", x_0)
        u_0 = np.concatenate((U[:,0], np.array([dtheta[0]])))
        self.ocp_solver.set(0, "u", u_0)



        print(f"x0: {x_0}")
        print(f"u0: {u_0}")
        
        states = np.array(np.reshape(x_0, (-1,1)))

        for i in range(self.parameters.N-1):
            x = np.concatenate((X[:,i+1],np.array([theta[i+1]]), v_U[:,i+1]))
            #x = np.reshape(x, (-1,1))
            states = np.append(states, np.reshape(x,(-1,1)), axis = -1)
            self.ocp_solver.set(i+1, "x", x)
            if dtheta[i] <self.MPCC_params["thetahatdot_min"]:
                dtheta[i] = self.MPCC_params["thetahatdot_min"]
            u = np.concatenate((np.array([dtheta[i]]),U[:,i]))
            self.ocp_solver.set(i, "u", u)
            
        


        self.ocp_solver.set(0, 'lbx', x_0)
        self.ocp_solver.set(0, 'ubx', x_0)
        self.ocp_solver.set(0, 'x', x_0)

        tol =   0.01
        t = 0
        for i in range(1000):
           
            self.ocp_solver.solve()
            res = self.ocp_solver.get_residuals()

            t += self.ocp_solver.get_stats("time_tot")
            num_iter = i+1
            if max(res) < tol:
                break #Tolerance limit reached
            if i %50 ==0:
                print(f"{i}. init itaration, residuals: {res}")
    

    def nonlin_dynamics(self,t, states, inputs):
        '''
        Class method calculating the state derivatives using nonlinear model equations
        :param t: Current time
        :param states: State vector
        :param inputs: Input vector
        :return: Derivative of states
        '''
        #states = np.reshape(states, (1,-1))
        #inputs = np.reshape(inputs, (1,-1))
        #states = states[0, :]
        #inputs = inputs[0,:]
        x = states[0]
        y = states[1]
        phi = states[2]
        v_xi = states[3]
        v_eta = states[4]
        omega = states[5]

        d = inputs[0]
        delta = inputs[1]


        # slip angles
                #Defining the slip angles
        if not self.lin_tire:
            alpha_r = cs.arctan2((-v_eta+self.parameters.l_r*omega),(v_xi+0.0001* cs.sign(self.parameters.d_max))) #In the documentation arctan2 is used but vxi can't be < 0
            alpha_f = delta- cs.arctan2((v_eta+self.parameters.l_f*omega),(v_xi+0.0001* cs.sign(self.parameters.d_max)))
        else:
            alpha_r = (-v_eta+self.parameters.l_r*omega)/(v_xi+0.0001* cs.sign(self.parameters.d_max))
            alpha_f = delta-(v_eta+self.parameters.l_f*omega)/(v_xi+0.0001* cs.sign(self.parameters.d_max))

        # tire forces
        F_xi = self.parameters.C_m1*d - self.parameters.C_m2*v_xi - self.parameters.C_m3*cs.sign(v_xi)
        F_reta = self.parameters.C_r*alpha_r
        F_feta = self.parameters.C_f*alpha_f

        # nonlinear state equations
        dx = v_xi * cs.cos(phi) - v_eta * cs.sin(phi)
        dy = v_xi * cs.sin(phi) + v_eta * cs.cos(phi)
        dphi = omega

        dv_xi = 1 / self.parameters.m * (F_xi + F_xi * cs.cos(delta) - F_feta * cs.sin(delta) + self.parameters.m * v_eta * omega)
        dv_eta = 1 / self.parameters.m * (F_reta + F_xi * cs.sin(delta) + F_feta * cs.cos(delta) - self.parameters.m * v_xi * omega)
        domega = 1 / self.parameters.I_z * (F_feta * self.parameters.l_f * cs.cos(delta) + F_xi * self.parameters.l_f * cs.sin(delta) - F_reta * self.parameters.l_r)
        d_states = cs.vertcat(dx, dy, dphi, dv_xi, dv_eta, domega)
        return d_states


    def predict(self, states, inputs, dt, t=0, method = "RK4"):
        ''' Class method predicting the next state of the model from previous state and given input
        :param states: State vector
        :param inputs: Input vector
        :param t: Current time
        :param dt: Sampling time
        :param method: Numerical method of solving the ODEs
        :return: Predicted state vector, predicted time
        '''
       
        if method == 'RK4':
            k1 = self.nonlin_dynamics(t, states, inputs)
            k2 = self.nonlin_dynamics(t + dt / 2, states + dt / 2 * k1, inputs)
            k3 = self.nonlin_dynamics(t + dt / 2, states + dt / 2 * k2, inputs)
            k4 = self.nonlin_dynamics(t + dt, states + dt * k3, inputs)
            states_next = states + dt/6 * (k1 + 2.0*k2 + 2.0*k3 + k4)
            t_next = t + dt

        elif method == 'FE':
            states_next = states + dt*self.nonlin_dynamics(t, states, inputs)
            t_next = t + dt
        return states_next, t_next
    
    def simulate(self, states, inputs, dt, t=0):
        """
        Class method for simulating the f1tenth vehicle
        :param states: State vector
        :param inputs: Input vector
        :param dt: integration time
        :param t: t_0
        :return states: State vector at t+dt time step
        :return t: t_0+dt
        """
        sim = ode(self.nonlin_dynamics).set_integrator('lsoda')
        sim.set_initial_value(states, t).set_f_params(inputs)

        return sim.integrate(dt), sim.t+dt
