from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
import casadi as cs
import numpy as np
from aimotion_f1tenth_simulator.classes.traj_classes import Spline_2D
from aimotion_f1tenth_simulator.classes.mpcc_util.theta_opt import Theta_opt
from scipy.interpolate import splev
import matplotlib.pyplot as plt
from aimotion_f1tenth_simulator.classes.mpcc_util.base_mpcc import Base_MPCC_Controller
r= 1

class mpcc_reverse_controller(Base_MPCC_Controller):
    """
    Controller for reverse driving of the F1TENTH vehicle using the kinematic bicycle model
    """


    def __init__(self, vehicle_params:dict, MPCC_params: dict, index = 0):
        """
        Args:
            vehicle_params(dict): dictionary containing the vehicle parameters
            MPCC_params(dict): dictionary containing the MPCC parameters
            index(int): index of the created acados solver, the generated c code is marked with this index
        """
        
        self.vehicle_params = vehicle_params
        self.MPCC_params = MPCC_params
        self.trajectory = None
        self.index = index
        self.theta = 0.0
        self.theta_dot = 0.0
        self.s_start = 0.0
        self.x0 = np.zeros(4)

        self.c_t = 0

        self.input = np.array([self.MPCC_params["d_max"],0])
       
        self.ocp_solver = None #acados solver to compute control

        self.casadi_solver = None #casadi_solver for initial guess

        self.errors = {"contouring" : 0, "longitudinal": 0, "progress": 0, "c_t": 0}
        self.finished = False
    
        self.prev_phi = 0

        self.reverse = False #Reverse mode flag

        self.prev_state = np.array([])
        self.load_parameters()

    def load_parameters(self):
        """
        Function for loading the solver parameters from the class dictionary (MPCC_params(dict),vehicle_params(dict))
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


        self.parameters.l_r = l_r
        self.parameters.l_f = l_f
        self.parameters.m = m
        self.parameters.l = l_f+l_r
        self.parameters.C_m1 = C_m1
        self.parameters.C_m2 = C_m2
        self.parameters.C_m3 = C_m3

        #incremental input constraints:

        self.parameters.ddot_max= float(self.MPCC_params["ddot_max"])
        self.parameters.deltadot_max = float(self.MPCC_params["deltadot_max"])
        self.parameters.thetahatdot_min = float(self.MPCC_params["thetahatdot_min"])
        self.parameters.thetahatdot_max = float(self.MPCC_params["thetahatdot_max"])

        #input constraints:
        delta_max = float(self.MPCC_params["delta_max"])


        d_max = -float(self.MPCC_params["d_min"])
        d_min = -float(self.MPCC_params["d_max"])

        self.casadi_MPCC_params["d_max"] = -float(self.MPCC_params["d_min"])

        self.casadi_MPCC_params["d_min"] = -float(self.MPCC_params["d_max"])

        
        

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

    def _generate_model(self)->AcadosModel:
            """
            Class method for creating the AcadosModel. 
            Returns:
                model(AcadosModel): The kinematic bicycle of the F1Tenth vehicle
            """

            m= self.parameters.m
            l = self.parameters.l
            C_m1=self.parameters.C_m1
            C_m2=self.parameters.C_m2
            C_m3=self.parameters.C_m3


            model = AcadosModel()

            model.name = "f1tenth_bicycle_model_reverse"

            """Creating the state vector"""

            x = cs.MX.sym("x")
            y = cs.MX.sym("y")
            phi = cs.MX.sym("phi")
            v = cs.MX.sym("v")
            thetahat = cs.MX.sym("thetahat")
            d = cs.MX.sym("d")
            delta = cs.MX.sym("delta")

            model.x = cs.vertcat(x,y,phi,v, thetahat, d,delta)

            """State derivates"""

            xdot = cs.MX.sym("xdot")
            ydot = cs.MX.sym("ydot")
            phidot = cs.MX.sym("phi")
            a = cs.MX.sym("a")
            thetahatdot = cs.MX.sym("thetahatdot")
            ddot = cs.MX.sym("ddot")
            deltadot = cs.MX.sym("deltadot")

            model.xdot = cs.vertcat(xdot, ydot, phidot, a, thetahatdot,ddot, deltadot)

            model.u = cs.vertcat(thetahatdot, ddot, deltadot)

            model.f_expl_expr = cs.vertcat(
                v*cs.cos(phi), #xdot
                v*cs.sin(phi), #ydot
                v/l*cs.tan(delta), #phidot
                2/m*(C_m1*d- C_m2*v-cs.sign(v)*C_m3), # vdot #USING the drivetrain model from Kristof Page 17(30/80) eq 4.6
                thetahatdot, 
                ddot,
                deltadot
            )

            point = cs.vertcat(x,y)
            model.cost_expr_ext_cost = self._cost_expr(point = point, thetahat = thetahat,thetahatdot = thetahatdot, ddot = ddot, deltadot = deltadot)


            return model
            
    def _cost_expr(self, point, thetahat, thetahatdot, ddot, deltadot):
        """
        Function for expressing the acados optimisation problem
        Args:
            point: array containg x and y coordinates
            thetahat: virtual path parameter
            thetahatdot: change of the virtual path parameter
            ddot: change of the motor reference
            deltadot: change of the steering reference
        Returns:
            cost(int)
        """

        e_l_c = self._cost_e_l_c(point, thetahat) 
        e_l = e_l_c[0]
        e_c = e_l_c[1]

        cost = self.parameters.q_con * e_c**2 + self.parameters.q_lat*e_l**2 - self.parameters.q_theta*thetahatdot+self.parameters.q_d*cs.fabs(ddot)+self.parameters.q_delta*cs.fabs(deltadot)
        return cost

    def _cost_e_l_c(self, point, theta):
        """
        Function for calculating the longitudinal and contouring error for the vehicle
        Args:
            point: array containing x and y coordinates
            theta: path parameter
        Returns:
            errors(np.array([long_error, con_error])): longitudinal and contouring errors
        """

        point_r, v =  self.trajectory.get_path_parameters(theta, self.s_start) #point: vertical, v: horizontal
        n = cs.hcat((v[:, 1], -v[:, 0])) #Creating a perpendicular vector

        e_c = cs.dot(n.T,(point_r-point))
        e_l = cs.dot(v.T,(point_r-point))

        return np.array([e_l, e_c])


    def _generate_ocp_solver(self, model: AcadosModel):
        """
        Creates the acados ocp solver
        Args:
            model(AcadosModel): model of the F1Tenth vehicle
        Returns:
            ocp_solver(AcadosOcpSolver): ocp solver
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
        ubx = np.array((self.trajectory.L*1.05,self.parameters.d_max, self.parameters.delta_max))

        ocp.constraints.lbx = lbx
        ocp.constraints.ubx = ubx
        ocp.constraints.idxbx = np.array((4,5,6)) #d and delta

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
        if generate_solver == True:
            self.generate_solver()

    def generate_solver(self):
        """
        Generate Acados and Casadi solvers. Previously generated solvers're  **deleted**
        """

        self.reset()
        self.load_parameters()
        self.input = np.array([self.parameters.d_min, 0])
        self.ocp_solver = self._generate_ocp_solver(self._generate_model())
        self.casadi_solver = casadi_reverse_MPCC(MPCC_params=self.casadi_MPCC_params,
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
        shifted_CoM = np.array([x0[0]-cs.cos(x0[2])*self.parameters.l_r, x0[1]-cs.sin(x0[2])*self.parameters.l_r])

        theta_finder = Theta_opt(point= shifted_CoM,
                              window= np.array([0,5]),
                              trajectory= self.trajectory)
        theta_start = theta_finder.solve() #Find current theta in the defined window


        self.load_parameters()


        self.theta = theta_start
        self.s_start = theta_start
        self.theta_dot = 0.0

        self.x0 = np.array([shifted_CoM[0], shifted_CoM[1], x0[2], x0[3]])
        self.prev_phi = self.x0[2]
        if self.x0[3] > -0.5:
            self.x0[3] = -0.5


        self.input = np.array([self.MPCC_params["d_max"],0])

        if self.trajectory == None:
            raise Exception("No trajectory defined")
        
        if self.ocp_solver == None: #If the solver hasn't been generated generate it
            self.ocp_solver = self._generate_ocp_solver(self._generate_model())
            self.casadi_solver = casadi_reverse_MPCC(MPCC_params=self.casadi_MPCC_params,
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
    
    def get_errors(self):
        return self.errors

    def compute_control(self, x0, setpoint,time, **kwargs):
        """
        Calculating the optimal inputs
        Args:
            x0(np.array): Initial state (1 x model.nx)
            setpoint(None): Not used arg
        Returns:
            u_opt(np.array): Vector of optimal inputs (1 x model.nu)
            errors(np.array): [contouring error, heading angle, longitudinal errors, theta, v_xi]
            finished(bool): execution finished flag
        """

        x0 = np.array([x0["pos_x"]-self.parameters.l_r*cs.cos(x0["head_angle"]), x0["pos_y"]-self.parameters.l_r*cs.sin(x0["head_angle"]), x0["head_angle"], x0["long_vel"]])

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

        if x0[3] > -0.5:
            x0[3] = -0.5

        x0 = np.concatenate((x0, np.array([self.theta]), self.input))

        self.state_vector = np.reshape(x0, (-1,1))

        self.ocp_solver.set(0, 'lbx', x0)
        self.ocp_solver.set(0, 'ubx', x0)
        self.ocp_solver.set(0, 'x', x0)

        t = 0
        for i in range(self.parameters.max_QP_iter):
            self.ocp_solver.solve()
            res = self.ocp_solver.get_residuals()

            t += self.ocp_solver.get_stats("time_tot")
            num_iter = i+1
            if max(res) < self.parameters.opt_tol:
                break #Tolerance limit reached
        
        x_opt = np.reshape(self.ocp_solver.get(1, "x"),(-1,1)) #Full predictied optimal state vector (x,y,phi, vxi, veta, omega, thetahat, d, delta)

        self.input = x_opt[5:, 0]
        self.theta = np.reshape(self.ocp_solver.get(1, "x"),(-1,1))[4,0]


        u_opt = np.reshape(self.ocp_solver.get(0, "x"),(-1,1))[5:,0]

        if (1/t < self.MPCC_params["freq_limit"]) or (max(res) > self.MPCC_params["res_limit"]):
            raise Exception(f"Slow computing, emergency shut down, current freq: {1/t}, residuals: {res}")
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
    
    #Basic functions that are not implemented:
    def reset(self):
        """Reset acados model and ocp solver"""
        self.model = None
        self.ocp_solver = None
        self.finished = False

    def train_GP_controllers(self, *args, **kwargs):
        raise NotImplementedError
    

class casadi_reverse_MPCC():

    def __init__(self,MPCC_params:dict,input_0, vehicle_param:dict, dt: float, q_c: float, q_l: float, q_t: float,q_delta:float, q_d: float, N: int, x_0, theta_0:float, trajectory: Spline_2D, lin_tire = False):
        """
        Args:
            q_c (float): Contouring error weight
            q_l (float): Lag error weight
            q_t (float): Progress weight
            N (int): Control horizon
            dt (float): Sampling time
        """
        self.MPCC_params = MPCC_params
        self.model = Model(vehicle_params=vehicle_param, MPPC_params=MPCC_params)


        self.nx = self.model.nx # = 4
        self.nu = self.model.nu # = 2


        self.dt = dt
        self.q_c = q_c
        self.q_l = q_l
        self.q_t = q_t
        self.q_delta = q_delta #The coeficience of the smoothness of the control
        self.q_d =q_d
        self.N = N
        self.trajectory = trajectory

        x_0 = np.reshape(x_0, (-1, 1))

        self.X_init = np.repeat(x_0, N, axis=1) #np.repeat(x_0, N, axis=1)  # np.zeros((6, N))
        self.X_init[0, :] = np.reshape(self.trajectory.spl_sx(np.linspace(theta_0, theta_0 + self.N*self.dt*2, self.N)), (-1))
        self.X_init[1, :] = np.reshape(self.trajectory.spl_sy(np.linspace(theta_0, theta_0 + self.N * self.dt * 2, self.N)), (-1))
        
        #Virtual input initital guess
        
        temp = np.reshape(input_0,(-1,1))
        self.v_U_init = np.repeat(temp, N-1, axis= 1)

        temp = np.zeros((2,1))
        
        
        
        temp = np.reshape(temp,(-1,1)) # I use this to create the initial guess for U (U_init)

        temp = np.zeros((2,1))

        self.U_init = np.repeat(temp, N-2, axis= 1) ##I assume that there is no steering input
        

        
        self.theta_init = np.ones(N)* theta_0
        self.theta_init[0] = theta_0
        self.start_theta = theta_0

        self.v_t_init = np.ones(N-1)*self.MPCC_params["thetahatdot_min"] #Store the initial guess for v_theta

        
        self.lam_g_init = 0 #What does this do?

        # Time measurement variables
        self.time_proc = np.array([])
        self.time_wall = np.array([])
        self.iter_count = np.array([])

        # Initialise optimizer
        self.__init_optimizer()


    def __init_optimizer(self):
        """
        Private method that initializes the optimizer
        """

        self.opti = cs.Opti()

        # Declare decision variables
        self.X = self.opti.variable(self.nx, self.N+1)  # state trajectory

        self.U = self.opti.variable(self.nu, self.N-1)  # input trajectory, containing dd and ddelta

        self.theta = self.opti.variable(self.N+1)  # virtual progress state trajectory

        self.v_t = self.opti.variable(self.N)  # virtual speed input trajectory

        self.v_U = self.opti.variable(self.nu, self.N)#virtual input trajectory containing d and delta

        # Parameters for initial conditions
        self.X_0 = self.opti.parameter(self.nx, 1) #The value is set at every call of the optistep(set to the current x[k])
        self.theta_0 = self.opti.parameter(1) #The value is set at every call of optistep  to theta_init[0]
        self.v_t_0 = self.opti.parameter(1) # -||- to v_t_init[0]

        self.U_0 = self.opti.parameter(self.nu,1)
        self.v_U_0 = self.opti.parameter(self.nu,1)


         # Set constraints for initial values#

        self.opti.subject_to(self.X[:, 0] == self.X_0)  # initial state

        self.opti.subject_to(self.v_t[0] == self.v_t_0)  # initial path speed

        self.opti.subject_to(self.theta[0] == self.theta_0)  # init path parameter

        #self.opti.subject_to(self.U[:,0] == self.U_0) ##Initial input constraint: dd and ddelta

        self.opti.subject_to(self.v_U[:,0] == self.v_U_0) #init virtual input (d & delta)


        x_pred = self.model.predict(states=self.X[:,:-1], inputs=self.v_U, dt=self.dt)[0]

        self.opti.subject_to(self.X[:, 1:] == x_pred) #Settings the constraint


        theta_next = self.theta[:-1] + self.dt * self.v_t #Theta[k+1] = Theta[k]+dt*v_t
        self.opti.subject_to(self.theta[1:] == theta_next) #Settings constraint


        v_U_next = self.v_U[:,:-1]+self.dt*self.U #v_U[k+1] = ... (basically the same)
        #v_U: is a 2xN array & U is a 2x(N-1) array

        self.opti.subject_to(self.v_U[:,1:] == v_U_next) #Constraint :)

        self.opti.subject_to((self.theta[-1]) <= self.trajectory.L)


        self.opti.subject_to((self.model.parameters["d_min"] <= cs.vec(self.v_U[0, 1:]), cs.vec(self.v_U[0, 1:]) <= self.model.parameters["d_max"]))  # motor reference constraints

        self.opti.subject_to((-self.model.parameters["delta_max"] <= cs.vec(self.v_U[1, 1:]), cs.vec(self.v_U[1, 1:]) <= self.model.parameters["delta_max"] ))  # steering angle constraints

         
        # Input constraints: derivate of d and delta
        self.opti.subject_to((-self.model.parameters["ddot_max"] <= cs.vec(self.U[0, :]), cs.vec(self.U[0, :]) <= self.model.parameters["ddot_max"]))  # motor reference constraints
        self.opti.subject_to((-self.model.parameters["deltadot_max"] <= cs.vec(self.U[1, :]), cs.vec(self.U[1, :]) <= self.model.parameters["deltadot_max"]))  # steering angle constraints
        

        self.opti.subject_to((self.model.parameters["d_min"] < cs.vec(self.v_U[0, 1:]), cs.vec(self.v_U[0, 1:]) <= self.model.parameters["d_max"]))  # motor reference constraints
        self.opti.subject_to((-self.model.parameters["delta_max"]<= cs.vec(self.v_U[1, 1:]), cs.vec(self.v_U[1, 1:]) <= self.model.parameters["delta_max"] ))  # steering angle constraints

         
        # Input constraints: derivate of d and delta
        self.opti.subject_to((-self.model.parameters["ddot_max"] <= cs.vec(self.U[0, :]), cs.vec(self.U[0, :]) <= self.model.parameters["ddot_max"]))  # motor reference constraints
        self.opti.subject_to((-self.model.parameters["deltadot_max"]<= cs.vec(self.U[1, :]), cs.vec(self.U[1, :]) <= self.model.parameters["deltadot_max"]))  # steering angle constraints
        

        # Path speed constraints
        self.opti.subject_to((self.model.parameters["thetahatdot_min"] <= cs.vec(self.v_t[1:]), cs.vec(self.v_t[:]) <= self.model.parameters["thetahatdot_max"]))

        # Set objective function
        self.opti.minimize(self.cost())

        # Solver setup
        p_opt = {'expand': False}
        s_opts = {'max_iter': 2000, 'print_level': 5}
        self.opti.solver('ipopt', p_opt, s_opts)

    def opti_step(self, x_0):
        """
        Method that calculates the optimization solution for given initial values

        Args:
            x_0(np.array): initial state vector (model.nx x 1)
        Returns:
            x_opt(np.array): Optimal state vectors (model.nx x N)
            v_u_opt(np.array): Optimal input vectors model.nu x (N-1)
            u_opt(np.array): Change of the optimal inputs (model.nu x (N-2))ű
            theta(np.array): Prediction of theta (1 x N)
            thetadot(np.array): Change of theta (1 x (N-1))
        """

        # Set new values for the opti initial parameters
        self.opti.set_value(self.X_0, x_0) #Setting the value of the opti variable
        #self.X_0 is used to as an initial constraint of the solution: self.X[:,0] == self.X_0


        self.opti.set_value(self.v_t_0, self.v_t_init[0])

        self.opti.set_value(self.U_0, self.U_init[:,0])
        self.opti.set_value(self.v_U_0, self.v_U_init[:,0])

        self.opti.set_value(self.theta_0, self.theta_init[0]) 
      
      
        # Giving initial guesses
        self.opti.set_initial(self.X[:, 1:], self.X_init)
        self.opti.set_initial(self.X[:, 1:-1], self.X_init[:, 1:])
        self.opti.set_initial(self.X[:, -1], self.X_init[:, -1])
        self.opti.set_initial(self.U[:, 1:], self.U_init)
        self.opti.set_initial(self.theta[1:], self.theta_init)
        self.opti.set_initial(self.v_t[1:], self.v_t_init)
        
        
        
        self.opti.set_initial(self.v_U[:,1:], self.v_U_init)
        self.opti.set_initial(self.U[:,1:], self.U_init)
        
        self.opti.set_initial(self.X[:, 0], x_0)
        self.opti.set_initial(self.theta[0], self.theta_init[0])
        self.opti.set_initial(self.v_t[0], self.v_t_init[0])

        self.opti.set_initial(self.opti.lam_g, self.lam_g_init)

        # Solve problem
        sol = self.opti.solve()

        # Measurement of IPOPT CPU time
        self.time_proc = np.append(self.time_proc, sol.stats()['t_proc_total'])
        self.time_wall = np.append(self.time_wall, sol.stats()['t_wall_total'])
        self.iter_count = np.append(self.iter_count, sol.stats()['iter_count'])

        # Saving solutions as initial guess for next step
        self.X_init = sol.value(self.X[:, 1:])
        self.U_init = sol.value(self.U[:, 1:])
        self.theta_init = sol.value(self.theta[1:])
        self.v_t_init = sol.value(self.v_t[1:])

        #Not sure wether these are right: (needs testing)
        self.v_U_init = sol.value(self.v_U[:,1:])
        self.U_init = sol.value(self.U[:,1:])


        self.lam_g_init = sol.value(self.opti.lam_g)
        #s = np.linspace(0,self.trajectory.L, 1000)
        #plt.plot(self.trajectory.spl_sx(s), self.trajectory.spl_sy(s))

        #x = sol.value(self.X[0, :])
        #y = sol.value(self.X[1, :])
        
        #plt.plot(x,y)
        #plt.show()
        return sol.value(self.X), sol.value(self.v_U),sol.value(self.U), sol.value(self.theta[:]), sol.value(self.v_t)


    def cost(self):
        """
        Cost expression
        Returns:
            cost(int): cost expression of the opt problem
        """
        
        shifted_center_point = cs.vcat((self.X[0, :], self.X[1, :]))

            
        e_l = self.e_l(shifted_center_point, self.theta)
        e_c = self.e_c(shifted_center_point, self.theta)
        
        e_smooth = self.e_smooth()
        e_smooth = e_smooth[0]*self.q_d + e_smooth[1]*self.q_delta
        
        #        cost = e_c**2*self.parameters.q_con+e_l**2*self.parameters.q_lat-thetahatdot*self.parameters.q_theta+self.parameters.q_d*ddot**2+self.parameters.q_delta*deltadot**2

        e_progress = cs.sum1(cs.fabs(self.v_t))

        cost = self.q_l * e_l**2 + self.q_c* e_c**2- self.q_t * e_progress + e_smooth 
        #cost = self.q_l * e_l + self.q_c * e_c- self.q_t * (self.trajectory.L-self.theta[-1]) + self.q_smooth  * e_smooth

        return cost

    def e_smooth(self):
        """
        Error of the change of the controll inputs
        Returns:
            e_smooth(np.array): cost_d, cost_delta
        """

        dd = cs.vec(cs.fabs(self.U[0,:]))

        ddelta = cs.vec(cs.fabs(self.U[1,:]))

        #d = cs.fabs(d)

        #d = cs.sum1(d)


        #delta = cs.fabs(delta)

        #delta = cs.sum1(delta)


        e_smooth = (dd.T@dd, ddelta.T @ ddelta)
        #e_smooth = (cs.sum1(dd), cs.sum1((ddelta)))
        #This way e_smoot is always >= 0

        return e_smooth

    def e_c(self, point, theta):
        """
        Contouring error function
        Args:
            point(np.array): array containing x and y coordinates (2 x ?)
            theta(np.array): path parameter(s) (1 x ?)
        Returns:
            contouring_error(int): error 
        """
        point_r, v = self.est_ref_pos(theta)
        n = cs.hcat((v[:, 1], -v[:, 0]))
        e_c = (point_r-point)*n.T


        #e_c = cs.vec(e_c[0, :] + e_c[1, :])

        e_c = cs.dot(n.T,(point_r-point))
        

        err = point_r-point
        e_c = cs.diag(err @ n)
        e_c = e_c.T @ e_c

        return e_c

    def e_l(self, point, theta):
        """
        Longitudinal error function
        Args:
            point(np.array): array containing x and y coordinates (2 x ?)
            theta(np.array): path parameter(s) (1 x ?)
        Returns:
            lag_error(int): error 
        """
        point_r, v = self.est_ref_pos(theta)
        e_l = (point_r-point)*v.T
        #e_l = cs.vec(e_l[0, :]+e_l[1, :])
        
        e_l = cs.dot(v.T,(point_r-point))

        err = point_r-point
        e_l = cs.diag(err @ v)
        e_l = e_l.T @ e_l
        return e_l
    
    def est_ref_pos(self, theta):
        if self.trajectory is None:
            # trial circle arc length parametrisation
            x_r = r-r*cs.cos(theta/r) + cs.sin(theta/r) * (theta-self.theta_0)
            y_r = r*cs.sin(theta/r) + cs.cos(theta/r) * (theta-self.theta_0)
            point = cs.vcat((x_r.T, y_r.T))
            v = cs.hcat((cs.cos(np.pi/2-theta/r), cs.sin(np.pi/2-theta/r)))
            return point, v
        else:
            return self.trajectory.get_path_parameters(theta, self.theta_0)

class Model():
    def __init__(self, vehicle_params: dict ,MPPC_params: dict, lin_tire = False):
        '''Class implementation of the dynamic model of a small scale ground vehicle
        Args:
            param_file (str): Yaml file name of the configuration file
        '''
        # load model parameters



        self.parameters = MPPC_params
        # assign model parameters
        self.m =vehicle_params['m']
        self.I_z = vehicle_params['I_z']
        self.l_f = vehicle_params['l_f']
        self.l_r = vehicle_params['l_r']
        self.l = self.l_f+self.l_r
        self.C_m1 = vehicle_params['C_m1']
        self.C_m2 = vehicle_params['C_m2']
        self.C_m3 = vehicle_params['C_m3']


        self.nx = 4
        self.nu = 2

    def nonlin_dynamics(self, t, states, inputs):
        '''
        Class method calculating the state derivatives using nonlinear model equations
        Args:
            t(int): Current time
            states(np.array): State vector (model.nx x ?)
            inputs(np.array): Input vector (model.nu x ?)
        Returns:
            xdot(np.array): state derivates (model.nx x ?) 
        '''
        if type(states) == cs.MX:
            x = states[0, :]
            y = states[1, :]
            phi = states[2, :]
            v = states[3, :]
            d = inputs[0, :]
            delta = inputs[1, :]
        else:
            x = states[0]
            y = states[1]
            phi = states[2]
            v = states[3]
            d = inputs[0]
            delta = inputs[1]


        dx = v*cs.cos(phi)
        dy = v*cs.sin(phi)
        dphi = v/self.l*cs.tan(delta)
      
        dv = 2/self.m*(self.C_m1*d- self.C_m2*v-cs.sign(v)*self.C_m3) # vdot #USING the drivetrain model from Kristof Page 17(30/80) eq 4.6

        d_states = cs.vertcat(dx, dy, dphi, dv)
        return d_states
    
    def predict(self, states, inputs, dt, t=0, method='RK4'):
        ''' Class method predicting the next state of the model from previous state and given input
        
        Args:
            states: State vector
            inputs: Input vector
            t: Current time
            dt: Sampling time
            method: Numerical method of solving the ODEs
            
        Returns: 
            x_pred(np.array): Predicted state
            t_pred(int): Predicted time
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
