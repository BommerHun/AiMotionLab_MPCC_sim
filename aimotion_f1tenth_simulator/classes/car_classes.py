from typing import Union
from aimotion_f1tenth_simulator.classes.trajectory_base import TrajectoryBase
from aimotion_f1tenth_simulator.classes.controller_base import ControllerBase
import numpy as np
from scipy.interpolate import splrep, splprep, splev
from scipy.integrate import quad
import matplotlib.pyplot as plt
import scipy as cp
#MPCC includes
import casadi as cs
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
from scipy.interpolate import splev


class CarTrajectory(TrajectoryBase):
    def __init__(self) -> None:
        """Class implementation of BSpline-based trajectories for autonomous ground vehicles
        """
        super().__init__()

        self.output = {}
        self.pos_tck = None
        self.evol_tck = None
        self.t_end = None
        self.length = None



    def build_from_points_const_speed(self, path_points: np.ndarray, path_smoothing: float, path_degree: int, const_speed: float):
        """Object responsible for storing the reference trajectory data.

        Args:
            path_points (numpy.ndarray): Reference points of the trajectory
            smoothing (float): Smoothing factor used for the spline interpolation
            degree (int): Degree of the fitted Spline
        """

        x_points = path_points[:, 0].tolist()
        y_points = path_points[:, 1].tolist()

        # fit spline and evaluate
        tck, u, *_ = splprep([x_points, y_points], k=path_degree, s=path_smoothing)
        XY=splev(u, tck)

        # calculate arc length
        def integrand(x):
            dx, dy = splev(x, tck, der=1)
            return np.sqrt(dx**2 + dy**2)
        self.length, _ = quad(integrand, u[0], u[-1])

        # build spline for the path parameter
        self.pos_tck, _, *_ = splprep(XY, k=path_degree, s=0, u=np.linspace(0, self.length, len(XY[0])))

        # build constant speed profile
        self.t_end= self.length / const_speed

        t_evol = np.linspace(0, self.t_end, 101)
        s_evol = np.linspace(0, self.length, 101)

        self.evol_tck = splrep(t_evol,s_evol, k=1, s=0) # NOTE: This is completely overkill for constant veloctities but can be easily adjusted for more complex speed profiles


    def export_to_time_dependent(self):
        """Exports the trajectory to a time dependent representation
        """
        t_eval=np.linspace(0, self.t_end, 100)
        
        s=splev(t_eval, self.evol_tck)
        (x,y)=splev(s, self.pos_tck)

        tck, t, *_ = splprep([x, y], k=5, s=0, u=t_eval)

        return tck
        

    def _project_to_closest(self, pos: np.ndarray, param_estimate: float, projetion_window: float, projection_step: float) -> float:
        """Projects the vehicle position onto the ginven path and returns the path parameter.
           The path parameter is the curvilinear abscissa along the path as the Bspline that represents the path is arc length parameterized

        Args:
            pos (np.ndarray): Vehicle x,y position
            param_esimate: Estimated value of the path parameter
            projetion_window (float): Length of the projection window along the path
            projection_step (float): Precision of the projection in the window

        Returns:
            float: The path parameter
        """

        
        # calulate the endpoints of the projection window
        floored = self._clamp(param_estimate - projetion_window / 2, [0, self.length])
        ceiled = self._clamp(param_estimate + projetion_window/2, [0, self.length])


        # create a grid on the window with the given precision
        window = np.linspace(floored, ceiled, round((ceiled - floored) / projection_step))


        # evaluate the path at each instance
        path_points = np.array(splev(window, self.pos_tck)).T

        # find & return the closest points
        deltas = path_points - pos
        indx = np.argmin(np.einsum("ij,ij->i", deltas, deltas))
        return floored + indx * projection_step
    


    def evaluate(self, state, i, time, control_step) -> dict:
        """Evaluates the trajectory based on the vehicle state & time"""

        if self.pos_tck is None or self.evol_tck is None: # check if data has already been provided
            raise ValueError("Trajectory must be defined before evaluation")
        
        pos=np.array([state["pos_x"],state["pos_y"]])

        # get the path parameter (position along the path)
        s_ref=splev(time, self.evol_tck) # estimate the path parameter based on the time
        s=self._project_to_closest(pos=pos, param_estimate=s_ref, projetion_window=5, projection_step=0.005) # the projection parameters cound be refined/not hardcoded

        # check if the retrievd is satisfies the boundary constraints & the path is not completed
        if time>=self.t_end: # substraction is required because of the projection step
            self.output["running"]=False # stop the controller
        else:
            self.output["running"]= True # the goal has not been reached, evaluate the trajectory

        # get path data at the parameter s
        (x, y) = splev(s, self.pos_tck)
        (x_, y_) = splev(s, self.pos_tck, der=1)
        (x__,y__) = splev(s, self.pos_tck,der=2)
        
        # calculate base vectors of the moving coordinate frame
        s0 = np.array(
            [x_ / np.sqrt(x_**2 + y_**2), y_ / np.sqrt(x_**2 + y_**2)]
        )
        z0 = np.array(
            [-y_ / np.sqrt(x_**2 + y_**2), x_ / np.sqrt(x_**2 + y_**2)]
        )

        # calculate path curvature
        c=abs(x__*y_-x_*y__)/((x_**2+y_**2)**(3/2))

        # get speed reference
        v_ref = splev(time, self.evol_tck, der=1)

        self.output["ref_pos"]=np.array([x,y])
        self.output["s0"] = s0
        self.output["z0"] = z0
        self.output["c"] = c

        self.output["s"] = s
        self.output["s_ref"]=s_ref # might be more effective if output["s"] & self.s are combined
        self.output["v_ref"] = v_ref

        return self.output
    
    def is_finished(self) -> bool:
        """Checks if the trajectory is finished

        Returns:
            bool: True if the trajectory is finished, False otherwise
        """
        try:
            finished =  not self.output["running"]
        except KeyError:
            finished = False

        return finished
    


    @staticmethod
    def _clamp(value: Union[float,int], bound: Union[int,float,list,tuple,np.ndarray]) -> float:
        """Helper function that clamps the given value with the specified bounds

        Args:
            value (float | int): The value to clamp
            bound (list | tuple | np.ndarray): If int | float the function constrains the value into [-bound,bound]
                                               If tuple| list | np.ndarray the value is constained into the range of [bound[0],bound[1]]

        Returns:
            float: The clamped value
        """
        if isinstance(bound, int) or isinstance(bound, float):
            if value < -bound:
                return float(-bound)
            elif value > bound:
                return float(bound)
            return float(value)
        elif isinstance(bound, tuple) or isinstance(bound,list) or isinstance(bound, np.ndarray):
            if value < bound[0]:
                return float(bound[0])
            elif value > bound[1]:
                return float(bound[1])
            return float(value)
    

    @staticmethod
    def _normalize(angle: float) -> float:
        """Normalizes the given angle into the [-pi/2, pi/2] range

        Args:
            angle (float): Input angle

        Returns:
            float: Normalized angle
        """
        while angle > np.pi:
            angle -= 2*np.pi
        while angle < -np.pi:
            angle += 2*np.pi

        return angle
    
    def plot_trajectory(self, block=True) -> None:
        """ Plots, the defined path of the trajectory in the X-Y plane. Nota, that this function interrupts the main thread and the simulator!
        """

        if self.pos_tck is None or self.evol_tck is None: # check if data has already been provided
            raise ValueError("No Spline trajectory is specified!")
        
        # evaluate the path between the bounds and plot
        t_eval=np.linspace(0, self.t_end, 100)
        
        s=splev(t_eval, self.evol_tck)
        (x,y)=splev(s, self.pos_tck)
        
        fig, axs = plt.subplot_mosaic("AB;CC;DD")

        axs["A"].plot(t_eval,x)
        axs["A"].set_xlabel("t [s]")
        axs["A"].set_ylabel("x [m]")
        axs["A"].set_title("X coordinate")

        axs["B"].plot(t_eval,y)
        axs["B"].set_xlabel("t [s]")
        axs["B"].set_ylabel("y [m]")
        axs["B"].set_title("Y coordinate")

        axs["C"].plot(x,y)
        axs["C"].set_xlabel("x [m]")
        axs["C"].set_ylabel("y [m]")
        axs["C"].set_title("X-Y trajectory")
        axs["C"].axis("equal")

        axs["D"].plot(t_eval, s)
        axs["D"].set_xlabel("t [s]")
        axs["D"].set_ylabel("s [m]")
        axs["D"].set_title("Path parameter")

        plt.tight_layout()
        plt.show(block=block)



class CarLPVController(ControllerBase):



    def __init__(self, model=None, K_long=None, K_long_outer=None, K_lat=None, control_step=None):
        """Trajectory tracking LPV feedback controller, based on the decoupled longitudinal and lateral dynamics

        Args:
            model (dict): Dict containing the vehicle model parameters
            K_long (np.ndarray): Longitudinal gains
            K_lat (np.ndarray): Lateral gains
            control_step (float): Control step of the controller
            K_long_outer (np.ndarray): Outer longitudinal gains
        """

        if model is None: # default model parameters
            self.model = {"m": 2.923, # [kg]
                          "l_f": 0.163, # [m]
                          "l_r": 0.168, # [m]
                          "I_z": 0.0796, # [kg m^2]

                          # drivetrain parameters
                          "C_m1": 61.383, # [N]
                          "C_m2": 3.012, # [Ns/m]
                          "C_m3": 0.604, # [N]

                          # tire parameters
                          "C_f": 41.7372, # [N/rad]
                          "C_r": 29.4662, # [N/rad]
                          }
        else:
            self.model = model

        if K_long is None:
            def K_long(p):
                k1=np.polyval([0.0010,-0.0132,0.4243], p)
                k2=np.polyval([-0.0044, 0.0563, 0.0959], p)
                return np.array([k1, k2])
            self._K_long = K_long
        else:
            self._K_long = K_long

        if K_lat is None:
            def K_lat(p):
                k1=np.polyval([-0.00963161973193937, 0.0766311773839150, -0.171265625838336, -0.127420691079483], p)
                k2=np.polyval([0.172, -1.17, 2.59, 2.14], p)
                k3=np.polyval([0.00423, 0.0948, 0.463, 0.00936], p)
                return np.array([k1, k2, k3])
            self._K_lat = K_lat
        else:
            self._K_lat = K_lat

        if K_long_outer is None:
            self._K_long_outer = -.5
        else:
            self._K_long_outer = K_long_outer

        if control_step is None:
            self.dt = 1.0/40.0

        # define the initial value of the lateral controller integrator
        self.q=0 # lateral controller integrator 
        self.r=0 # longitudinal controller integrator

        # empty dict for storing current error and input values
        self.errors = {}
        self.input = {}



    def compute_control(self, state: dict, setpoint: dict, time, **kwargs) -> np.array:
        """Method for calculating the control input, based on the current state and setpoints

        Args:
            state (dict): Dict containing the state variables
            setpoint (dict): Setpoint determined by the trajectory object
            time (float): Current simuator time

        Returns:
            np.array: Computed control inputs [d, delta]
        """

        # check if the the trajectory exectuion is still needed
        if not setpoint["running"]:
            return np.array([0,0])

        # retrieve setpoint & state data
        s0=setpoint["s0"]
        z0=setpoint["z0"]
        ref_pos=setpoint["ref_pos"]
        c=setpoint["c"]
        s=setpoint["s"]
        s_ref=setpoint["s_ref"]
        v_ref=setpoint["v_ref"]
        

        pos=np.array([state["pos_x"], state["pos_y"]])
        phi=state["head_angle"]
        v_xi=state["long_vel"]
        v_eta=state["lat_vel"]

        #v_r=v_ref+10*self._K_long_outer*(s-s_ref)

        beta=np.arctan2(v_eta,abs(v_xi)) # abs() needed for reversing 

        theta_p = np.arctan2(s0[1], s0[0])

        # lateral error
        z1=np.dot(pos - ref_pos, z0)

        # heading error
        theta_e=self._normalize(phi-theta_p)

        # longitudinal model parameter and integrator
        p=abs(np.cos(theta_e+beta)/np.cos(beta)/(1-c*z1))
        #self.r+=v_xi-v_r


        # invert z1 for lateral dynamics:
        e=-z1
        self.q+=e
        self.q=self._clamp(self.q,0.1)


        # estimate error derivative
        try:
            self.edot=0.5*((e-self.ep)/self.dt-self.edot)+self.edot # calculate \dot{e} by finite difference
            self.ep=e                                               # 0.5 coeff if used for smoothing
        except AttributeError: # if no previous error value exist assume 0 & store the current value
            self.edot=0
            self.ep=e

        # compute control inputs
        delta=-theta_e + (self._K_lat(v_xi) @ np.array([[self.q],[e],[self.edot]])).item() \
                  - self.model["m"]/self.model["C_f"]*((self.model["l_r"]*self.model["C_r"]-self.model["l_f"]*self.model["C_f"])/self.model["m"]-1)*c
        
        d=(self.model["C_m2"]*v_ref/p+self.model["C_m3"]*np.sign(v_ref))/self.model["C_m1"]-self._K_long(p)@np.array([[s-s_ref],[v_xi-v_ref/p]])
        
        # clamp control inputs into the feasible range
        d=self._clamp(d,(0,0.25)) # currently only forward motion, TODO: reversing control
        delta=self._clamp(delta, (-.5,.5))

        # store current error & input values to be accessable from outside
        self.errors = {"lateral" : e, "heading" : theta_e, "velocity" : v_xi-v_ref, "longitudinal": s-s_ref}
        self.input = {"d" : d, "delta" : delta}

        return np.array([d, delta])

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
        return self.input


    @staticmethod
    def _clamp(value: Union[float,int], bound: Union[int,float,list,tuple,np.ndarray]) -> float:
        """Helper function that clamps the given value with the specified bounds

        Args:
            value (float | int): The value to clamp
            bound (list | tuple | np.ndarray): If int | float the function constrains the value into [-bound,bound]
                                               If tuple| list | np.ndarray the value is constained into the range of [bound[0],bound[1]]

        Returns:
            float: The clamped value
        """
        if isinstance(bound, int) or isinstance(bound, float):
            if value < -bound:
                return float(-bound)
            elif value > bound:
                return float(bound)
            return float(value)
        elif isinstance(bound, tuple) or isinstance(bound,list) or isinstance(bound, np.ndarray):
            if value < bound[0]:
                return float(bound[0])
            elif value > bound[1]:
                return float(bound[1])
            return float(value)
    

    @staticmethod
    def _normalize(angle: float) -> float:
        """Normalizes the given angle into the [-pi/2, pi/2] range

        Args:
            angle (float): Input angle

        Returns:
            float: Normalized angle
        """
        while angle > np.pi:
            angle -= 2*np.pi
        while angle < -np.pi:
            angle += 2*np.pi

        return angle
    

class Spline_2D:
    def __init__(self, points: np.array, bc_type: str = 'natural'):
        """

        :param points: Points to which the time-parametrized spline is interpolated
        :param bc_type: Type of boundary condition
        """

        self.shape = np.shape(points)
        self.original_points = points
        self.bc_type = bc_type
        self.equally_space_points = None

        self.spl_t = cp.interpolate.CubicSpline(points[:, 0], points[:, 1:], bc_type=self.bc_type)
        self.__find_arclength_par()


        self.spl_border_left = None

        self.spl_border_right = None

    def make_border_splines(self, offset = 0.1):

        i = np.linspace(0, self.L, 1000)
        points, v = self.get_path_parameters(i)

        points = points


    def __find_arclength_par(self, m: int = 30, e: float = 10**(-3), dt: float = 10**(-4)):
        """Class method that finds approximation of the arc length parametrisation of the time-parameterized spline

        :param m: Number of sections
        :param e: Error margin of bisection method
        :param dt: Precision of numerical integration
        """

        # Calculation of the total arc length
        t = np.arange(min(self.original_points[:, 0]), max(self.original_points[:, 0]) + dt, dt)
        dxydt = self.spl_t.derivative(1)(t) # calculating the derivative of the time-parametrized spline
        ds = np.sqrt(dxydt[:, 0]**2+dxydt[:, 1]**2)  # Length of arc element
        self.L = cp.integrate.simpson(y=ds, dx=dt)  # Simpsons 3/4 rule

        # Splitting the spline into m sections with length l using bisection
        self.l = self.L/m

        # Initializing bisection method
        tstart = min(self.original_points[:, 0])
        tend = max(self.original_points[:, 0])
        tmid = (tstart+tend)/2
        t_arr = np.array([0])
        self.s_arr = np.array([0])
        s_mid = 10000000

        # Solving problem with bisection
        for i in range(1, m):
            if i != 1:
                tstart = tmid
                tend = max(self.original_points[:, 0])
                tmid = (tstart + tend) / 2
                s_mid = 10000000

            while abs(s_mid-self.l) >= e:
                tmid_arr = np.arange(t_arr[-1], tmid + dt, dt)
                grad_mid = self.spl_t.derivative(1)(tmid_arr)
                ds_mid = np.sqrt(grad_mid[:, 0] ** 2 + grad_mid[:, 1] ** 2)
                s_mid = cp.integrate.simpson(y=ds_mid, dx=dt)

                if self.l < s_mid:
                    tend = tmid
                    tmid = (tend+tstart)/2
                else:
                    tstart = tmid
                    tmid = (tend + tstart) / 2
            self.s_arr = np.append(self.s_arr, s_mid+i*self.l)
            t_arr = np.append(t_arr, tmid)

        self.s_arr = np.reshape(self.s_arr, (-1, 1))

        self.equally_space_points = np.concatenate((self.s_arr, self.spl_t(t_arr)), 1)  # array that contains the new points
        if (self.original_points[0, 1:] == self.original_points[-1, 1:]).all():
            self.equally_space_points = np.concatenate((self.equally_space_points, [[self.L+self.l, self.original_points[-1, 1], self.original_points[-1, 2]]]))
            #print(self.equally_space_points)

        self.spl_sx = cs.interpolant('n', 'bspline', [self.equally_space_points[:, 0]], self.equally_space_points[:, 1])  # fitting casadi spline to the x coordinate
        self.spl_sy = cs.interpolant('n', 'bspline', [self.equally_space_points[:, 0]], self.equally_space_points[:, 2])  # fitting casadi spline to the y coordinate
        self.spl_s = cp.interpolate.CubicSpline(self.equally_space_points[:, 0], self.equally_space_points[:, 1:], bc_type=self.bc_type)

    def get_path_parameters_ang(self, theta: cs.MX):
        """ Class method that returns the symbolic path parameters needed to calculate the lag and contouring error
            Path parameters using angles
        :param theta: path parameter (s)
        :return: x, y coordinate and the tangent angle
        """

        x = self.spl_sx(theta)
        y = self.spl_sy(theta)

        jac_x = self.spl_sx.jacobian()
        jac_y = self.spl_sy.jacobian()
        phi = cs.arctan2(jac_y(theta, theta)+0.001, (jac_x(theta, theta)+0.001))

        return x, y, phi

    def get_path_parameters(self, theta, theta_0=None):
        """
        Path parameters using vectors
        :param theta:
        :param theta_0:
        :return:
        """
        point = cs.hcat((self.spl_sx(theta), self.spl_sy(theta))).T

        jac_x = self.spl_sx.jacobian()
        jac_y = self.spl_sy.jacobian()

        v = cs.hcat((jac_x(theta, theta), jac_y(theta, theta)))  # unit direction vector
        #l = v**2
        #v = cs.hcat((v[:, 0]/cs.sqrt(l[:,0]+l[:,1]), v[:, 1]/cs.sqrt(l[:,0]+l[:,1])))#cs.hcat((cs.sqrt(l[:, 0]+l[: 1]), cs.sqrt(l[:, 0]+l[: 1])))
        return point, v

    def get_path_parameters_lin(self, theta, theta_0):
        """
        Path parameters using first order Taylor
        :param theta:
        :param theta_0:
        :return:
        """
        x_0 = self.spl_sx(theta_0)
        y_0 = self.spl_sy(theta_0)

        jac_x = self.spl_sx.jacobian()
        jac_x = jac_x(theta_0, theta_0)
        jac_y = self.spl_sy.jacobian()
        jac_y = jac_y(theta_0, theta_0)

        x_lin = x_0 + jac_x * (theta - theta_0)
        y_lin = y_0 + jac_y * (theta - theta_0)

        point = cs.hcat((x_lin, y_lin)).T
        v = cs.hcat((jac_x, jac_y))/cs.sqrt(jac_x**2+jac_y**2)

        return point, v

    def get_path_parameters_lin2(self, theta, theta_0, point_0, jac_0):
        """
        Path parameters using first order Taylor
        :param theta:
        :param theta_0:
        :return:
        """
        point = point_0 + jac_0 * cs.vcat(((theta - theta_0.T).T, (theta - theta_0.T).T))
        v = (cs.vcat((jac_0[0, :], jac_0[1, :]))/cs.vcat((cs.sqrt(jac_0[0, :]**2+jac_0[1, :]**2), cs.sqrt(jac_0[0, :]**2+jac_0[1, :]**2)))).T

        return point, v

    def get_path_parameters_quad(self, theta, theta_0):
        """
        Path parameters using second order Taylor
        :param theta:
        :param theta_0:
        :return:
        """
        x_0 = self.spl_sx(theta_0)
        y_0 = self.spl_sy(theta_0)

        jac_x = self.spl_sx.jacobian()
        jac2_x = jac_x.jacobian()
        jac_x = jac_x(theta_0, theta_0)
        jac2_x = jac2_x(theta_0, theta_0, theta_0)[:,1]

        jac_y = self.spl_sy.jacobian()
        jac2_y = jac_y.jacobian()
        jac_y = jac_y(theta_0, theta_0)
        jac2_y = jac2_y(theta_0, theta_0, theta_0)[:,1]

        x_lin = x_0 + jac_x * (theta - theta_0) + jac2_x/2 * (theta - theta_0)**2
        y_lin = y_0 + jac_y * (theta - theta_0) + jac2_y/2 * (theta - theta_0)**2

        point = cs.hcat((x_lin, y_lin)).T

        jac_x_lin = jac_x + jac2_x * (theta - theta_0)
        jac_y_lin = jac_y + jac2_y * (theta - theta_0)

        v = cs.hcat((jac_x_lin, jac_y_lin))
        l = v ** 2
        v = cs.hcat((v[:, 0]/cs.sqrt(l[:,0]+l[:,1]), v[:, 1]/cs.sqrt(l[:,0]+l[:,1])))
        return point, v
    

    def e_c(self, point, theta):
            """
            Contouring error function
            :param point: array containing x and y coordinates
            :param theta: path parameter(s)
            :return: contouring error
            """
            point_r, v =  self.get_path_parameters(theta, 0) #point: vertical, v: horizontal
            n = cs.hcat((v[:, 1], -v[:, 0])) #Creating a perpendicular vector
            #e_c = n*(point_r-point) #Why does this return a 2*1 vector???? format: [value, 0]
            e_c = cs.dot(n.T,(point_r-point))

            #print(f"contouring error: {e_c}")
            #e_c = vec(e_c[0, :] + e_c[1, :]) #old code
            return e_c

    def e_l(self, point, theta):
        """
        Lag error function
        :param point: array containing x and y coordinates
        :param theta: path parameter(s)
        :return: lag error
        """
        point_r, v = self.get_path_parameters(theta, 0)
        #e_l = v*(point_r-point) #Why does this return a 2*1 vector???? format: [value, 0]
        e_l = cs.dot(v.T,(point_r-point))
        #print(f"lateral error: {e_l}")
        #e_l = vec(e_l[0, :]+e_l[1, :]) #old code
        return e_l


r= 1
class Casadi_MPCC:
    def __init__(self,MPCC_params:dict,input_0, vehicle_param:dict, dt: float, q_c: float, q_l: float, q_t: float,q_delta:float, q_d: float, N: int, x_0, theta_0:float, trajectory: Spline_2D):
        """
        Args:
            q_c (float): Contouring error weight
            q_l (float): Lag error weight
            q_t (float): Progress weight
            N (int): Control horizon
            dt (float): Sampling time
        """
        self.model = Model(vehicle_params=vehicle_param, MPPC_params=MPCC_params)
        self.nx = self.model.nx # = 6
        self.nu = self.model.nu # = 2
        
        self.dt = dt
        self.q_c = q_c
        self.q_l = q_l
        self.q_t = q_t
        self.q_delta = q_delta #The coeficience of the smoothness of the control
        self.q_d =q_d
        self.N = N
        self.trajectory = trajectory

        # Variable horizons for initial guess
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

        self.v_t_init = np.zeros(N-1) #Store the initial guess for v_theta

        
        self.lam_g_init = 0 #What does this do?

        # Time measurement variables
        self.time_proc = np.array([])
        self.time_wall = np.array([])
        self.iter_count = np.array([])

        # Initialise optimizer
        self.__init_optimizer()

    def opti_step(self, x_0):
        """
        Method that calculates the optimization solution for given initial values
        :param x_0: initial state
        :return: optimal input, path parameter
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
        return sol.value(self.X), sol.value(self.v_U),sol.value(self.U), sol.value(self.theta[:]), sol.value(self.v_t)
    

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
        self.X_0 = self.opti.parameter(6, 1) #The value is set at every call of the optistep(set to the current x[k])
        self.theta_0 = self.opti.parameter(1) #The value is set at every call of optistep  to theta_init[0]
        self.v_t_0 = self.opti.parameter(1) # -||- to v_t_init[0]

        self.U_0 = self.opti.parameter(2,1)
        self.v_U_0 = self.opti.parameter(2,1)

        
        # Set constraints for initial values#

        self.opti.subject_to(self.X[:, 0] == self.X_0)  # initial state

        self.opti.subject_to(self.v_t[0] == self.v_t_0)  # initial path speed

        self.opti.subject_to(self.theta[0] == self.theta_0 + self.dt * self.v_t_0)  # init path parameter

        #self.opti.subject_to(self.U[:,0] == self.U_0) ##Initial input constraint: dd and ddelta

        self.opti.subject_to(self.v_U[:,0] == self.v_U_0+ self.dt*self.U_0) #init virtual input (d & delta)


        

        # Dynamic constraint #

        x_pred = self.model.predict(states=self.X[:,:-1], inputs=self.v_U, dt=self.dt)[0] 
        # x[k] = model.predict(Self.X[k-1], self.v_U[k-1]) -> arrays are sent as arguments-> array is returned
        #Size of the arguments: 1) self.X[:,:-1]: 6xN 2) self.v_U: 2xN-> returns 6xN new states (note that they are shifted by 1 timespamp)


        self.opti.subject_to(self.X[:, 1:] == x_pred) #Settings the constraint


        theta_next = self.theta[:-1] + self.dt * self.v_t #Theta[k+1] = Theta[k]+dt*v_t
        self.opti.subject_to(self.theta[1:] == theta_next) #Settings constraint


        v_U_next = self.v_U[:,:-1]+self.dt*self.U #v_U[k+1] = ... (basically the same)
        #v_U: is a 2xN array & U is a 2x(N-1) array

        self.opti.subject_to(self.v_U[:,1:] == v_U_next) #Constraint :)

        self.opti.subject_to((self.theta[-1]) <= self.trajectory.L)


        
        # State constraints
        #self.opti.subject_to((0 < cs.vec(self.X[3, 1:]), cs.vec(self.X[3, 1:]) <= 5.5))  # Xi speed component constraint
        #self.opti.subject_to((-3 <= cs.vec(self.X[4, 1:]), cs.vec(self.X[4, 1:]) <= 3))  # Eta speed component constraint
        #self.opti.subject_to((-5 <= cs.vec(self.X[5, 1:]), cs.vec(self.X[5, 1:]) <= 5))  # Angular speed constraint
        
        

        # Virtual Input constraints: d and delta
        #self.opti.subject_to((0.0 < cs.vec(self.v_U[0, 1:]), cs.vec(self.v_U[0, 1:]) <= 1))  # motor reference constraints
        self.opti.subject_to((self.model.parameters["d_min"] <= cs.vec(self.v_U[0, 1:]), cs.vec(self.v_U[0, 1:]) <= self.model.parameters["d_max"]))  # motor reference constraints

        self.opti.subject_to((-self.model.parameters["delta_max"] <= cs.vec(self.v_U[1, 1:]), cs.vec(self.v_U[1, 1:]) <= self.model.parameters["delta_max"] ))  # steering angle constraints

         
        # Input constraints: derivate of d and delta
        self.opti.subject_to((-self.model.parameters["ddot_max"] <= cs.vec(self.U[0, :]), cs.vec(self.U[0, :]) <= self.model.parameters["ddot_max"]))  # motor reference constraints
        self.opti.subject_to((-self.model.parameters["deltadot_max"] <= cs.vec(self.U[1, :]), cs.vec(self.U[1, :]) <= self.model.parameters["deltadot_max"]))  # steering angle constraints
        
        """
        self.opti.subject_to((self.model.parameters["d_min"] < cs.vec(self.v_U[0, 1:]), cs.vec(self.v_U[0, 1:]) <= self.model.parameters["d_max"]))  # motor reference constraints
        self.opti.subject_to((-self.model.parameters["delta_max"]<= cs.vec(self.v_U[1, 1:]), cs.vec(self.v_U[1, 1:]) <= self.model.parameters["delta_max"] ))  # steering angle constraints

         
        # Input constraints: derivate of d and delta
        self.opti.subject_to((-self.model.parameters["ddot_max"] <= cs.vec(self.U[0, :]), cs.vec(self.U[0, :]) <= self.model.parameters["ddot_max"]))  # motor reference constraints
        self.opti.subject_to((-self.model.parameters["deltadot_max"]<= cs.vec(self.U[1, :]), cs.vec(self.U[1, :]) <= self.model.parameters["deltadot_max"]))  # steering angle constraints
        """

        # Path speed constraints
        self.opti.subject_to((self.model.parameters["thetahatdot_min"] <= cs.vec(self.v_t[1:]), cs.vec(self.v_t[:]) <= self.model.parameters["thetahatdot_max"]))

        # Set objective function
        self.opti.minimize(self.cost())

        # Solver setup
        p_opt = {'expand': False}
        s_opts = {'max_iter': 2000, 'print_level': 0}
        self.opti.solver('ipopt', p_opt, s_opts)
   


    def cost(self):
        """
        Method which returns the cost
        :return: cost
        """
        e_l = self.e_l(cs.vcat((self.X[0, :], self.X[1, :])), self.theta)
        e_c = self.e_c(cs.vcat((self.X[0, :], self.X[1, :])), self.theta)
        
        e_smooth = self.e_smooth()[0]*self.q_d + self.e_smooth()[1]*self.q_delta
        
        cost = self.q_l * e_l.T @ e_l + self.q_c * e_c.T @ e_c- self.q_t * (self.v_t.T @ self.v_t) + e_smooth 
        #cost = self.q_l * e_l + self.q_c * e_c- self.q_t * (self.trajectory.L-self.theta[-1]) + self.q_smooth  * e_smooth

        return cost

    def e_smooth(self):
        """
        This function puts the to the rows of the input (dd and ddelta) above each other in a vector and returns it
        """

        d = cs.vec(self.U[0,:])

        delta = cs.vec(self.U[1,:])


        e_smooth = np.array([d.T @ d, delta.T @ delta])  #Basically I calculate the lenght^2 of the the d and the delta vectors and add them together
        #This way e_smoot is always >= 0

        return e_smooth

    def e_c(self, point, theta):
        """
        Contouring error function
        :param point: array containing x and y coordinates
        :param theta: path parameter(s)
        :return: contouring error
        """
        point_r, v = self.est_ref_pos(theta)
        n = cs.hcat((v[:, 1], -v[:, 0]))
        e_c = (point_r-point)*n.T
        e_c = cs.vec(e_c[0, :] + e_c[1, :])
        #e_c = e_c.T @ e_c
        return e_c

    def e_l(self, point, theta):
        """
        Lag error function
        :param point: array containing x and y coordinates
        :param theta: path parameter(s)
        :return: lag error
        """
        point_r, v = self.est_ref_pos(theta)
        e_l = (point_r-point)*v.T
        e_l = cs.vec(e_l[0, :]+e_l[1, :])
        #e_l = e_l.T @ e_l
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
        

class Model:
    def __init__(self, vehicle_params: dict ,MPPC_params: dict):
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

        self.C_m1 = vehicle_params['C_m1']
        self.C_m2 = vehicle_params['C_m2']
        self.C_m3 = vehicle_params['C_m3']

        self.C_r = vehicle_params['C_r']
        self.C_f = vehicle_params['C_f']

        self.nx = 6
        self.nu = 2

        # constraints


    def nonlin_dynamics(self, t, states, inputs):
        '''
        Class method calculating the state derivatives using nonlinear model equations
        :param t: Current time
        :param states: State vector
        :param inputs: Input vector
        :return: Derivative of states
        '''
        if type(states) == cs.MX:
            x = states[0, :]
            y = states[1, :]
            phi = states[2, :]
            v_xi = states[3, :]
            v_eta = states[4, :]
            omega = states[5, :]
            d = inputs[0, :]
            delta = inputs[1, :]
        else:
            x = states[0]
            y = states[1]
            phi = states[2]
            v_xi = states[3]
            v_eta = states[4]
            omega = states[5]
            d = inputs[0]
            delta = inputs[1]

        # slip angles
        alpha_r = cs.arctan((-v_eta + self.l_r*omega)/(v_xi+0.001))
        alpha_f = delta - cs.arctan((v_eta + self.l_f * omega)/(v_xi+0.001))

        # tire forces
        F_xi = self.C_m1*d - self.C_m2*v_xi - self.C_m3*cs.sign(v_xi)
        F_reta = self.C_r*alpha_r
        F_feta = self.C_f*alpha_f

        # nonlinear state equations
        dx = v_xi * cs.cos(phi) - v_eta * cs.sin(phi)
        dy = v_xi * cs.sin(phi) + v_eta * cs.cos(phi)
        dphi = omega

        dv_xi = 1 / self.m * (F_xi + F_xi * cs.cos(delta) - F_feta * cs.sin(delta) + self.m * v_eta * omega)
        dv_eta = 1 / self.m * (F_reta + F_xi * cs.sin(delta) + F_feta * cs.cos(delta) - self.m * v_xi * omega)
        domega = 1 / self.I_z * (F_feta * self.l_f * cs.cos(delta) + F_xi * self.l_f * cs.sin(delta) - F_reta * self.l_r)
        d_states = cs.vertcat(dx, dy, dphi, dv_xi, dv_eta, domega)
        return d_states

    def predict(self, states, inputs, dt, t=0, method='RK4'):
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


class CarMPCCController(ControllerBase):
    def __init__(self, vehicle_params: dict, MPCC_params: dict, mute = False):
        """
        Init controller parameters
        :param vehicle_params: dict
        :param MPCC_params: dict
        """
        self.muted = mute
        self.vehicle_params = vehicle_params
        self.MPCC_params = MPCC_params
        self.trajectory = None

        self.theta = 0.0
        self.s_start = 0.0
        self.x0 = np.zeros((1,6))

        self.input = np.array([self.MPCC_params["d_max"],0])
       
        self.ocp_solver = None #acados solver to compute control

        self.casadi_solver = None #casadi_solver for initial guess

        self.errors = {"lateral" : 0, "heading" : 0, "velocity" : 0, "longitudinal": float(0)}
        self.finished = False
        self.load_parameters()

    def load_parameters(self):
        """
        Load self parameters from the dict-s
        """
        self.parameters = cs.types.SimpleNamespace()

        m = float(self.vehicle_params["m"])
        l_f = float(self.vehicle_params["l_f"])
        l_r = float(self.vehicle_params["l_r"])
        I_z = float(self.vehicle_params["I_z"])

        C_m1 = float(self.vehicle_params["C_m1"])
        C_m2 = float(self.vehicle_params["C_m2"])
        C_m3 = float(self.vehicle_params["C_m3"])

        C_f = float(self.vehicle_params["C_f"])
        C_r = float(self.vehicle_params["C_r"])


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
        if self.theta >= self.trajectory.L:
            errors = np.array([0.0, 0.0,0.0,float(self.theta), 0.0])
            u_opt = np.array([0,0])
            self.input = np.array([0,0])
            self.errors = {"lateral" : 0, "heading" : self.theta, "velocity" : 0, "longitudinal": float(0)}

            self.finished = True
            return u_opt
        if x0[3] <0.001:
            x0[3] = 0.001

        x0 = np.concatenate((x0, np.array([self.theta]), self.input))

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
            if t > 1/60: #If the controller frequency is below 60 Hz break
                if self.muted == False:
                    print("\nTime limit reached\n")
                #break

        x_opt = np.reshape(self.ocp_solver.get(1, "x"),(-1,1)) #Full predictied optimal state vector (x,y,phi, vxi, veta, omega, thetahat, d, delta)
        self.theta = x_opt[6,0]
        self.input = x_opt[7:, 0]
        u_opt = np.reshape(self.ocp_solver.get(0, "x"),(-1,1))[7:,0]
        if (1/t < self.MPCC_params["freq_limit"]) or (max(res) > self.MPCC_params["res_limit"]):
            raise Exception(f"Slow computing, emergency shut down, current freq: {1/t}, residuals: {res}")
        if self.muted == False:
            print(f"\rFrequency: {(1/(t)):4f}, solver time: {t:.5f}, QP iterations: {num_iter:2}, progress: {self.theta/self.trajectory.L*100:.2f}%, input: {u_opt}, residuals: {res}               \r", end = '', flush=True)
        
        for i in range(self.parameters.N-1):
            self.ocp_solver.set(i, "x", self.ocp_solver.get(i+1, "x"))

        for i in range(self.parameters.N-2):
            self.ocp_solver.set(i, "u", self.ocp_solver.get(i+1, "u"))


        e_con = self.trajectory.e_c(x_opt[:2,0], self.theta)[0][0]

        e_long = self.trajectory.e_l(x_opt[:2,0],self.theta)[0][0]

        
        errors = np.array([float(e_con), float(x_opt[2,0]),float(e_long),float(self.theta), float(x_opt[3,0])])


        self.errors = {"lateral" : float(e_con), "heading" : float(self.theta), "velocity" : float(x_opt[3,0]), "longitudinal": float(e_long)}
        
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
    
    def controller_init(self):
        """
        Calculate intial guess for the state horizon.
        """
        if self.muted == False:
            print("Casadi init started...")

        X, v_U,U, theta, dtheta = self.casadi_solver.opti_step(self.x0) #Call casadi solver for optimal initial guess

        x_0 = np.concatenate((self.x0, np.array([self.theta]), v_U[:,0]))
        self.ocp_solver.set(0, "x", x_0)
        u_0 = np.concatenate((U[:,0], np.array([dtheta[0]])))
        self.ocp_solver.set(0, "u", u_0)

    
        if self.muted == False:
            print(f"x0: {x_0}")
            print(f"u0: {u_0}")

        states = np.array(np.reshape(x_0, (-1,1)))

        if self.muted == False: 
            print(f"___________{0}._________")
            print(u_0)
            print(x_0)

        for i in range(self.parameters.N-1):
            x = np.concatenate((X[:,i+1],np.array([theta[i+1]]), v_U[:,i+1]))
            #x = np.reshape(x, (-1,1))
            states = np.append(states, np.reshape(x,(-1,1)), axis = -1)
            self.ocp_solver.set(i+1, "x", x)
            if dtheta[i] <self.MPCC_params["thetahatdot_min"]:
                dtheta[i] = self.MPCC_params["thetahatdot_min"]
            u = np.concatenate((np.array([dtheta[i]]),U[:,i]))
            self.ocp_solver.set(i, "u", u)
            if self.muted == False:
                print(f"___________{i+1}._________")
                print(u)
                print(x)
        #Acados controller init: 
        if self.muted == False:
            print("Acados init started...")

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
        if self.muted == False:
            print(f"Number of init iterations: {num_iter}")
            print("")

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
        alpha_r = cs.arctan2((-veta+l_r*omega),(vxi+0.0001)) #In the documentation arctan2 is used but vxi can't be < 0
        alpha_f = delta- cs.arctan2((veta+l_f*omega),(vxi+0.0001))

        #Wheel forces

        Fxi = C_m1*d-C_m2*vxi-vxi*C_m3

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
        F_tire = Fxi**2/self.MPCC_params["mu_xi"]+Freta**2/self.MPCC_params["mu_eta"]

        model.con_h_expr = cs.vertcat(F_tire)

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
        cost = e_c**2*self.parameters.q_con+e_l**2*self.parameters.q_lat-thetahatdot*self.parameters.q_theta+self.parameters.q_d*ddot**2+self.parameters.q_delta*deltadot**2
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
        pass

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
        ocp.code_export_directory = 'c_generated_code'
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
        ocp.constraints.lh = np.array([-1])
        ocp.constraints.uh = np.array([1])

        ocp.cost.zl = np.array([0.1])  # lower slack penalty
        ocp.cost.zu = np.array([0.1])  # upper slack penalty
        ocp.cost.Zl = np.array([0.5])  # lower slack weight
        ocp.cost.Zu = np.array([0.5])  # upper slack weight

        ## Initialize slack variables for lower and upper bounds
        ocp.constraints.lsh = np.zeros(1)
        ocp.constraints.ush = np.zeros(1)
        ocp.constraints.idxsh = np.arange(1)
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

        ocp_solver = AcadosOcpSolver(ocp, json_file = 'acados_ocp.json')
        return ocp_solver
    

    def set_trajectory(self, pos_tck, evol_tck, x0, theta_start):
        """
        Evaluetes the reference spline from the given spline tck, and converts it into a Spline2D instance
        :param pos_tck: array
        :param evol_tck: array, not used
        :param x0: initial state, used for initialising the controller
        :param thetastart: float, starting arc lenght of the trajectory
        """

        self.load_parameters()

        self.theta = theta_start
        self.s_start = theta_start
        
        self.x0 = x0 #The current position must be the initial condition

        self.x0[3] = 0.01 #Give a small forward speed to make the problem feasable
        self.x0[5] = 0
        self.x0[4] = 0

        self.input = np.array([self.MPCC_params["d_max"],0])


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
        print(f"initial state: {self.x0}")

        print(f"starting point: {self.trajectory.get_path_parameters_ang(self.theta)}")

        self.ocp_solver = self._generate_ocp_solver(self._generate_model())
        self.casadi_solver = Casadi_MPCC(MPCC_params=self.MPCC_params,
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
        self.controller_init()


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
        alpha_r = cs.arctan((-v_eta + self.parameters.l_r*omega)/(v_xi+0.001))
        alpha_f = delta - cs.arctan((v_eta + self.parameters.l_f * omega)/(v_xi+0.001))

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
