from typing import Union
from aimotion_f1tenth_simulator.classes.trajectory_base import TrajectoryBase
from aimotion_f1tenth_simulator.classes.controller_base import ControllerBase
import numpy as np
from scipy.interpolate import splrep, splprep, splev
from scipy.integrate import quad
import matplotlib.pyplot as plt



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
                k1=np.polyval([-0.156375989795226, 0, -0.174818911809753], p)
                return k1
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

        v_r=v_ref+self._K_long_outer*(s-s_ref)

        beta=np.arctan2(v_eta,abs(v_xi)) # abs() needed for reversing 

        theta_p = np.arctan2(s0[1], s0[0])

        # lateral error
        z1=np.dot(pos - ref_pos, z0)

        # heading error
        theta_e=self._normalize(phi-theta_p)

        # longitudinal model parameter
        p=abs(np.cos(theta_e+beta)/np.cos(beta)/(1-c*z1))


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
        
        d=(self.model["C_m2"]*v_r+self.model["C_m3"]*np.sign(v_ref))/self.model["C_m1"]+self._K_long(delta)*(v_xi-v_r/p)
        
        # clamp control inputs into the feasible range
        d=self._clamp(d,(0,0.25)) # currently only forward motion, TODO: reversing control
        delta=self._clamp(delta, (-.5,.5))

        # store current error & input values to be accessable from outside
        self.errors = {"lateral" : e, "heading" : theta_e, "velocity" : v_xi-v_r, "longitudinal": s-s_ref}
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