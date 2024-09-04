from aimotion_f1tenth_simulator.classes.trajectory_base import TrajectoryBase
import casadi as cs
import numpy as np
from scipy.interpolate import splprep, splev, splrep
from typing import Union
from scipy.integrate import quad
import scipy as cp

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
        plt.figure()

        
        plt.tight_layout()
        plt.show(block=block)


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

