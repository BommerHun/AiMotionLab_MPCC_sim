import numpy as np
import casadi as cs
from scipy.interpolate import splev
from aimotion_f1tenth_simulator.classes.traj_classes import Spline_2D
r = 1

class Theta_opt:
    def __init__(self, point: np.array, window: np.array, trajectory: Spline_2D):
        self.opti = None
        self.point = point
        self.window = window
        self.trajectory = trajectory
        self.__init_optimiser()
        
    def __init_optimiser(self):
        self.opti = cs.Opti()
        self.X = self.opti.parameter(2)
        self.theta_bound = self.opti.parameter(2)
        
        self.theta = self.opti.variable(1)

        self.opti.set_value(self.X[0], self.point[0])
        self.opti.set_value(self.X[1], self.point[1])
        self.opti.set_value(self.theta_bound[0], self.window[0])
        self.opti.set_value(self.theta_bound[1], self.window[1])


        self.opti.subject_to((self.theta >= self.theta_bound[0], self.theta<= self.theta_bound[1]))
        

        self.opti.minimize(self.cost())
        p_opt = {'expand': False}
        s_opts = {'max_iter': 2000, 'print_level': 0}
        self.opti.solver('ipopt', p_opt, s_opts)

    def solve(self):
        self.opti.solve()

        return (self.opti.value(self.theta))


    def cost(self):
        """
        Method which returns the cost
        :return: cost
        """
        e_l = self.e_l(cs.vcat((self.X[0], self.X[1])), self.theta)
        e_c = self.e_c(cs.vcat((self.X[0], self.X[1])), self.theta)
        
        

        cost = e_l**2 + e_c**2 
        #cost = self.q_l * e_l + self.q_c * e_c- self.q_t * (self.trajectory.L-self.theta[-1]) + self.q_smooth  * e_smooth

        return cost
    
    def est_ref_pos(self, theta):
        if self.trajectory is None:
            # trial circle arc length parametrisation
            x_r = r-r*cs.cos(theta/r) + cs.sin(theta/r) * (theta-self.window[0])
            y_r = r*cs.sin(theta/r) + cs.cos(theta/r) * (theta-self.window[0])
            point = cs.vcat((x_r.T, y_r.T))
            v = cs.hcat((cs.cos(np.pi/2-theta/r), cs.sin(np.pi/2-theta/r)))
            return point, v
        else:
            return self.trajectory.get_path_parameters(theta, self.window[0])
        


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


        #e_c = cs.vec(e_c[0, :] + e_c[1, :])

        e_c = cs.dot(n.T,(point_r-point))

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
        #e_l = cs.vec(e_l[0, :]+e_l[1, :])
        
        e_l = cs.dot(v.T,(point_r-point))

        #e_l = e_l.T @ e_l
        return e_l
    

    def set_trajectory(self, evol_tck, pos_tck):
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

