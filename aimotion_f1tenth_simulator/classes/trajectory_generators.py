import numpy as np
from aimotion_f1tenth_simulator.classes.car_classes import CarTrajectory

def quat_2_yaw(quat):
    x = quat[0]
    y = quat[1]
    z = quat[2]
    w = quat[3]
    siny_cosp = 2*(w*z + x*y)
    cosy_cosp = 1-2*(y*y + z*z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return yaw

def paperclip():
    focus_x = [0, 0]
    focus_y = [-1, 1]
    r = 2
    len_straight = focus_y[1] - focus_y[0]
    len_turn = r * np.pi
    ppm = 6
    num_straight = int(len_straight * ppm)
    num_turn = int(len_turn * ppm)
    x = np.hstack((np.linspace(focus_x[0] + r, focus_x[1] + r, num_straight),
                   focus_x[1] + r * np.cos(np.linspace(0, np.pi, num_turn)),
                   np.linspace(focus_x[1] - r, focus_x[0] - r, num_straight),
                   focus_x[0] + r * np.cos(np.linspace(np.pi, 2*np.pi, num_turn))
                   ))
    y = np.hstack((np.linspace(focus_y[0], focus_y[1], num_straight),
                   focus_y[1] + r * np.sin(np.linspace(0, np.pi, num_turn)),
                   np.linspace(focus_y[1], focus_y[0], num_straight),
                   focus_y[0] + r * np.sin(np.linspace(np.pi, 2*np.pi, num_turn))
                   ))
    x = np.roll(x, 6)
    y = np.roll(y, 6)
    points = np.array([[x_, y_] for x_, y_ in zip(x, y)])
    delete_idx = []
    for i, point in enumerate(points):
        if i > 0:
            if np.linalg.norm(point - points[i-1, :]) < 0.1:
                delete_idx += [i]
    points = np.delete(points, delete_idx, 0)
    vel = np.ones([points.shape[0],1])

    return points, vel

def dented_paperclip():
    focus_x = [0, 0]
    focus_y = [-2, 2]
    r = 2
    len_straight = focus_y[1] - focus_y[0]
    len_turn = r * np.pi
    r_dent = 0.4
    len_dent = cosine_arc_length(r_dent, 2*np.pi/len_straight, 0, len_straight)
    ppm = 4
    num_straight = int(len_straight * ppm)
    num_turn = int(len_turn * ppm)
    num_dent = int(len_dent * ppm)
    x = np.hstack((np.linspace(focus_x[1] + r, focus_x[0] + r, num_straight),
                   focus_x[1] + r * np.cos(np.linspace(0, np.pi, num_turn)),
                   -r + r_dent - r_dent * np.cos(np.linspace(0, 2*np.pi, num_dent)),
                   focus_x[0] + r * np.cos(np.linspace(np.pi, 2*np.pi, num_turn))
                   ))
    y = np.hstack((np.linspace(focus_y[0], focus_y[1], num_dent),
                   focus_y[1] + r * np.sin(np.linspace(0, np.pi, num_turn)),
                   np.linspace(focus_y[1], focus_y[0], num_straight),
                   focus_y[0] + r * np.sin(np.linspace(np.pi, 2*np.pi, num_turn))
                   ))
    x = np.roll(x, -15)
    y = np.roll(y, -15)
    points = np.array([[x_, y_] for x_, y_ in zip(x, y)])
    delete_idx = []
    for i, point in enumerate(points):
        if i > 0:
            if np.linalg.norm(point - points[i-1, :]) < 0.1:
                delete_idx += [i]
    points = np.delete(points, delete_idx, 0)
    vel = np.ones([points.shape[0],1])

    return points, vel

def cosine_arc_length(amplitude, frequency, start, end):
    # Define the derivative of the cosine function
    def derivative_cos(x):
        return -amplitude * frequency * np.sin(frequency * x)

    # Define the integrand
    def integrand(x):
        return np.sqrt(1 + derivative_cos(x) ** 2)

    # Integrate the integrand function using scipy's quad function
    from scipy.integrate import quad
    arc_length, _ = quad(integrand, start, end)

    return arc_length


def null_infty():
    points = 1.1*np.array([[0,0],
                       [1*np.cos(-np.pi/4), 1.5+np.sin(-np.pi/4)],
                        [1*np.cos(np.pi/8), 1.5+np.sin(-np.pi/8)],
                       [1, 1.5],
                       [1*np.cos(np.pi/8), 1.5+np.sin(np.pi/8)],
                       [1*np.cos(np.pi/4), 1.5+np.sin(np.pi/4)],
                        [1*np.cos(3*np.pi/8), 1.5+np.sin(3*np.pi/8)],
                       [0, 2.5],
                        [1*np.cos(5*np.pi/8), 1.5+np.sin(5*np.pi/8)],
                       [1*np.cos(6*np.pi/8), 1.5+np.sin(6*np.pi/8)],
                        [1*np.cos(7*np.pi/8), 1.5+np.sin(7*np.pi/8)],
                       [-1,1.5],
                       [-1, 1],
                       [-1, .5],
                       [-1,0],
                       [-1, -.5],
                       [-1, -1],
                       [-1,-1.5],
                       [1*np.cos(np.pi+np.pi/8), -1.5+np.sin(np.pi+np.pi/8)],
                       [1*np.cos(np.pi+np.pi/4), -1.5+np.sin(np.pi+np.pi/4)],
                        [1*np.cos(np.pi+3*np.pi/8), -1.5+np.sin(np.pi+3*np.pi/8)],
                       [0,-2.5],
                        [1*np.cos(np.pi+5*np.pi/8), -1.5+np.sin(np.pi+5*np.pi/8)],
                       [1*np.cos(np.pi+6*np.pi/8), -1.5+np.sin(np.pi+6*np.pi/8)],
                        [1*np.cos(np.pi+7*np.pi/8), -1.5+np.sin(np.pi+7*np.pi/8)],
                       [1,-1.5],
                       [1,-1],
                       [1,-0.5],
                       [1,0],
                       [1, 0.5],
                       [1, 1],
                       [1,1.5],
                        [1*np.cos(np.pi/8), 1.5+np.sin(np.pi/8)],
                       [1*np.cos(np.pi/4), 1.5+np.sin(np.pi/4)],
                        [1*np.cos(3*np.pi/8), 1.5+np.sin(3*np.pi/8)],
                       [0,2.5],
                        [1*np.cos(5*np.pi/8), 1.5+np.sin(5*np.pi/8)],
                       [1*np.cos(6*np.pi/8), 1.5+np.sin(6*np.pi/8)],
                        [1*np.cos(7*np.pi/8), 1.5+np.sin(7*np.pi/8)],
                       [-1,1.5],
                        [1*np.cos(9*np.pi/8), 1.5+np.sin(9*np.pi/8)],
                        [1*np.cos(10*np.pi/8), 1.5+np.sin(10*np.pi/8)],
                       [0,0],
                       [1*np.cos(np.pi/4), -1.5+np.sin(np.pi/4)],
                        [1*np.cos(np.pi/8), -1.5+np.sin(np.pi/8)],
                       [1,-1.5],
                       [1*np.cos(-np.pi/8), -1.5+np.sin(-np.pi/8)],
                       [1*np.cos(-np.pi/4), -1.5+np.sin(-np.pi/4)],
                        [1*np.cos(-3*np.pi/8), -1.5+np.sin(-3*np.pi/8)],
                       [0,-2.5],
                        [1*np.cos(-5*np.pi/8), -1.5+np.sin(-5*np.pi/8)],
                       [1*np.cos(-6*np.pi/8), -1.5+np.sin(-6*np.pi/8)],
                        [1*np.cos(-7*np.pi/8), -1.5+np.sin(-7*np.pi/8)],
                       [-1,-1.5],
                        [1*np.cos(-9*np.pi/8), -1.5+np.sin(-9*np.pi/8)],
                        [1*np.cos(-10*np.pi/8), -1.5+np.sin(-10*np.pi/8)],
                       [0,0]])
    
    vvelocities = np.array([.6,
                       .8,
                        .7,
                       .7,
                       .7,
                       .7,
                        .7,
                       .7,
                        .7,
                       .7,
                       .7,
                       .9,
                       1,
                       1.2,
                       1.5,
                       1.2,
                       1,
                       .9,
                       .7,
                       .7,
                        .7,
                       .7,
                        .7,
                       .7,
                        .7,
                       .9,
                       1,
                       1.2,
                       1.5,
                       1.2,
                       .9,
                       .7,
                         .7,
                        .7,
                         .7,
                       .7,
                         .7,
                        .7,
                         .7,
                       .7,
                         .7,
                        .7,
                       .7,
                       .7,
                        .7,
                        .7,
                        .7,
                       .7,
                         .7,
                      .7,
                        .7,
                       .7,
                       .7,
                       .7,
                         .7,
                       .6,
                       .6])
    return points, vvelocities

def eight():
    path_points = .5*np.flip(np.array(
    [
        [0, 0],
        [1, 1],
        [2, 2],
        [3, 2],
        [4, 1],
        [4.5, 0],
        [4, -1],
        [3, -2],
        [2, -2],
        [1, -1],
        [0, 0],
        [-1, 1],
        [-2, 2],
        [-3, 2],
        [-4, 1],
        [-4.5, 0],
        [-4, -1],
        [-3, -2],
        [-2, -2],
        [-1, -1],
        [0, 0],
    ]), axis=1)
    vel = np.ones([path_points.shape[0],1])
    return path_points, vel



def null_paperclip():
    points = 1.5*np.array([[0,0],
                       [1*np.cos(-np.pi/4), 1.5+np.sin(-np.pi/4)],
                        [1*np.cos(np.pi/8), 1.5+np.sin(-np.pi/8)],
                       [1, 1.5],
                       [1*np.cos(np.pi/8), 1.5+np.sin(np.pi/8)],
                       [1*np.cos(np.pi/4), 1.5+np.sin(np.pi/4)],
                        [1*np.cos(3*np.pi/8), 1.5+np.sin(3*np.pi/8)],
                       [0, 2.5],
                        [1*np.cos(5*np.pi/8), 1.5+np.sin(5*np.pi/8)],
                       [1*np.cos(6*np.pi/8), 1.5+np.sin(6*np.pi/8)],
                        [1*np.cos(7*np.pi/8), 1.5+np.sin(7*np.pi/8)],
                       [-1,1.5],
                       [-1, 1],
                       [-1, .5],
                       [-1,0],
                       [-1, -.5],
                       [-1, -1],
                       [-1,-1.5],
                       [1*np.cos(np.pi+np.pi/8), -1.5+np.sin(np.pi+np.pi/8)],
                       [1*np.cos(np.pi+np.pi/4), -1.5+np.sin(np.pi+np.pi/4)],
                        [1*np.cos(np.pi+3*np.pi/8), -1.5+np.sin(np.pi+3*np.pi/8)],
                       [0,-2.5],
                        [1*np.cos(np.pi+5*np.pi/8), -1.5+np.sin(np.pi+5*np.pi/8)],
                       [1*np.cos(np.pi+6*np.pi/8), -1.5+np.sin(np.pi+6*np.pi/8)],
                        [1*np.cos(np.pi+7*np.pi/8), -1.5+np.sin(np.pi+7*np.pi/8)],
                       [1,-1.5],
                       [1,-1],
                       [1,-0.5],
                       [1,0],
                       [1, 0.5],
                       [1, 1],
                       [1,1.5],
                        [1*np.cos(np.pi/8), 1.5+np.sin(np.pi/8)],
                       [1*np.cos(np.pi/4), 1.5+np.sin(np.pi/4)],
                        [1*np.cos(3*np.pi/8), 1.5+np.sin(3*np.pi/8)],
                       [0,2.5],
                        [1*np.cos(5*np.pi/8), 1.5+np.sin(5*np.pi/8)],
                       [1*np.cos(6*np.pi/8), 1.5+np.sin(6*np.pi/8)],
                        [1*np.cos(7*np.pi/8), 1.5+np.sin(7*np.pi/8)],
                       [-1,1.5],
                       [1*np.cos(9*np.pi/8), 1.5+np.sin(9*np.pi/8)],
                       [1*np.cos(10*np.pi/8), 1.5+np.sin(10*np.pi/8)],
                       [0,0]]
                       )
    vel = 1.1*np.array([.6,
                       .7,
                        .7,
                       .7,
                       .7,
                       .7,
                        .7,
                       .8,
                        .8,
                       .8,
                        .9,
                       1,
                       1.2,
                       1.3,
                       1.1,
                       .9,
                       .8,
                       .7,
                       .7,
                       .7,
                        .7,
                       .7,
                        .7,
                       .8,
                        .9,
                       1,
                       1.2,

                       1.3,
                       1.1,
                       .9,
                       .8,
                        .7,
                       .7,
                        .7,
                       .7,
                        .7,
                       .7,
                      .7,
                       .7,
                       .7,
                       .7,
                       .5]
                       )
    return points,vel
    
def train8(v=1):
    points = np.array([[0,0],
                       [0.37,0.41],
                       [1*np.cos(-np.pi/4), 1.5+np.sin(-np.pi/4)],
                        [1*np.cos(np.pi/8), 1.5+np.sin(-np.pi/8)],
                       [1, 1.5],
                       [1*np.cos(np.pi/8), 1.5+np.sin(np.pi/8)],
                       [1*np.cos(np.pi/4), 1.5+np.sin(np.pi/4)],
                        [1*np.cos(3*np.pi/8), 1.5+np.sin(3*np.pi/8)],
                       [0, 2.5],
                        [1*np.cos(5*np.pi/8), 1.5+np.sin(5*np.pi/8)],
                       [1*np.cos(6*np.pi/8), 1.5+np.sin(6*np.pi/8)],
                        [1*np.cos(7*np.pi/8), 1.5+np.sin(7*np.pi/8)],
                        [-1, 1.5],
                        [1*np.cos(9*np.pi/8), 1.5+np.sin(9*np.pi/8)],
                       [1*np.cos(10*np.pi/8), 1.5+np.sin(10*np.pi/8)],
                                              [-0.37,0.41],
                       [0,0],
                                              [0.37,-0.41],
                       [1*np.cos(np.pi/4), -1.5+np.sin(np.pi/4)],
                        [1*np.cos(np.pi/8), -1.5+np.sin(np.pi/8)],
                       [1,-1.5],
                       [1*np.cos(-np.pi/8), -1.5+np.sin(-np.pi/8)],
                       [1*np.cos(-np.pi/4), -1.5+np.sin(-np.pi/4)],
                        [1*np.cos(-3*np.pi/8), -1.5+np.sin(-3*np.pi/8)],
                       [0,-2.5],
                        [1*np.cos(-5*np.pi/8), -1.5+np.sin(-5*np.pi/8)],
                       [1*np.cos(-6*np.pi/8), -1.5+np.sin(-6*np.pi/8)],
                        [1*np.cos(-7*np.pi/8), -1.5+np.sin(-7*np.pi/8)],
                       [-1,-1.5],
                        [1*np.cos(-9*np.pi/8), -1.5+np.sin(-9*np.pi/8)],
                        [1*np.cos(-10*np.pi/8), -1.5+np.sin(-10*np.pi/8)],
                                               [-0.37,-0.41],
                       [0,0]])
    vel = v*np.ones([points.shape[0], 1])
    return points, vel


if __name__=="__main__":
    import matplotlib.pyplot as plt
    path, V = train8()
    print(path)
    plt.plot(path[:,0], path[:,1])
    plt.scatter(path[:,0], path[:,1])
    plt.axis("equal")
    vel = np.ones([path.shape[0],1])

    t = CarTrajectory("t")
    t.build_from_waypoints(path, vel, 0, 5)
    x,y,v,c1 = t.CarTrajectory()
    plt.plot(x,y)
    plt.show(block = False)
    plt.figure()
    plt.plot(c1)
    plt.show()