import os
import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import butter, filtfilt

from aimotion_f1tenth_simulator.classes.active_simulation import ActiveSimulator
from aimotion_f1tenth_simulator.util import xml_generator
from aimotion_f1tenth_simulator.classes.car import Car
from aimotion_f1tenth_simulator.classes.car_classes import CarTrajectory, CarLPVController, CarMPCCController
from aimotion_f1tenth_simulator.util import mujoco_helper, carHeading2quaternion
from aimotion_f1tenth_simulator.classes.object_parser import parseMovingObjects
from aimotion_f1tenth_simulator.classes.MPCC_plotter import MPCC_plotter
from aimotion_f1tenth_simulator.classes.trajectory_generators import eight, null_paperclip, null_infty, dented_paperclip, paperclip
from aimotion_f1tenth_simulator.classes.original_trajectories import race_track
import yaml


alpha = 1
omega_filt = 0
v_eta_filt = 0
GUI = True # if True the simulator window will be visible, if False the simulator will run in the background 

# color definitions for multiple cars
RED_COLOR = "0.85 0.2 0.2 1.0"



# the XML generator will create the scene for the simulation. All the XML files and the dependencies are located
# in the same folder (e.g. assets). The path to this folder needs to be specified here
abs_path = os.path.dirname(os.path.abspath(__file__))
xml_path = os.path.join(abs_path, "..", "assets")
xml_base_filename = "scene_without_walls.xml" # the base xml file that contains the necessary information for the simulation
save_filename = "built_scene.xml" # the name of the xml file that will be created by the XML generator


# parameters for the car can be specified by passing keyword arguments to the add_car method -> these will remain constant during the simulation 
# if not specified the default values will be used identified in the AIMoitonLab area
friction =" 2.5 2.5 .009 .0001 .0001"
mass = "2.9"
inertia = ".05 .05 .078"
wheel_width = ".022225"
wheel_radius = ".072388"


# create xml with a car
scene = xml_generator.SceneXmlGenerator(xml_base_filename) # load the base scene
#heading: 0.64424
car0_name = scene.add_car(pos="0 0 0",
                          quat=carHeading2quaternion(0.64424),
                          color=RED_COLOR,
                          is_virtual=True,
                          has_rod=False,)
                          #friction=friction,
                          #mass=mass,
                          #inertia=inertia,
                          #wheel_radius=wheel_radius) # add the car to the scene




x0 = np.array([0, 0,0.64424,0,0,0])
# saving the scene as xml so that the simulator can load it
scene.save_xml(os.path.join(xml_path, save_filename))

# create list of parsers
virt_parsers = [parseMovingObjects]



control_step, graphics_step = 1/20, 1/20 # the car controller operates in 40 Hz by default
xml_filename = os.path.join(xml_path, save_filename)


# recording interval for automatic video capture can be specified by a list
#rec_interval=[1,25] # start recording at 1s, stop at 25s
rec_interval = None # no video capture

# initializing simulator
simulator = ActiveSimulator(xml_filename, rec_interval, control_step, graphics_step)

# ONLY for recording: the camera angles can be adjusted by the following commands
#simulator.activeCam
#simulator.activeCam.distance=9
#simulator.activeCam.azimuth=230

# grabbing the car
car0 = simulator.get_MovingObject_by_name_in_xml(car0_name)

# additional modeling opportunities: the drivetrain parameters can be adjusted
#car0.set_drivetrain_parameters(C_m1=40, C_m2=3, C_m3=0.5) # if not specified the default values will be used 
#car0.set_steering_parameters(offset=0.3, gain=1) # if not specified the default values will be used


# create a trajectory
car0_trajectory=CarTrajectory()


path, v = null_infty()
car0_trajectory.build_from_points_const_speed(path*1.2, path_smoothing=0.01, path_degree=4, const_speed=1.5)
# the biult in trajectory generator fits 2D splines onto the given coordinates and generates the trajectory with contstant reference velocity
#car0_trajectory.build_from_points_const_speed(path_points=path_points, path_smoothing=0.01, path_degree=4, const_speed=1.5)
#car0_trajectory.plot_trajectory() # this is a blocking method close the plot to proceed

#MPCC modifications:


#car0_controller = CarLPVController() # init controller
parent_dir = os.path.dirname(os.path.dirname(__file__))

file_name = os.path.join(parent_dir, "examples/Simulator_config.yaml")
with open(file_name) as file:
    params = yaml.full_load(file)

args = {}

args["vehicle_params"] = params["parameter_server"]["ros__parameters"]["vehicle_params"]
args["MPCC_params"] = params["parameter_server"]["ros__parameters"]["controllers"]["MPCC"]
args["drive_bridge"] = params["parameter_server"]["ros__parameters"]["drive_bridge"]
args["crazy_observer"] = params["parameter_server"]["ros__parameters"]["crazy_observer"]
MOTOR_LIMIT = args["drive_bridge"]["MOTOR_LIMIT"]


car0_controller = CarMPCCController(vehicle_params= args["vehicle_params"], mute = False, MPCC_params= args["MPCC_params"])

# add the controller to a list and define an update method: this is useful in case of multiple controllers and controller switching
car0_controllers = [car0_controller]
def update_controller_type(state, setpoint, time, i):
    return 0


# setting update_controller_type method, trajectory and controller for car0
car0.set_update_controller_type_method(update_controller_type)
car0.set_trajectory(car0_trajectory)

car0_controller.set_trajectory(car0_trajectory.pos_tck, car0_trajectory.evol_tck, x0, 0.05)
car0.set_controllers(car0_controllers)


#Setting up the horizon plotter:


plotter = MPCC_plotter()

s = np.linspace(0, car0_controller.trajectory.L,10000)



plotter.set_ref_traj(np.array(car0_controller.trajectory.spl_sx(s)), np.array(car0_controller.trajectory.spl_sy(s)))

plotter.show()


# start simulation and collect position data
x = []
y = []
v_xi = []
v_eta = []
phi = []
omega = []
lateral_errors = []
longitudinal_errors = []
theta = []
theta_dot = []
d = []
delta = []
t = []
freq = []

st=car0.get_state()

phi_prev = st["head_angle"]

din_sim_data = {}

din_sim_data["x"] = [x0[0]]
din_sim_data["y"] = [x0[1]]
din_sim_data["phi"] = [x0[2]]
din_sim_data["v_xi"] = [x0[3]]
din_sim_data["v_eta"] = [x0[4]]
din_sim_data["omega"] = [x0[5]]

#din_sim_data["x"] = []
#din_sim_data["y"] = []
#din_sim_data["phi"] = []
#din_sim_data["v_xi"] = []
#din_sim_data["v_eta"] = []
#din_sim_data["omega"] = []


while( not (simulator.glfw_window_should_close()) & (car0_controller.finished == False)): # the loop runs until the window is closed
    # the simulator also has an iterator that counts simualtion steps (simulator.i) and a simualtion time (simulator.time) attribute that can be used to simualte specific scenarios
    if GUI:
        simulator.update()
    else:
        simulator.update_()
    """
    st=car0.get_state() # states corresponding to a dynamic single track representation
    
    omega_filt = st["yaw_rate"]*alpha   + (1-alpha)*omega_filt
    v_eta_filt = st["lat_vel"]*alpha   + (1-alpha)*v_eta_filt
    st["yaw_rate"] = omega_filt
    st["lat_vel"] = v_eta_filt

    phi_cur = st["head_angle"]

    while phi_cur-phi_prev > np.pi:
        phi_cur = phi_cur-2*np.pi
    while phi_cur-phi_prev < -np.pi:
        phi_cur = phi_cur+2*np.pi
    

    st["head_angle"] = phi_cur
    phi_prev = phi_cur
    x.append(st["pos_x"])
    y.append(st["pos_y"])
    v_xi.append(st["long_vel"])
    v_eta.append(st["lat_vel"])
    phi.append(st["head_angle"])
    omega.append(st["yaw_rate"])
    """
    
    st = car0_controller.prev_state

    x_ref, y_ref = (car0_controller.trajectory.spl_sx(car0_controller.theta),car0_controller.trajectory.spl_sy(car0_controller.theta))
    plotter.set_ref_point(np.array(float(x_ref)), np.array(float(y_ref)))

    (x0, args) = car0_controller.simulate(states = st, inputs = car0_controller.input, dt = car0_controller.MPCC_params["Tf"]/ car0_controller.MPCC_params["N"])

    x.append(st[0])
    y.append(st[1])
    phi.append(st[2])
    v_xi.append(st[3])
    v_eta.append(st[4])
    omega.append(st[5])
    theta.append(car0_controller.theta)
    theta_dot.append(car0_controller.theta_dot)


    din_sim_data["x"].append(x0[0])
    din_sim_data["y"].append(x0[1])
    din_sim_data["phi"].append(x0[2])
    din_sim_data["v_xi"].append(x0[3])
    din_sim_data["v_eta"].append(x0[4])
    din_sim_data["omega"].append(x0[5])

    # get errors
    errors = car0_controller.get_errors()
    lateral_errors.append(errors["lateral"])
    longitudinal_errors.append(errors["longitudinal"])
    
    freq.append(car0_controller.freq)

    # get control inputs
    inputs= car0_controller.get_inputs()
    d.append(inputs["d"])
    delta.append(inputs["delta"])

    # get time
    t.append(simulator.i*simulator.control_step)

    if car0_trajectory.is_finished() or car0_controller.finished == True:
        break

    
    #update horizon plotter
    horizon = np.array(np.reshape(car0_controller.ocp_solver.get(0, 'x'),(-1,1)))
    for i in range(car0_controller.parameters.N-1):
        x_temp   = car0_controller.ocp_solver.get(i+1, 'x')
        x_temp = np.reshape(x_temp, (-1,1))
        horizon = np.append(horizon, x_temp, axis = 1)
    plotter.update_plot(new_x = horizon[0,:], new_y = horizon[1,:])
    #input("Press enter to continue")


    

simulator.close()

# plot simulation results
plt.figure()
plt.plot(np.array(car0_controller.trajectory.spl_sx(s)), np.array(car0_controller.trajectory.spl_sy(s)), "b")
plt.plot(x,y, "r")
plt.plot(din_sim_data["x"], din_sim_data["y"], "g")
plt.axis('equal')
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.title("Trajectory")
plt.show(block=False)



fix, axs = plt.subplots(2, 1)
axs[0].plot(t,lateral_errors)
axs[0].set_ylabel("Lateral error (m)")
axs[0].set_xlabel("Time (s)")
axs[1].plot(t,longitudinal_errors)
axs[1].set_ylabel("Longitudinal error (m)")
axs[1].set_xlabel("Time (s)")


fix, axs = plt.subplots(5, 2)

axs[0, 0].set_title("x")
axs[0][0].plot(t, x)
axs[0][0].plot(t, din_sim_data["x"][:-1], "g")

axs[1, 0].set_title("y")
axs[1][0].plot(t, y)
axs[1][0].plot(t, din_sim_data["y"][:-1], "g")

axs[2, 0].set_title("phi")
axs[2][0].plot(t, phi)
axs[2][0].plot(t, din_sim_data["phi"][:-1],"g")

axs[0, 1].set_title("v_xi")
axs[0][1].plot(t, v_xi)
axs[0][1].plot(t, din_sim_data["v_xi"][:-1],"g")

axs[1, 1].set_title("v_eta")
axs[1][1].plot(t, v_eta)
axs[1][1].plot(t, din_sim_data["v_eta"][:-1],"g")

axs[2, 1].set_title("omega")
axs[2][1].plot(t, omega)
axs[2][1].plot(t,din_sim_data["omega"][:-1],"g")

axs[3, 0].set_title("d")
axs[3][0].plot(t, d)


axs[3, 1].set_title("delta")
axs[3][1].plot(t, delta)


axs[4, 0].set_title("theta")
axs[4][0].plot(t, theta)

axs[4, 1].set_title("theta_dot")
axs[4][1].plot(t, theta_dot)



plt.figure()

plt.plot(t, freq)
plt.ylim([0,600])
plt.title("Computing frequency")
plt.ylabel("frequency (Hz)")
plt.xlabel("time (s)")
plt.ion()
plt.show()

input("Press enter to close")