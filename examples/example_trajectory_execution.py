import os
import numpy as np
import matplotlib.pyplot as plt

from aimotion_f1tenth_simulator.classes.active_simulation import ActiveSimulator
from aimotion_f1tenth_simulator.util import xml_generator
from aimotion_f1tenth_simulator.classes.car import Car
from aimotion_f1tenth_simulator.classes.car_classes import CarTrajectory, CarLPVController
from aimotion_f1tenth_simulator.util import mujoco_helper, carHeading2quaternion
from aimotion_f1tenth_simulator.classes.object_parser import parseMovingObjects

GUI = False # if True the simulator window will be visible, if False the simulator will run in the background 

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
mass = "2.5"
inertia = ".05 .05 .078"
wheel_width = ".022225"
wheel_radius = ".072388"


# create xml with a car
scene = xml_generator.SceneXmlGenerator(xml_base_filename) # load the base scene
car0_name = scene.add_car(pos="0 0 0.052",
                          quat=carHeading2quaternion(0.64424),
                          color=RED_COLOR,
                          is_virtual=True,
                          has_rod=False,)
                          #friction=friction,
                          #mass=mass,
                          #inertia=inertia,
                          #wheel_radius=wheel_radius) # add the car to the scene
 

# saving the scene as xml so that the simulator can load it
scene.save_xml(os.path.join(xml_path, save_filename))

# create list of parsers
virt_parsers = [parseMovingObjects]



control_step, graphics_step = 0.025, 0.025 # the car controller operates in 40 Hz by default
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
#car0.set_drivetrain_parameters(C_m1=60, C_m2=2, C_m3=0.5) # if not specified the default values will be used 



# create a trajectory
car0_trajectory=CarTrajectory()

# define path points and build the path
path_points = 8*np.array(
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
        [-4, -2.1],
        [-3, -2.3],
        [-2, -2],
        [-1, -1],
        [0, 0],
    ]
)

# the biult in trajectory generator fits 2D splines onto the given coordinates and generates the trajectory with contstant reference velocity
car0_trajectory.build_from_points_const_speed(path_points=path_points, path_smoothing=0.01, path_degree=4, const_speed=1.5)
car0_trajectory.plot_trajectory() # this is a blocking method close the plot to proceed


car0_controller = CarLPVController() # init controller


# add the controller to a list and define an update method: this is useful in case of multiple controllers and controller switching
car0_controllers = [car0_controller]
def update_controller_type(state, setpoint, time, i):
    return 0



# setting update_controller_type method, trajectory and controller for car0
car0.set_update_controller_type_method(update_controller_type)
car0.set_trajectory(car0_trajectory)
car0.set_controllers(car0_controllers)


# start simulation and collect position data
x = []
y = []
lateral_errors = []
longitudinal_errors = []
d = []
delta = []
t = []


while not simulator.glfw_window_should_close(): # the loop runs until the window is closed
    # the simulator also has an iterator that counts simualtion steps (simulator.i) and a simualtion time (simulator.time) attribute that can be used to simualte specific scenarios
    if GUI:
        simulator.update()
    else:
        simulator.update_()

    st=car0.get_state() # states corresponding to a dynamic single track representation
    x.append(st["pos_x"])
    y.append(st["pos_y"])

    # get errors
    errors = car0_controller.get_errors()
    lateral_errors.append(errors["lateral"])
    longitudinal_errors.append(errors["longitudinal"])


    # get control inputs
    inputs= car0_controller.get_inputs()
    d.append(inputs["d"])
    delta.append(inputs["delta"])

    # get time
    t.append(simulator.i*simulator.control_step)

    if car0_trajectory.is_finished():
        break
    
simulator.close()

# plot simulation results
plt.figure()
plt.plot(x,y)
plt.axis('equal')
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.title("Trajectory")
plt.show(block=False)

fig, axs = plt.subplots(2, 1)
axs[0].plot(t,delta)
axs[0].set_ylabel("Steering angle (rad)")
axs[0].set_xlabel("Time (s)")
axs[1].plot(t,d)
axs[1].set_ylabel("Motor reference (1)")
axs[1].set_xlabel("Time (s)")
plt.show(block=False)

fix, axs = plt.subplots(2, 1)
axs[0].plot(t,lateral_errors)
axs[0].set_ylabel("Lateral error (m)")
axs[0].set_xlabel("Time (s)")
axs[1].plot(t,longitudinal_errors)
axs[1].set_ylabel("Longitudinal error (m)")
axs[1].set_xlabel("Time (s)")
plt.show(block=True)