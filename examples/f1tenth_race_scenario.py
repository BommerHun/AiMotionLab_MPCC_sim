import os
import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import butter, filtfilt

from aimotion_f1tenth_simulator.classes.active_simulation import ActiveSimulator
from aimotion_f1tenth_simulator.util import xml_generator
from aimotion_f1tenth_simulator.classes.car import Car
from aimotion_f1tenth_simulator.classes.car_classes import CarTrajectory, CarMPCCController
from aimotion_f1tenth_simulator.util import mujoco_helper, carHeading2quaternion
from aimotion_f1tenth_simulator.classes.object_parser import parseMovingObjects
from aimotion_f1tenth_simulator.classes.mpcc_util.MPCC_plotter import MPCC_plotter
from aimotion_f1tenth_simulator.classes.trajectory_generators import eight, null_paperclip, null_infty, dented_paperclip, paperclip
from aimotion_f1tenth_simulator.classes.original_trajectories import race_track, hungaroring
from aimotion_f1tenth_simulator.classes.car_classes import Theta_opt
import yaml
from scipy.interpolate import splev


GUI = True # if True the simulator window will be visible, if False the simulator will run in the background 

# color definitions for multiple cars
RED_COLOR = "0.85 0.2 0.2 1.0"
BLUE_COLOR = "0.2 0.2 0.85 1.0"
BLACK_COLOR = "0.1 0.1 0.1 1.0"


abs_path = os.path.dirname(os.path.abspath(__file__))
xml_path = os.path.join(abs_path, "..", "assets")
xml_base_filename = "scene_without_walls.xml" # the base xml file that contains the necessary information for the simulation
save_filename = "built_race_scene.xml" # the name of the xml file that will be created by the XML generator

#MPCC param files
parent_dir = os.path.dirname(os.path.dirname(__file__))
file_name_1 = os.path.join(parent_dir, "examples/car_1.yaml")
with open(file_name_1) as file:
    params_1 = yaml.full_load(file)

file_name_2 = os.path.join(parent_dir, "examples/car_2.yaml")
with open(file_name_2) as file:
    params_2 = yaml.full_load(file)



# create xml with 2 f1tenth cars
scene = xml_generator.SceneXmlGenerator(xml_base_filename) # load the base scene

x0_1 = np.array([0, 0, 0.64424, 0,0,0])
x0_2 = np.array([0, 0, 0.64424, 0,0,0])


car1_name = scene.add_car(pos = f"{x0_1[0]} {x0_1[1]} 0",
                          quat = carHeading2quaternion(x0_1[2]),
                          color= RED_COLOR,
                          is_virtual=True,
                          has_rod=False)

car2_name = scene.add_car(pos = f"{x0_2[0]} {x0_2[1]} 0",
                          quat = carHeading2quaternion(x0_2[2]),
                          color= BLACK_COLOR,
                          is_virtual=True,
                          has_rod=False)


car_trajectory=CarTrajectory()

path, v = null_infty(laps=5)
car_trajectory.build_from_points_const_speed(path, path_smoothing=0.01, path_degree=4, const_speed=1.5)

theta_finder = Theta_opt(x0_1[:2], np.array([0, 5]), car_trajectory.evol_tck, car_trajectory.pos_tck)

theta0_1 = theta_finder.solve()

theta_finder = Theta_opt(x0_2[:2], np.array([0, 5]), car_trajectory.evol_tck, car_trajectory.pos_tck)

theta0_2 = theta_finder.solve()

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


car1 = simulator.get_MovingObject_by_name_in_xml(car1_name)


car2 = simulator.get_MovingObject_by_name_in_xml(car2_name)

car1_controller = CarMPCCController(vehicle_params=  params_1["parameter_server"]["ros__parameters"]["vehicle_params"],
                                    MPCC_params=  params_1["parameter_server"]["ros__parameters"]["controllers"]["MPCC"],
                                    mute = True,
                                    index = 1)

car2_controller = CarMPCCController(vehicle_params=  params_2["parameter_server"]["ros__parameters"]["vehicle_params"],
                                    MPCC_params=  params_2["parameter_server"]["ros__parameters"]["controllers"]["MPCC"],
                                    mute = True,
                                    index= 2)

car1_controllers = [car1_controller]

car1.set_trajectory(car_trajectory)

car1_controller.set_trajectory(car_trajectory.pos_tck, car_trajectory.evol_tck, x0_1, theta0_1)
car1.set_controllers(car1_controllers)


car2_controllers = [car2_controller]

car2.set_trajectory(car_trajectory)

car2_controller.set_trajectory(car_trajectory.pos_tck, car_trajectory.evol_tck, x0_2, theta0_2)
car2.set_controllers(car2_controllers)


while( not (simulator.glfw_window_should_close()) and not (car1_controller.finished == True and car2_controller.finished == True )): # the loop runs until the window is closed
    # the simulator also has an iterator that counts simualtion steps (simulator.i) and a simualtion time (simulator.time) attribute that can be used to simualte specific scenarios
    if GUI:
        simulator.update()
    else:
        simulator.update_()
  
    

