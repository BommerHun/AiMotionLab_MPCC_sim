import os
import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import butter, filtfilt

from aimotion_f1tenth_simulator.classes.active_simulation import ActiveSimulator
from aimotion_f1tenth_simulator.util import xml_generator
from aimotion_f1tenth_simulator.classes.car import Car
from aimotion_f1tenth_simulator.classes.traj_classes import CarTrajectory
from aimotion_f1tenth_simulator.classes.car_classes import CarMPCCController
from aimotion_f1tenth_simulator.util import mujoco_helper, carHeading2quaternion
from aimotion_f1tenth_simulator.classes.object_parser import parseMovingObjects
from aimotion_f1tenth_simulator.classes.mpcc_util.MPCC_plotter import MPCC_plotter
from aimotion_f1tenth_simulator.classes.trajectory_generators import eight, null_paperclip, null_infty, dented_paperclip, paperclip, slalom
from aimotion_f1tenth_simulator.classes.mpcc_util.complex_trajectories import paperclip_forward, paperclip_backward
from aimotion_f1tenth_simulator.classes.original_trajectories import race_track, hungaroring
import yaml
from scipy.interpolate import splev
from aimotion_f1tenth_simulator.classes.mpcc_util.mpcc_reverse import mpcc_reverse_controller
from matplotlib.collections import LineCollection

reversed = True
GUI = True # if True the simulator window will be visible, if False the simulator will run in the background 
trajectory = "null_infinity"

if trajectory == "null_infinity" or trajectory == "null_paperclip":
    x0 = np.array([0.1, 0.1,0.64424,0,0,0])
    x0[2] = x0[2]+np.pi
elif trajectory =="slalom":
    x0 = np.array([0.0, 0.0,-np.pi,0,0,0])



# color definitions for multiple cars
RED_COLOR = "0.85 0.2 0.2 1.0"
BLUE_COLOR = "0.2 0.2 0.85 1.0"
BLACK_COLOR = "0.1 0.1 0.1 1.0"



# the XML generator will create the scene for the simulation. All the XML files and the dependencies are located
# in the same folder (e.g. assets). The path to this folder needs to be specified here
abs_path = os.path.dirname(os.path.abspath(__file__))
xml_path = os.path.join(abs_path, "..", "assets")
xml_base_filename = "scene_without_walls.xml" # the base xml file that contains the necessary information for the simulation
save_filename = "built_scene.xml" # the name of the xml file that will be created by the XML generator


#MPCC param file
parent_dir = os.path.dirname(os.path.dirname(__file__))
file_name = os.path.join(parent_dir, "examples/reverse_config.yaml")
with open(file_name) as file:
    params = yaml.full_load(file)


args = {}
args["vehicle_params"] = params["parameter_server"]["ros__parameters"]["vehicle_params"]
args["MPCC_params"] = params["parameter_server"]["ros__parameters"]["controllers"]["MPCC"]
args["drive_bridge"] = params["parameter_server"]["ros__parameters"]["drive_bridge"]
args["crazy_observer"] = params["parameter_server"]["ros__parameters"]["crazy_observer"]
MOTOR_LIMIT = args["drive_bridge"]["MOTOR_LIMIT"]


# parameters for the car can be specified by passing keyword arguments to the add_car method -> these will remain constant during the simulation 
# if not specified the default values will be used identified in the AIMoitonLab area
friction =" 2.5 2.5 .009 .0001 .0001"
mass = "2.9"
inertia = ".05 .05 .078"
wheel_width = ".022225"
wheel_radius = ".072388"


# create xml with a car
scene = xml_generator.SceneXmlGenerator(xml_base_filename) # load the base scene



#Adding the f1tenth vehicle
car0_name = scene.add_car(pos=f"{x0[0]} {x0[1]} 0",
                          quat=carHeading2quaternion(x0[2]),
                          color=RED_COLOR,
                          is_virtual=True,
                          has_rod=False,)
                          #friction=friction,
                          #mass=mass,
                          #inertia=inertia,
                          #wheel_radius=wheel_radius) # add the car to the scene

#MPCC horizon markers
horizon_markers = scene.add_MPCC_markers(args["MPCC_params"]["N"], BLUE_COLOR, f"{x0[0]} {x0[1]} 0", quat = carHeading2quaternion(x0[2]), size=0.1)

# create a trajectory
car0_trajectory=CarTrajectory()


if trajectory == "slalom":
    path, v = slalom(loops=4, r = 2)
elif trajectory == "null_paperclip":
    path, v = null_paperclip()
elif trajectory  == "null_infinity":
    path,  v = null_infty()
car0_trajectory.build_from_points_const_speed(path, path_smoothing=0.01, path_degree=4, const_speed=1.5)

#Reference trajectory points:
t_end = car0_trajectory.evol_tck[0][-1]

        
t_eval=np.linspace(0, t_end, 100)
s=splev(t_eval, car0_trajectory.evol_tck)

(x,y) = splev(s, car0_trajectory.pos_tck)

ref_traj_markers = scene.add_trajectory_markers(x,y,BLACK_COLOR, size = 0.05)




# saving the scene as xml so that the simulator can load it
scene.save_xml(os.path.join(xml_path, save_filename))

# create list of parsers
virt_parsers = [parseMovingObjects]



control_step, graphics_step = 1/40, 1/20 # the car controller operates in 40 Hz by default
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







if not reversed:
    car0_controller = CarMPCCController(vehicle_params= args["vehicle_params"], mute = False, MPCC_params= args["MPCC_params"])
else:
    car0_controller = mpcc_reverse_controller(vehicle_params= args["vehicle_params"], MPCC_params= args["MPCC_params"])
# add the controller to a list and define an update method: this is useful in case of multiple controllers and controller switching
car0_controllers = [car0_controller]

errors = np.array(np.zeros((4,1)))

def update_controller_type(state, setpoint, time, i):
    return 0


# setting update_controller_type method, trajectory and controller for car0
car0.set_update_controller_type_method(update_controller_type)
car0.set_trajectory(car0_trajectory)

car0_controller.set_trajectory(pos_tck = car0_trajectory.pos_tck,
                                evol_tck=  car0_trajectory.evol_tck,
                                generate_solver= True)
car0.set_controllers(car0_controllers)



car0_controller.init_controller(x0=x0)

#input()
# start simulation and collect position data
x = []
y = []
v_xi = []
v_eta = []
phi = []
omega = []
contouring_errors = []
longitudinal_errors = []
theta = []
theta_dot = []
d = []
delta = []
t = []
c_t = []

st=car0.get_state()

phi_prev = st["head_angle"]


u_sim = np.zeros((2,1))


while( not (simulator.glfw_window_should_close()) and (car0_controller.finished == False )): # the loop runs until the window is closed
    # the simulator also has an iterator that counts simualtion steps (simulator.i) and a simualtion time (simulator.time) attribute that can be used to simualte specific scenarios
    if GUI:
        simulator.update()
    else:
        simulator.update_()
  
    x.append(car0_controller.prev_state[0])
    y.append(car0_controller.prev_state[1])
    phi.append(car0_controller.prev_state[2])
    v_xi.append(np.abs(car0_controller.prev_state[3]))
    theta.append(car0_controller.theta)
    d.append(car0_controller.input[0])
    delta.append(car0_controller.input[1])
    c_t.append(car0_controller.c_t)

    
    u_sim = np.append(u_sim, np.reshape(car0_controller.input,(-1,1)), axis = 1)
    error = car0_controller.get_errors()
    error = np.array([error["contouring"], error["longitudinal"], error["progress"], error["c_t"]])
    error = np.reshape(error, (-1,1))
    errors = np.append(errors, error, axis = 1)
    #update horizon plotter and mujoco markers
    for i in range(car0_controller.parameters.N):
        x_temp   = car0_controller.ocp_solver.get(i, 'x')
        x_temp = np.reshape(x_temp, (-1,1))
        
        
        try:
            id = simulator.model.body(f"mpcc_{i}").id
            
            id = simulator.model.body_mocapid[id]
            simulator.data.mocap_pos[id] = np.concatenate((x_temp[:2, 0], np.array([0])))

        except Exception as e:
            print(e)



simulator.close()


#Creating simulation result plots
s = np.linspace(0, car0_controller.trajectory.L,1000)



plt.figure()
plt.plot(car0_controller.trajectory.spl_sx(s), car0_controller.trajectory.spl_sy(s))
plt.xlabel("x[m]")
plt.ylabel("y[m]")
plt.axis("equal")
plt.grid(True)



plt.figure()
plt.plot(car0_controller.trajectory.spl_sx(s), car0_controller.trajectory.spl_sy(s))
points = np.array([x, y]).T.reshape(-1,1,2)
segments = np.concatenate([points[:-1], points[1:]], axis = 1)
plt.grid(True)
norm = plt.Normalize(vmin =0, vmax=3.5)
lc = LineCollection(segments=segments, cmap = "turbo", norm=norm)
lc.set_array(v_xi)
lc.set_linewidth(2)

plt.gca().add_collection(lc)
plt.xlabel("x[m]")
plt.ylabel("y[m]")
cbar = plt.colorbar(lc, label = "$v_{\\xi}$ [m/s]")
plt.axis('equal')



plt.xlabel("x[m]")
plt.ylabel("y[m]")


fig, axs = plt.subplots(2,1, figsize = (10,6))

axs[0].title.set_text("Computing time historgram")
axs[0].hist(np.array(c_t)*1000)
#axs[0].axvline(x = car0_controller.MPCC_params["Tf"]/car0_controller.MPCC_params["N"]*1000, color = 'r', label = 'sampling time [ms]')
#axs[0].legend()
axs[0].set_xlabel("Computing time [ms]")
axs[0].set_ylabel("Number of iterations [-]")
axs[0].set_xlim(left = 0)

axs[1].title.set_text("Computing time")
axs[1].set_xlabel("Iteration [-]")
axs[1].set_ylabel("Computing time [ms]")
axs[1].plot(np.arange(np.shape(c_t)[0]),np.array(c_t)*1000 , label = "computing time [ms]")
#axs[1].axhline(y = car0_controller.MPCC_params["Tf"]/car0_controller.MPCC_params["N"]*1000, color = 'r', label = 'sampling time [ms]')
#axs[1].legend()

plt.tight_layout()


fig, axs = plt.subplots(3,1)



axs[0].title.set_text("Motor reference")
axs[0].set_xlabel("Iteration [-]")
axs[0].set_ylabel("d [-]")
axs[0].plot(np.arange(np.shape(u_sim[0,1:-1])[0]),u_sim[0,1:-1] )


axs[1].title.set_text("Steering servo reference")
axs[1].set_xlabel("Iteration [-]")
axs[1].set_ylabel("$\\delta$ [-]")
axs[1].plot(np.arange(np.shape(u_sim[1,1:-1])[0]),u_sim[1,1:-1] )


axs[2].title.set_text("Errors")
axs[2].set_xlabel("Iteration [-]")
axs[2].set_ylabel("errors [m]")
axs[2].plot(np.arange(np.shape(errors[0,1:-1])[0]),errors[0,1:-1], label = '$e_c$ [m]')
axs[2].plot(np.arange(np.shape(errors[1,1:-1])[0]),errors[1,1:-1], label = '$e_l$ [m]')
axs[2].legend()
for ax in axs:
    ax.grid(True)

plt.tight_layout()
plt.ion()
plt.show()

input("Press enter to close")
