import os
import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import butter, filtfilt

from aimotion_f1tenth_simulator.classes.active_simulation import ActiveSimulator
from aimotion_f1tenth_simulator.util import xml_generator
from aimotion_f1tenth_simulator.classes.car import Car
from aimotion_f1tenth_simulator.classes.traj_classes import CarTrajectory
from aimotion_f1tenth_simulator.classes.car_classes import CarMPCCController
from aimotion_f1tenth_simulator.classes.mpcc_util.mpcc_reverse import mpcc_reverse_controller
from aimotion_f1tenth_simulator.util import mujoco_helper, carHeading2quaternion
from aimotion_f1tenth_simulator.classes.object_parser import parseMovingObjects
from aimotion_f1tenth_simulator.classes.mpcc_util.MPCC_plotter import MPCC_plotter
from aimotion_f1tenth_simulator.classes.mpcc_util.complex_trajectories import paperclip_forward, paperclip_backward
from aimotion_f1tenth_simulator.classes.trajectory_generators import null_paperclip
import yaml
from scipy.interpolate import splev
from matplotlib.collections import LineCollection


# color definitions for multiple cars
RED_COLOR = "0.85 0.2 0.2 1.0"
BLUE_COLOR = "0.2 0.2 0.85 1.0"
BLACK_COLOR = "0.1 0.1 0.1 1.0"
GREEN_COLOR = "0.1 0.85 0.1 1.0"
ORANGE_COLOR = "0.8 0.5 0.1 1.0"

friction =" 2.5 2.5 .009 .0001 .0001"
mass = "2.9"
inertia = ".05 .05 .078"
wheel_width = ".022225"
wheel_radius = ".072388"

GUI = True # if True the simulator window will be visible, if False the simulator will run in the background 
r = 0.75



forward_MPCC_param_file = "Simulator_config.yaml"
reverse_MPCC_param_file = "reverse_config.yaml"

def main():
    

    # the XML generator will create the scene for the simulation. All the XML files and the dependencies are located
    # in the same folder (e.g. assets). The path to this folder needs to be specified here
    abs_path = os.path.dirname(os.path.abspath(__file__))
    xml_path = os.path.join(abs_path, "..", "assets")
    xml_base_filename = "scene_without_walls.xml" # the base xml file that contains the necessary information for the simulation
    save_filename = "built_scene.xml" # the name of the xml file that will be created by the XML generator


    #MPCC param file
    args_forward = get_param_args(forward_MPCC_param_file)

    args_reverse = get_param_args(reverse_MPCC_param_file)

 
    # create xml with a car
    scene = xml_generator.SceneXmlGenerator(xml_base_filename) # load the base scene


    #Adding the f1tenth vehicle
    x0 = np.array([0.1,0, np.pi/4, 0,0,0])

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
    horizon_markers = scene.add_MPCC_markers(args_forward["MPCC_params"]["N"], BLUE_COLOR, "0 0 0", quat = carHeading2quaternion(0.64424), size=0.1)


    # create a trajectory

    car0_trajectory_reverse=CarTrajectory()
    path, v = paperclip_backward(r = 1.5)
    car0_trajectory_reverse.build_from_points_const_speed(path, path_smoothing=0.01, path_degree=4, const_speed=1.5)


    car0_trajectory_forward=CarTrajectory()
    path, v = paperclip_forward(r = 1.5)
    car0_trajectory_forward.build_from_points_const_speed(path, path_smoothing=0.01, path_degree=4, const_speed=1.5)

    #Reference trajectory points:
    
    forward_trajectory_markers = add_trajectory_markers(scene= scene,
                                                        trajectory= car0_trajectory_forward,
                                                        color = GREEN_COLOR,
                                                        size = 0.05)
    reverse_trajectory_markers = add_trajectory_markers(scene= scene,
                                                        trajectory= car0_trajectory_reverse,
                                                        color = ORANGE_COLOR,
                                                        size = 0.05)
    
    # saving the scene as xml so that the simulator can load it
    scene.save_xml(os.path.join(xml_path, save_filename))

    # create list of parsers
    virt_parsers = [parseMovingObjects]



    control_step, graphics_step = args_forward["MPCC_params"]["Tf"]/ args_forward["MPCC_params"]["N"], args_forward["MPCC_params"]["Tf"]/ args_forward["MPCC_params"]["N"] # the car controller operates in 40 Hz by default
    xml_filename = os.path.join(xml_path, save_filename)


    # recording interval for automatic video capture can be specified by a list
    #rec_interval=[1,25] # start recording at 1s, stop at 25s
    rec_interval = None # no video capture

    # initializing simulator
    simulator = ActiveSimulator(xml_filename, rec_interval, control_step, graphics_step)
  

    # grabbing the car
    car0 = simulator.get_MovingObject_by_name_in_xml(car0_name)
    car0.set_drivetrain_parameters(C_m1 = args_forward["vehicle_params"]["C_m1"], C_m2 = args_forward["vehicle_params"]["C_m2"], C_m3 = args_forward["vehicle_params"]["C_m3"])

    # additional modeling opportunities: the drivetrain parameters can be adjusted
    car0.set_drivetrain_parameters(C_m1=40, C_m2=3, C_m3=0.5) # if not specified the default values will be used 





    car0_controller_forward = CarMPCCController(vehicle_params= args_forward["vehicle_params"], mute = False, MPCC_params= args_forward["MPCC_params"])
    car0_controller_reverse = mpcc_reverse_controller(vehicle_params= args_reverse["vehicle_params"], MPCC_params= args_reverse["MPCC_params"])

    car0_controller_forward.set_trajectory(car0_trajectory_forward.pos_tck, car0_trajectory_forward.evol_tck)
    car0_controller_reverse.set_trajectory(car0_trajectory_reverse.pos_tck, car0_trajectory_reverse.evol_tck, True)



    
    # add the controller to a list and define an update method: this is useful in case of multiple controllers and controller switching
    def update_controller_type(state, setpoint, time, i):
        return 0


    # setting update_controller_type method, trajectory and controller for car0
    car0.set_update_controller_type_method(update_controller_type)
    car0.set_trajectory(car0_trajectory_forward)

    
    car0_controllers = [car0_controller_forward]
    car0.set_controllers(car0_controllers)

    car0_controller_forward.init_controller(x0)

    #Setting up the horizon plotter:


    plotter = MPCC_plotter()
    s_forw = np.linspace(0, car0_controller_forward.trajectory.L,10000)



    plotter.set_ref_traj(np.array(car0_controller_forward.trajectory.spl_sx(s_forw)), np.array(car0_controller_forward.trajectory.spl_sy(s_forw)))

    #plotter.show()

    x_ref, y_ref = (car0_controller_forward.trajectory.spl_sx(car0_controller_forward.theta),car0_controller_forward.trajectory.spl_sy(car0_controller_forward.theta))
    plotter.set_ref_point(np.array(float(x_ref)), np.array(float(y_ref)))
    #update horizon plotter
    horizon = np.array(np.reshape(car0_controller_forward.ocp_solver.get(0, 'x'),(-1,1)))
    for i in range(car0_controller_forward.parameters.N):
        x_temp   = car0_controller_forward.ocp_solver.get(i, 'x')
        x_temp = np.reshape(x_temp, (-1,1))
        horizon = np.append(horizon, x_temp, axis = 1)
    plotter.update_plot(new_x = horizon[0,:], new_y = horizon[1,:])


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
    freq = []

    st=car0.get_state()

   
    u_sim = np.zeros((2,1))
    errors = np.array(np.zeros((4,1)))


    while( not (simulator.glfw_window_should_close()) and not (car0_controller_forward.finished == True and car0.get_state()["long_vel"] < 0.001)): # the loop runs until the window is closed
        # the simulator also has an iterator that counts simualtion steps (simulator.i) and a simualtion time (simulator.time) attribute that can be used to simualte specific scenarios
        if GUI:
            simulator.update()
        else:
            simulator.update_()
    
        u_sim = np.append(u_sim, np.reshape(car0_controller_forward.input, (-1,1)), axis = 1)
        error = car0_controller_forward.get_errors()
        error = np.array([error["contouring"], error["longitudinal"], error["progress"], error["c_t"]])
        error = np.reshape(error, (-1,1))
        errors = np.append(errors, error, axis = 1)

        st = car0_controller_forward.prev_state
        (x_ref, y_ref) = splev(car0_controller_forward.theta, car0_trajectory_forward.pos_tck)
        plotter.set_ref_point(x_ref, y_ref)


        x.append(st[0])
        y.append(st[1])
        phi.append(st[2])
        v_xi.append(st[3])
        v_eta.append(st[4])
        omega.append(st[5])
        theta.append(car0_controller_forward.theta)
        theta_dot.append(car0_controller_forward.theta_dot)


    

        freq.append(car0_controller_forward.c_t)

        # get control inputs
        inputs= car0_controller_forward.get_inputs()
        d.append(inputs["d"])
        delta.append(inputs["delta"])

        # get time
        t.append(simulator.i*simulator.control_step)

        
        #update horizon plotter and mujoco markers
        horizon = np.array(np.reshape(car0_controller_forward.ocp_solver.get(0, 'x'),(-1,1)))
        for i in range(car0_controller_forward.parameters.N):
            x_temp   = car0_controller_forward.ocp_solver.get(i, 'x')
            x_temp = np.reshape(x_temp, (-1,1))
            horizon = np.append(horizon, x_temp, axis = 1)

            try:
                id = simulator.model.body(f"mpcc_{i}").id

                id = simulator.model.body_mocapid[id]
                simulator.data.mocap_pos[id] = np.concatenate((x_temp[:2, 0], np.array([0])))

            except Exception as e:
                print(e)

        plotter.update_plot(new_x = horizon[0,:], new_y = horizon[1,:])
        
       

    car0_controllers = [car0_controller_reverse]
    car0.set_controllers(car0_controllers)
    car0_controller_reverse.init_controller(np.array([x[-1], y[-1], phi[-1], 0,0,0]))

    while( not (simulator.glfw_window_should_close()) and not (car0_controller_reverse.finished == True and abs(car0.get_state()["long_vel"] < 0.001))): # the loop runs until the window is closed
        if GUI:
            simulator.update()
        else:
            simulator.update_()

        #log data: 
        x.append(car0_controller_reverse.prev_state[0])
        y.append(car0_controller_reverse.prev_state[1])
        phi.append(car0_controller_reverse.prev_state[2])
        v_xi.append(np.abs(car0_controller_reverse.prev_state[3]))
        theta.append(car0_controller_reverse.theta)
        d.append(car0_controller_reverse.input[0])
        delta.append(car0_controller_reverse.input[1])
        t.append(car0_controller_reverse.c_t)


        u_sim = np.append(u_sim, np.reshape(car0_controller_reverse.input,(-1,1)), axis = 1)
        error = car0_controller_reverse.get_errors()
        error = np.array([error["contouring"], error["longitudinal"], error["progress"], error["c_t"]])
        error = np.reshape(error, (-1,1))
        errors = np.append(errors, error, axis = 1)
        #update horizon plotter and mujoco markers
        horizon = np.array(np.reshape(car0_controller_reverse.ocp_solver.get(0, 'x'),(-1,1)))
        for i in range(car0_controller_reverse.parameters.N):
            x_temp   = car0_controller_reverse.ocp_solver.get(i, 'x')
            x_temp = np.reshape(x_temp, (-1,1))
            horizon = np.append(horizon, x_temp, axis = 1)

            try:
                id = simulator.model.body(f"mpcc_{i}").id

                id = simulator.model.body_mocapid[id]
                simulator.data.mocap_pos[id] = np.concatenate((x_temp[:2, 0], np.array([0])))

            except Exception as e:
                print(e)

        plotter.update_plot(new_x = horizon[0,:], new_y = horizon[1,:])


    simulator.close()

    #Creating simulation result plots
    s_forw = np.linspace(0, car0_controller_forward.trajectory.L,1000)
    s_backw = np.linspace(0, car0_controller_reverse.trajectory.L,1000)
    plt.figure()
    plt.plot(car0_controller_forward.trajectory.spl_sx(s_forw), car0_controller_forward.trajectory.spl_sy(s_forw))
    plt.axis('equal')
    plt.xlabel("x[m]")
    plt.ylabel("y[m]")
    plt.grid(True)


    plt.figure()
    plt.plot(car0_controller_forward.trajectory.spl_sx(s_forw), car0_controller_forward.trajectory.spl_sy(s_forw))
    plt.plot(car0_controller_reverse.trajectory.spl_sx(s_backw), car0_controller_reverse.trajectory.spl_sy(s_backw))

    points = np.array([x, y]).T.reshape(-1,1,2)

    segments = np.concatenate([points[:-1], points[1:]], axis = 1)

    norm = plt.Normalize(vmin =0, vmax=6)
    lc = LineCollection(segments=segments, cmap = "turbo", norm=norm)
    lc.set_array(v_xi)
    lc.set_linewidth(2)

    plt.gca().add_collection(lc)
    plt.xlabel("x[m]")
    plt.ylabel("y[m]")
    cbar = plt.colorbar(lc, label = '$v_{\\xi}$')
    plt.axis('equal')
    plt.grid(True)


    plt.xlabel("x[m]")
    plt.ylabel("y[m]")


    fig, axs = plt.subplots(2,1, figsize = (10,6))

    axs[0].title.set_text("Computing time historgram")
    axs[0].hist(errors[3,1:-1]*1000)
    axs[0].axvline(x = car0_controller_forward.MPCC_params["Tf"]/car0_controller_forward.MPCC_params["N"]*1000, color = 'r', label = 'sampling time [ms]')
    axs[0].legend()
    axs[0].set_xlabel("Computing time [ms]")
    axs[0].set_ylabel("Number of iterations [-]")
    axs[0].set_xlim(left = 0)

    axs[1].title.set_text("Computing time")
    axs[1].set_xlabel("Iteration [-]")
    axs[1].set_ylabel("Computing time [ms]")
    axs[1].plot(np.arange(np.shape(errors[3,1:-1])[0]),errors[3,1:-1]*1000 , label = "computing time [ms]")
    axs[1].axhline(y = car0_controller_forward.MPCC_params["Tf"]/car0_controller_forward.MPCC_params["N"]*1000, color = 'r', label = 'sampling time [ms]')
    axs[1].legend()

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

    plt.ion()
    plt.tight_layout()
    plt.show()


    input("Press enter to close")


def add_trajectory_markers(scene, trajectory: CarTrajectory, color: str, size: int = 0.05):
    """
    Create the reference path points on the scene

    Args:
        scene: current scene (xml generator)
        trajectory(CarTrajectory): built CarTrajectory instance
        color(str): color of the trajectory markers
        size(int): diameter of the trajectory markers [m]
    """
    #Reference trajectory points:
    t_end = trajectory.evol_tck[0][-1]


    t_eval=np.linspace(0, t_end, int(t_end*3))
    s=splev(t_eval, trajectory.evol_tck)

    (x,y) = splev(s, trajectory.pos_tck)

    traj_markers = scene.add_trajectory_markers(x,y,color, size = size)

    return traj_markers

def get_param_args(file_name: str):
    """
    Return vehicle and MPCC parameters in a dictionary

    Args:
        filename(str): name of the yaml file (short form)
    Returns:
        args(dict): contatins vehicle, MPCC, drive_bridge, crazy_observer parameters with the keys respectivly
    """
    parent_dir = os.path.dirname(os.path.dirname(__file__))

    try:
        with open(file_name) as file:
            params = yaml.full_load(file)
    except Exception as e:
        print("file nound found. trying it in the local directory")
        try:
            file_name = os.path.join(parent_dir, "examples/", file_name)
            with open(file_name) as file:
                params = yaml.full_load(file)
        except Exception as e:
            raise e

    args = {}
    args["vehicle_params"] = params["parameter_server"]["ros__parameters"]["vehicle_params"]
    args["MPCC_params"] = params["parameter_server"]["ros__parameters"]["controllers"]["MPCC"]
    args["drive_bridge"] = params["parameter_server"]["ros__parameters"]["drive_bridge"]
    args["crazy_observer"] = params["parameter_server"]["ros__parameters"]["crazy_observer"]
    
    return args


if __name__ == "__main__":
    main()
