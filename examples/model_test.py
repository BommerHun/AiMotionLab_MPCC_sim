
# Short script to test current mpcc model
from aimotion_f1tenth_simulator.classes.car_classes import CarMPCCController
from aimotion_f1tenth_simulator.classes.mpcc_util.MPCC_plotter import MPCC_plotter
from aimotion_f1tenth_simulator.classes.traj_classes import CarTrajectory
from aimotion_f1tenth_simulator.classes.trajectory_generators import null_infty
import numpy as np
import os 
import yaml
import time


reversed = True


x0 =np.array([0, 0,0.64424,0,0,0])

if reversed == True:
    x0[2] = x0[2]+ np.pi


#MPCC param file
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


# create a trajectory
car0_trajectory=CarTrajectory()


path, v = null_infty(laps=1, scale = 1)
car0_trajectory.build_from_points_const_speed(path, path_smoothing=0.01, path_degree=4, const_speed=1.5)


MPCC_controller = CarMPCCController(vehicle_params=args["vehicle_params"],
                                    MPCC_params= args["MPCC_params"],
                                    mute = False,
                                    lin_tire= False,
                                    index= 3)

dt = args["MPCC_params"]["Tf"]/args["MPCC_params"]["N"]

MPCC_controller.set_trajectory(car0_trajectory.pos_tck, car0_trajectory.evol_tck, x0)


#Create MPCC plotter:

plotter = MPCC_plotter()
s = np.linspace(0, MPCC_controller.trajectory.L,10000)

plotter.set_ref_traj(np.array(MPCC_controller.trajectory.spl_sx(s)), np.array(MPCC_controller.trajectory.spl_sy(s)))

plotter.show()

#        x0 = np.array([x0["pos_x"], x0["pos_y"], x0["head_angle"], x0["long_vel"], x0["lat_vel"], x0["yaw_rate"]])

while not MPCC_controller.finished:
    state_vector = {"pos_x":x0[0],
                    "pos_y":x0[1],
                    "head_angle":x0[2],
                    "long_vel": x0[3],
                    "lat_vel": x0[4],
                    "yaw_rate": x0[5]}
    
    u_opt = MPCC_controller.compute_control(x0 = state_vector,setpoint= None,time= None)
    x0,t = MPCC_controller.simulate(x0, u_opt, dt, 0)
    #update horizon plotter and mujoco markers
    horizon = np.array(np.reshape(MPCC_controller.ocp_solver.get(0, 'x'),(-1,1)))
    for i in range(MPCC_controller.parameters.N):
        x_temp   = MPCC_controller.ocp_solver.get(i, 'x')
        x_temp = np.reshape(x_temp, (-1,1))
        horizon = np.append(horizon, x_temp, axis = 1)

    plotter.update_plot(new_x = horizon[0,:], new_y = horizon[1,:])
    time.sleep(dt)




