# aimotion-f1tenth-simulator

MuJoCo simulation environment for F1TENTh vehicles, based on the [AIMotionLab-Virtual package](https://github.com/AIMotionLab-SZTAKI/AIMotionLab-Virtual)

## Installation
Clone the repository:
```
$ git clone https://github.com/BommerHun/AiMotionLab_MPCC_sim
```
Create a virtual environment and install the package:
```
$ cd aimotion-f1tenth-simulator
$ python -m venv venv
$ source venv/bin/activate
$ pip install -e .
```
**Note** that Acados must be installed to use the simulator.

See https://github.com/acados/acados for detailes.

After Acados is installed:
```
$ pip install <acados_dir>/interfaces/acados_template/
$ source activate_framework.sh
```
## Usage
See `examples/example_trajectory_execution.py` for a detailed example script. 