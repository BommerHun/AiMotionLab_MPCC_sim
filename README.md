# aimotion-f1tenth-simulator

MuJoCo simulation environment for F1TENTh vehicles, based on the [AIMotionLab-Virtual package](https://github.com/AIMotionLab-SZTAKI/AIMotionLab-Virtual)

## Installation
Clone the repository:
```
$ git clone https://github.com/flochkristof/aimotion-f1tenth-simulator.git
```
Create a virtual environment and install the package:
```
$ cd aimotion-f1tenth-simulator
$ python -m venv venv
$ pip install -e .
```
Windows users also need to install curses:
```
$ pip install windows-curses
```
## Usage
See `examples/example_trajectory_execution.py` for a detailed example script. 