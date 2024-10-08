from abc import ABC, abstractmethod
from aimotion_f1tenth_simulator.classes.controller_base import ControllerBase

class Base_MPCC_Controller(ABC, ControllerBase):

    @abstractmethod
    def set_trajectory(self, pos_tck, evol_tck, generate_solver: bool):
        pass

    @abstractmethod
    def init_controller(self, x0):
        pass

    @abstractmethod
    def generate_solver(self, pos_tck = None, evol_tck = None, x0 = None):
        pass
    