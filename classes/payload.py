from classes.moving_object import MovingMocapObject, MovingObject
from util import mujoco_helper
from enum import Enum
import numpy as np


class PAYLOAD_TYPES(Enum):
    Box = "Box"
    Teardrop = "Teardrop"

class PayloadMocap(MovingMocapObject):

    def __init__(self, model, data, mocapid, name_in_xml, name_in_motive) -> None:
        super().__init__(name_in_xml, name_in_motive)

        self.data = data
        self.mocapid = mocapid
    
    
    def update(self, pos, quat):

        self.data.mocap_pos[self.mocapid] = pos
        self.data.mocap_quat[self.mocapid] = quat
    
    def get_qpos(self):
        return np.append(self.data.mocap_pos[self.mocapid], self.data.mocap_quat[self.mocapid])

    @staticmethod
    def parse(data, model):
        payloads = []
        plc = 1

        body_names = mujoco_helper.get_body_name_list(model)

        for name in body_names:
            if name.startswith("loadmocap") and not name.endswith("hook"):
                
                mocapid = model.body(name).mocapid[0]
                c = PayloadMocap(model, data, mocapid, name, "loadmocap" + str(plc))
                
                payloads += [c]
                plc += 1
        
        return payloads


class Payload(MovingObject):

    def __init__(self, model, data, name_in_xml, top_subdivision_x, top_subdivision_y) -> None:
        super().__init__(model, name_in_xml)

        self.data = data

        # supporting only rectangular objects for now
        self.geom = self.model.geom(name_in_xml)
        
        free_joint = self.data.joint(self.name_in_xml)
        self.qfrc_passive = free_joint.qfrc_passive
        self.qfrc_applied = free_joint.qfrc_applied

        self.size = self.geom.size # this is half size on each axis
        self.top_surface_area = 2 * self.size[0] * 2 * self.size[1]
        
        self.sensor_posimeter = self.data.sensor(self.name_in_xml + "_posimeter").data
        self.sensor_orimeter = self.data.sensor(self.name_in_xml + "_orimeter").data
    
    def update(self, i, control_step):
        
        return

    def set_top_subdivision(self, top_subdivision_x, top_subdivision_y):
        self.__top_subdivision_x = top_subdivision_x
        self.__top_subdivision_y = top_subdivision_y
        self.top_miniractangle_area = self.top_surface_area / (top_subdivision_x * top_subdivision_y)

    
    def __calc_minirectangle_positions(self):
        """ 3D vectors pointing from the center of the box, to the center of the small rectangles"""

        self.__minirectangle_positions = np.zeros((self.__top_subdivision_x, self.__top_subdivision_y, 3))


        for i in range(self.__top_subdivision_x):
            division_size_x = (2 * self.size[0]) / self.__top_subdivision_x
            distance_x = i * division_size_x + (division_size_x / 2)

            for j in range(self.__top_subdivision_y):
                

                distance_z = self.size[2] # no need to divide by 2, because it's half size

                #self.__minirectangle_positions[i, j] = 

    
    def get_top_position_at(self, i, j):
        """ get the center in world coordinates of a small rectangle on the top of the box"""

        # 
    

    @staticmethod
    def parse(data, model):
        payloads = []
        plc = 0

        joint_names = mujoco_helper.get_joint_name_list(model)

        for name in joint_names:
            if name.startswith("load"):
                
                
                c = Payload(model, data, name, 100, 100)
                
                payloads += [c]
                plc += 1
        
        return payloads