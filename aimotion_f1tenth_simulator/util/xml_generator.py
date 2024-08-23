import xml.etree.ElementTree as ET
import numpy as np
import aimotion_f1tenth_simulator.util.mujoco_helper as mh
import math

from aimotion_f1tenth_simulator.classes.car import F1T_PROP

import os



PROP_COLOR = "0.1 0.1 0.1 1.0"
PROP_LARGE_COLOR = "0.1 0.02 0.5 1.0"

SITE_NAME_END = "_cog"


class SceneXmlGenerator:

    def __init__(self, base_scene_filename):

        self.root = ET.Element("mujoco")
        ET.SubElement(self.root, "include", file=base_scene_filename)
        self.worldbody = ET.SubElement(self.root, "worldbody")
        self.contact = ET.SubElement(self.root, "contact")
        self.actuator = ET.SubElement(self.root, "actuator")
        self.sensor = ET.SubElement(self.root, "sensor")

        self.parking_lot = None
        self.airport = None
        self.hospital = None
        self.post_office = None
        self.sztaki = None

        self._bicycle_counter = 0

        self._pole_cntr = 0
        self._virtcrazyflie_cntr = 0
        self._virtbumblebee_cntr = 0
        self._virtbumblebee_hooked_cntr = 0
        self._realcrazyflie_cntr = 0
        self._realbumblebee_cntr = 0
        self._realbumblebee_hooked_cntr = 0
        self._virtfleet1tenth_cntr = 0
        self._realfleet1tenth_cntr = 0
        
        self._box_payload_cntr = 0
        self._teardrop_payload_cntr = 0
        self._mocap_box_payload_cntr = 0
        self._mocap_teardrop_payload_cntr = 0

        self._mocap_drone_names = []
        self._mocap_payload_names = []

        self.option = ET.SubElement(self.root, "option")



        ET.SubElement(self.option, "flag", contact = "enable")

#        ET.SubElement(self.root, "option", contact = "false")
    def add_bicycle(self, pos, quat, color):

        name = "Bicycle_" + str(self._bicycle_counter)
        self._bicycle_counter += 1

        site_name = name + "_cog"

        bicycle = ET.SubElement(self.worldbody, "body", name=name, pos=pos, quat=quat)

        ET.SubElement(bicycle, "inertial", pos="0 0 0", diaginertia=".01 .01 .01", mass="1.0")
        ET.SubElement(bicycle, "joint", name = name, type="free")
        ET.SubElement(bicycle, "site", name=site_name, pos="0 0 0")

        ET.SubElement(bicycle, "geom", name=name + "_crossbar", type="box", size=".06 .015 .02", pos="0 0 0", rgba=color)

        front_wheel_name = name + "_wheelf"
        wheelf = ET.SubElement(bicycle, "body", name=front_wheel_name)

        ET.SubElement(wheelf, "joint", name=front_wheel_name, type="hinge", pos="0.1 0 0",
                      axis="0 1 0", frictionloss="0.001", damping="0.00001", armature="0.01")
        ET.SubElement(wheelf, "geom", name=front_wheel_name, type="cylinder", size="0.04 0.015",
                      pos="0.1 0 0", euler="1.571 0 0", material="material_check")

        rear_wheel_name = name + "_wheelr"
        wheelr = ET.SubElement(bicycle, "body", name=rear_wheel_name)

        ET.SubElement(wheelr, "joint", name=rear_wheel_name, type="hinge", pos="-0.1 0 0",
                      axis="0 1 0", frictionloss="0.001", damping="0.00001", armature="0.01")
        ET.SubElement(wheelr, "geom", name=rear_wheel_name, type="cylinder", size="0.04 0.015",
                      pos="-0.1 0 0", euler="1.571 0 0", material="material_check")
        
        ET.SubElement(self.actuator, "motor", name=name + "_actr", joint=rear_wheel_name)
        ET.SubElement(self.sensor, "velocimeter", site=site_name, name=name + "_velocimeter")


    def add_airport(self, pos, quat=None):
        if self.airport is None:

            tag = "geom"
            name = "airport"
            size = "0.105 0.105 .05"
            type = "plane"
            material = "mat-airport"

            if quat is None:
                self.airport = ET.SubElement(self.worldbody, tag, name=name, pos=pos, size=size, type=type, material=material)
            else:
                self.airport = ET.SubElement(self.worldbody, tag, name=name, pos=pos, quat=quat, size=size, type=type, material=material)
            return self.airport
        else:
            print("[SceneXmlGenerator] Airport already added")


    def add_parking_lot(self, pos, quat=None):
        if self.parking_lot is None:

            tag = "geom"
            name = "parking_lot"
            size = "0.105 0.115 .05"
            type = "plane"
            material = "mat-parking_lot"

            if quat is None:
                self.parking_lot = ET.SubElement(self.worldbody, tag, name=name, pos=pos, size=size, type=type, material=material)
            else:
                self.parking_lot = ET.SubElement(self.worldbody, tag, name=name, pos=pos, quat=quat, size=size, type=type, material=material)
            return self.parking_lot
        else:
            print("[SceneXmlGenerator] Parking lot already added")
    

    def add_pole(self, pos, quat=None):
        name = "pole_" + str(self._pole_cntr)
        self._pole_cntr += 1
        tag = "body"
        if quat is None:
            pole = ET.SubElement(self.worldbody, tag, name=name, pos=pos)
        else:
            pole = ET.SubElement(self.worldbody, tag, name=name, pos=pos, quat=quat)
        
        ET.SubElement(pole, "geom", {"class": "pole_top"})
        ET.SubElement(pole, "geom", {"class": "pole_bottom1"})
        ET.SubElement(pole, "geom", {"class": "pole_bottom2"})

        return pole


    @staticmethod
    def print_elements(root, tabs=""):

        for v in root:
            if "name" in v.attrib:
                print(tabs + str(v.attrib["name"]))

            tbs = tabs + "\t"

            SceneXmlGenerator.print_elements(v, tbs)


    def add_hospital(self, pos, quat=None):
        name = "hospital"
        if self.hospital is None:
            tag = "body"
            if quat is None:
                self.hospital = ET.SubElement(self.worldbody, tag, name=name, pos=pos)
            else:
                self.hospital = ET.SubElement(self.worldbody, tag, name=name, pos=pos, quat=quat)

            ET.SubElement(self.hospital, "geom", name=name, type="box", pos="0 0 0.445", size="0.1275 0.13 0.445", material="mat-hospital")

            return self.hospital
        else:
            print("[SceneXmlGenerator] Hospital already added")


    def add_post_office(self, pos, quat=None):
        name = "post_office"
        if self.post_office is None:
            tag = "body"
            if quat is None:
                self.post_office = ET.SubElement(self.worldbody, tag, name=name, pos=pos)
            else:
                self.post_office = ET.SubElement(self.worldbody, tag, name=name, pos=pos, quat=quat)

            ET.SubElement(self.post_office, "geom", name=name, type="box", pos="0 0 0.205", size="0.1275 0.1275 0.205", material="mat-post_office")

            return self.post_office
        else:
            print("[SceneXmlGenerator] Post office already added")


    def add_landing_zone(self, name, pos, quat=None):
        tag = "body"
        if quat is None:
            landing_zone = ET.SubElement(self.worldbody, tag, name=name, pos=pos)
        else:
            landing_zone = ET.SubElement(self.worldbody, tag, name=name, pos=pos, quat=quat)
        
        ET.SubElement(landing_zone, "geom", {"class" : "landing_zone"})

        return landing_zone


    def add_sztaki(self, pos, quat):
        if self.sztaki is None:
            name = "sztaki"
            
            self.sztaki = ET.SubElement(self.worldbody, "body", name=name, pos=pos, quat=quat)

            ET.SubElement(self.sztaki, "geom", name=name, type="box", pos="0 0 0.0925", size="0.105 0.105 0.0925", rgba="0.8 0.8 0.8 1.0", material="mat-sztaki")

            return self.sztaki

        else:
            print("[SceneXmlGenerator] Sztaki already added")
    

    
    def add_car(self, pos, quat, color, is_virtual, has_rod=False, type="fleet1tenth", **kwargs):

        name = None

        if is_virtual and type == "fleet1tenth":
            name = "Fleet1Tenth_" + str(self._virtfleet1tenth_cntr)
            self._add_fleet1tenth(pos, quat, name, color, has_rod, **kwargs)
            self._virtfleet1tenth_cntr += 1
            if self._virtfleet1tenth_cntr >=2:
                for i in range(self._virtfleet1tenth_cntr-1):
                    ET.SubElement(self.contact, "exclude", body1 = name, body2 = f"Fleet1Tenth_{i}")
        
        elif not is_virtual and type == "fleet1tenth":
            name = "CarMocap_fleet1tenth_" + str(self._realfleet1tenth_cntr)
            self._add_mocap_fleet1tenth(pos, quat, name, color, has_rod, **kwargs)
            self._realfleet1tenth_cntr += 1
        
        else:
            print("[SceneXmlGenerator] Unknown car type")
            return None
        
        return name
    
    def add_trajectory_markers(self, x,y, color, size = 0.01):
        name = "ref_trajectory_body"
        trajectory_ref = ET.SubElement(self.worldbody, "body", name = name, mocap = "true", pos = "0 0 0")

        for i in range(np.shape(x)[0]):
            marker = ET.SubElement(trajectory_ref, "geom", type = "sphere",size = f"{size}",contype="0", conaffinity="0", rgba = color, pos = f"{x[i]} {y[i]} 0")

        ET.SubElement(self.contact, "exclude",name = "marker_exclude", body1= "Fleet1Tenth_0",body2=  name)

    def add_MPCC_markers(self, n, color = "256 0 0", pos = "0 0 0", quat= "1 0 1", size = 0.1):
        markers = []
        
        for i in range(n):
            name = f"mpcc_{i}"
            x,y,z = 0+i*1, 0, 1

            markers.append(ET.SubElement(self.worldbody, "body", name = name,mocap = "true", pos = f"{x} {y} {z}", quat = quat))

            m = ET.SubElement(markers[i], "geom", name = f"mpcc_marker{i}", type = "sphere",contype="0" ,conaffinity="0", size = f"{size}", pos = "0 0 0", rgba = color)
            #joint = ET.SubElement(markers[i], "joint", type = "free", name = f"marker_{i}_free_joint")
            #markers.append(ET.SubElement(marker_root, "geom", type = "sphere", size = ".05", pos = f"{x} {y} {z}", rgba = color))
        for i in range(n):
            ET.SubElement(self.contact, "exclude", name = f"car_marker_exc{i}", body1= "Fleet1Tenth_0",body2=  f"mpcc_{i}")

        return markers
        
    

    def _add_fleet1tenth(self, pos, quat, name, color, has_rod, **kwargs):
        
        wheel_width = kwargs["wheel_width"] if "wheel_width" in kwargs else F1T_PROP.WHEEL_WIDTH.value
        wheel_radius = kwargs["wheel_radius"] if "wheel_radius" in kwargs else F1T_PROP.WHEEL_RADIUS.value
    

        wheel_size = wheel_radius + " " + wheel_width
        
        site_name = name + SITE_NAME_END

        posxyz = str.split(pos)
        pos = posxyz[0] + " " + posxyz[1] + " " + wheel_radius
        
        car = ET.SubElement(self.worldbody, "body", name=name, pos=pos, quat=quat, )

        mass = kwargs["mass"] if "mass" in kwargs else "3.0"
        inertia = kwargs["inertia"] if "intertia" in kwargs else ".05 .05 .08"

        ET.SubElement(car, "inertial", pos="0 0 0", diaginertia=inertia, mass=mass)
        ET.SubElement(car, "joint", name=name, type="free")
        ET.SubElement(car, "site", name=site_name, pos="0 0 0")

        self._add_fleet1tenth_body(car, name, color, has_rod)

        armature = "0.05"
        armature_steer = "0.001"
        fric_steer = "0.2"
        damp_steer = "0.2"
        damping = "0.00001"
        frictionloss = "0.01"

        steer_range = "-0.6 0.6"


        wheelfl = ET.SubElement(car, "body", name=name + "_wheelfl" )
        ET.SubElement(wheelfl, "joint", name=name + "_wheelfl_steer", type="hinge", pos="0.16113 .10016 0", limited="true", frictionloss=fric_steer, damping=damp_steer, armature=armature_steer, range=steer_range, axis="0 0 1")
        ET.SubElement(wheelfl, "joint", name=name + "_wheelfl", type="hinge", pos="0.16113 .122385 0", axis="0 1 0", frictionloss=frictionloss, damping=damping, armature=armature, limited="false")

        ET.SubElement(wheelfl, "geom", name=name + "_wheelfl", type="cylinder", contype="0", conaffinity="0", size=wheel_size, pos="0.16113 .122385 0", mass="0.1", material="material_check", euler="1.571 0 0")

        wheelrl = ET.SubElement(car, "body", name=name + "_wheelrl" )
        ET.SubElement(wheelrl, "joint", name=name + "_wheelrl", type="hinge", pos="-0.16113 .122385 0", axis="0 1 0", frictionloss=frictionloss, damping=damping, armature=armature, limited="false")

        ET.SubElement(wheelrl, "geom", name=name + "_wheelrl", type="cylinder",contype="0", conaffinity="0", size=wheel_size, pos="-0.16113 .122385 0", mass="0.1", material="material_check", euler="1.571 0 0")

        wheelfr = ET.SubElement(car, "body", name=name + "_wheelfr" )
        ET.SubElement(wheelfr, "joint", name=name + "_wheelfr_steer", type="hinge", pos="0.16113 -.10016 0", limited="true", frictionloss=fric_steer, damping=damp_steer, armature=armature_steer, range=steer_range, axis="0 0 1")
        ET.SubElement(wheelfr, "joint", name=name + "_wheelfr", type="hinge", pos="0.16113 -.122385 0", axis="0 1 0", frictionloss=frictionloss, damping=damping, armature=armature, limited="false")

        ET.SubElement(wheelfr, "geom", name=name + "_wheelfr", type="cylinder", contype="0", conaffinity="0",size=wheel_size, pos="0.16113 -.122385 0", mass="0.1", material="material_check", euler="1.571 0 0")

        wheelrr = ET.SubElement(car, "body", name=name + "_wheelrr" )
        ET.SubElement(wheelrr, "joint", name=name + "_wheelrr", type="hinge", pos="-0.16113 -.122385 0", axis="0 1 0", frictionloss=frictionloss, damping=damping, armature=armature, limited="false")

        ET.SubElement(wheelrr, "geom", name=name + "_wheelrr", type="cylinder", contype="0", conaffinity="0", size=wheel_size, pos="-0.16113 -.122385 0", mass="0.1", material="material_check", euler="1.571 0 0")

        friction=kwargs["friction"] if "friction" in kwargs else "2.5 2.5 .009 .0001 .0001"

        ET.SubElement(self.contact, "pair", geom1=name + "_wheelfl", geom2="roundabout", condim="6", friction=friction)
        ET.SubElement(self.contact, "pair", geom1=name + "_wheelfr", geom2="roundabout", condim="6", friction=friction)
        ET.SubElement(self.contact, "pair", geom1=name + "_wheelrl", geom2="roundabout", condim="6", friction=friction)
        ET.SubElement(self.contact, "pair", geom1=name + "_wheelrr", geom2="roundabout", condim="6", friction=friction)

        ET.SubElement(self.actuator, "motor", name=name + "_wheelfl_actr", joint=name + "_wheelfl")
        ET.SubElement(self.actuator, "motor", name=name + "_wheelfr_actr", joint=name + "_wheelfr")
        ET.SubElement(self.actuator, "motor", name=name + "_wheelrl_actr", joint=name + "_wheelrl")
        ET.SubElement(self.actuator, "motor", name=name + "_wheelrr_actr", joint=name + "_wheelrr")

        kp = "15"
        ET.SubElement(self.actuator, "position", forcelimited="true", forcerange="-5 5", name=name + "_wheelfl_actr_steer", joint=name + "_wheelfl_steer", kp=kp)
        ET.SubElement(self.actuator, "position", forcelimited="true", forcerange="-5 5", name=name + "_wheelfr_actr_steer", joint=name + "_wheelfr_steer", kp=kp)


        ET.SubElement(self.sensor, "gyro", site=site_name, name=name + "_gyro")
        ET.SubElement(self.sensor, "velocimeter", site=site_name, name=name + "_velocimeter")
        ET.SubElement(self.sensor, "framepos", objtype="site", objname=site_name, name=name + "_posimeter")
        ET.SubElement(self.sensor, "framequat", objtype="site", objname=site_name, name=name + "_orimeter")
        ET.SubElement(self.sensor, "accelerometer", site=site_name, name=name + "_accelerometer")

    
    def _add_fleet1tenth_body(self, car, name, color, has_rod):
        ET.SubElement(car, "geom", name=name + "_chassis_b", type="box", contype="0", conaffinity="0",size=".10113 .1016 .02", pos= "-.06 0 0", rgba=color)
        ET.SubElement(car, "geom", name=name + "_chassis_f", type="box", contype="0", conaffinity="0",size=".06 .07 .02", pos=".10113 0 0", rgba=color)
        ET.SubElement(car, "geom", name=name + "_front", type="box",contype="0", conaffinity="0", size=".052388 .02 .02", pos=".2135 0 0", rgba=color)
        ET.SubElement(car, "geom", name=name + "_back", type="box",contype="0", conaffinity="0", size=".052388 .02 .02", pos="-.2135 0 0", rgba=color)
        ET.SubElement(car, "geom", name=name + "_front_bumper", type="box", contype="0", conaffinity="0",size=".005 .09 .02", pos=".265888 0 0.02", rgba=color)
        ET.SubElement(car, "geom", name=name + "_back_bumper", type="box", contype="0", conaffinity="0",size=".005 .08 .02", pos="-.265888 0 0.02", rgba=color)
        ET.SubElement(car, "geom", name=name + "_number", type="cylinder",contype="0", conaffinity="0", size=".01984 .03", pos=".12 0 .05", rgba="0.1 0.1 0.1 1.0")
        ET.SubElement(car, "geom", name=name + "_camera", type="box",contype="0", conaffinity="0", size=".012 .06 0.02", pos=".18 0 .08")
        ET.SubElement(car, "geom", name=name + "_camera_holder", type="box",contype="0", conaffinity="0", size=".012 .008 .02", pos=".18 0 .04")
        ET.SubElement(car, "geom", name=name + "_circuits", type="box", contype="0", conaffinity="0",size=".08 .06 .03", pos="-.05 0 .05", rgba=color)
        ET.SubElement(car, "geom", name=name + "_antennal", type="box", contype="0", conaffinity="0",size=".007 .004 .06", pos="-.16 -.01 .105", euler="0.2 0 0", rgba=".1 .1 .1 1.0")
        ET.SubElement(car, "geom", name=name + "_antennar", type="box",contype="0", conaffinity="0", size=".007 .004 .06", pos="-.16 .01 .105", euler="-0.2 0 0", rgba=".1 .1 .1 1.0")
        ET.SubElement(car, "geom", name=name + "_antenna_holder", type="box",contype="0", conaffinity="0", size=".008 .008 .02", pos="-.16 0 .04", rgba=".1 .1 .1 1.0")

        if has_rod:
            ET.SubElement(car, "geom", name=name + "_rod", type="cylinder", contype="0", conaffinity="0",size="0.02 0.5225", pos="-.175 0 0.5225", rgba="0.3 0.3 0.3 1.0", euler="0 0.1 0")


    def save_xml(self, file_name):
        
        tree = ET.ElementTree(self.root)
        #ET.indent(tree, space="\t", level=0) # uncomment this if python version >= 3.9
        tree.write(file_name)
        print()
        print("[SceneXmlGenerator] Scene xml file saved at: " + os.path.normpath(file_name))

