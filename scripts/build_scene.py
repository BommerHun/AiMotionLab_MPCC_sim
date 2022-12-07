from dataclasses import dataclass
import numpy as np
import os
from util import xml_generator
from classes.passive_display import PassiveDisplay
from gui.building_input_gui import BuildingInputGui
from gui.drone_input_gui import DroneInputGui
from gui.cargo_input_gui import CargoInputGui


# open the base on which we'll build
xml_path = os.path.join("..", "xml_models")
xmlBaseFileName = "scene.xml"
save_filename = "built_scene.xml"

build_based_on_optitrack = False

scene = xml_generator.SceneXmlGenerator(os.path.join(xml_path, xmlBaseFileName))
display = PassiveDisplay(os.path.join(xml_path, xmlBaseFileName), False)
#display.set_drone_names()

drone_counter = 0

drone_positions = ["-1 -1 0.5", "1 -1 0.5", "-1 1 0.5", "1 1 0.5"]
drone_colors = ["0.1 0.9 0.1 1", "0.9 0.1 0.1 1", "0.1 0.1 0.9 1", "0.5 0.5 0.1 1"]

RED_COLOR = "0.85 0.2 0.2 1.0"
BLUE_COLOR = "0.2 0.2 0.85 1.0"

landing_zone_counter = 0
pole_counter = 0

def add_building():
    global scene, display
    input_gui = BuildingInputGui()
    input_gui.show()

    # add airport
    if input_gui.building == "Airport":
        if input_gui.position != "" and input_gui.quaternion != "":
            splt_pos = input_gui.position.split()
            if len(splt_pos) == 3:
                # if the 3rd coordinate is 0
                # since it's a plane, need to move it up a little so that the carpet would not cover it
                if(splt_pos[2] == '0'):
                    splt_pos[2] = "0.01"
            input_gui.position = splt_pos[0] + " " + splt_pos[1] + " " + splt_pos[2]
            scene.add_airport(input_gui.position, input_gui.quaternion)
            save_and_reload_model(scene, display, os.path.join(xml_path, save_filename))
    
    # add parking lot
    elif input_gui.building == "Parking lot":
        if input_gui.position != "" and input_gui.quaternion != "":
            splt_pos = input_gui.position.split()
            if len(splt_pos) == 3:
                # if the 3rd coordinate is 0
                # since it's a plane, need to move it up a little so that the carpet would not cover it
                if(splt_pos[2] == '0'):
                    splt_pos[2] = "0.01"
            input_gui.position = splt_pos[0] + " " + splt_pos[1] + " " + splt_pos[2]
            scene.add_parking_lot(input_gui.position, input_gui.quaternion)
            save_and_reload_model(scene, display, os.path.join(xml_path, save_filename))

    # add hospital
    elif input_gui.building == "Hospital":
        if input_gui.position != "" and input_gui.quaternion != "":
            scene.add_hospital(input_gui.position, input_gui.quaternion)
            save_and_reload_model(scene, display, os.path.join(xml_path, save_filename))
    
    # add post-office
    elif input_gui.building == "Post office":
        if input_gui.position != "" and input_gui.quaternion != "":
            scene.add_post_office(input_gui.position, input_gui.quaternion)
            save_and_reload_model(scene, display, os.path.join(xml_path,save_filename))

    # add landing zone
    elif input_gui.building == "Landing zone":
        global landing_zone_counter
        lz_name = "landing_zone" + str(landing_zone_counter)
        landing_zone_counter += 1
        if input_gui.position != "" and input_gui.quaternion != "":
            scene.add_landing_zone(lz_name, input_gui.position, input_gui.quaternion)
            save_and_reload_model(scene, display, save_filename)

    # add Sztaki
    elif input_gui.building == "Sztaki":
        if input_gui.position != "" and input_gui.quaternion != "":
            scene.add_sztaki(input_gui.position, input_gui.quaternion)
            save_and_reload_model(scene, display, os.path.join(xml_path,save_filename))
    
    # add pole
    elif input_gui.building == "Pole":
        global pole_counter
        p_name = "pole" + str(pole_counter)
        pole_counter += 1
        if input_gui.position != "" and input_gui.quaternion != "":
            scene.add_pole(p_name, input_gui.position, input_gui.quaternion)
            save_and_reload_model(scene, display, os.path.join(xml_path,save_filename))
    
    else:
        print("Non-existent building")
    

    
def add_drone():
    global scene, display
    
    input_gui = DroneInputGui()
    input_gui.show()
    if input_gui.drone_type == "Virtual crazyflie":
        if input_gui.position != "" and input_gui.quaternion != "":
            scene.add_drone(input_gui.position, input_gui.quaternion, RED_COLOR, True, "crazyflie")
            save_and_reload_model(scene, display, os.path.join(xml_path,save_filename))
    elif input_gui.drone_type == "Virtual bumblebee":
        if input_gui.position != "" and input_gui.quaternion != "":
            scene.add_drone(input_gui.position, input_gui.quaternion, RED_COLOR, True, "bumblebee", False)
            save_and_reload_model(scene, display, os.path.join(xml_path,save_filename))
    elif input_gui.drone_type == "Virtual bb with hook":
        if input_gui.position != "" and input_gui.quaternion != "":
            scene.add_drone(input_gui.position, input_gui.quaternion, RED_COLOR, True, "bumblebee", True)
            save_and_reload_model(scene, display, os.path.join(xml_path,save_filename))
    elif input_gui.drone_type == "Real crazyflie":
        if input_gui.position != "" and input_gui.quaternion != "":
            scene.add_drone(input_gui.position, input_gui.quaternion, BLUE_COLOR, False, "crazyflie")
            save_and_reload_model(scene, display, os.path.join(xml_path,save_filename))
    elif input_gui.drone_type == "Real bumblebee":
        if input_gui.position != "" and input_gui.quaternion != "":
            scene.add_drone(input_gui.position, input_gui.quaternion, BLUE_COLOR, False, "bumblebee", False)
            save_and_reload_model(scene, display, os.path.join(xml_path,save_filename))
    elif input_gui.drone_type == "Real bb with hook":
        if input_gui.position != "" and input_gui.quaternion != "":
            scene.add_drone(input_gui.position, input_gui.quaternion, BLUE_COLOR, False, "bumblebee", True)
            save_and_reload_model(scene, display, os.path.join(xml_path,save_filename))
    
    else:
        #print(input_gui.drone_type)
        print("Non-existent drone type " + input_gui.drone_type)

def add_load():
    global scene, display
    input_gui = CargoInputGui()
    input_gui.show()

    if input_gui.position != "" and input_gui.size != "" and input_gui.mass != "" and input_gui.quaternion != "":
        scene.add_load(input_gui.position, input_gui.size, input_gui.mass, input_gui.quaternion, input_gui.color)

        save_and_reload_model(scene, display, os.path.join(xml_path,save_filename))


def save_and_reload_model(scene, display, save_filename, drone_names_in_motive=None):
        scene.save_xml(save_filename)
        display.reload_model(save_filename, drone_names_in_motive)

def clear_scene():
    global scene, display, drone_counter, landing_zone_counter, pole_counter
    scene = xml_generator.SceneXmlGenerator(os.path.join(xml_path, xmlBaseFileName))
    display.reload_model(os.path.join(xml_path, xmlBaseFileName))
    
    drone_counter = 0
    landing_zone_counter = 0
    pole_counter = 0


def build_from_optitrack():
    global scene, display

    drone_names_in_motive = []

    if not display.connect_to_optitrack:
        display.connect_to_Optitrack()

    display.mc.waitForNextFrame()
    for name, obj in display.mc.rigidBodies.items():

        # have to put rotation.w to the front because the order is different
        orientation = str(obj.rotation.w) + " " + str(obj.rotation.x) + " " + str(obj.rotation.y) + " " + str(obj.rotation.z)
        position = str(obj.position[0]) + " " + str(obj.position[1]) + " " + '0'


        if name.startswith("cf"):
            scene.add_landing_zone("lz_" + name, position, "1 0 0 0")
            position = str(obj.position[0]) + " " + str(obj.position[1]) + " " + str(obj.position[2])
            scene.add_drone(position, orientation, BLUE_COLOR, False, False)
            drone_names_in_motive += [name]

        elif name == "bu11":
            scene.add_hospital(position, "1 0 0 0")
        elif name == "bu12":
            scene.add_sztaki(position, "1 0 0 0")
        elif name == "bu13":
            scene.add_post_office(position, "1 0 0 0")
        elif name == "bu14":
            position = str(obj.position[0]) + " " + str(obj.position[1]) + " 0.01"
            scene.add_airport(position, "1 0 0 0")

        elif name.startswith("obs"):
            scene.add_pole(name, position, orientation)

    save_and_reload_model(scene, display, os.path.join(xml_path,save_filename), drone_names_in_motive)


def main():
    display.set_key_b_callback(add_building)
    display.set_key_d_callback(add_drone)
    display.set_key_o_callback(build_from_optitrack)
    display.set_key_t_callback(add_load)
    display.set_key_delete_callback(clear_scene)
    
    #display.print_optitrack_data()

    if build_based_on_optitrack:
        build_from_optitrack()

    display.run()

if __name__ == '__main__':
    main()
