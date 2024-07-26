import mujoco as mj
import numpy as np
# import mediapy as media
import matplotlib.pyplot as plt

import os
from mujoco.glfw import glfw

# define the XMl file path
xml_file_path = "./models/achilles.xml"
absolute_xml_file_path = os.path.abspath(xml_file_path)

# create model from the XML file
# model = mj.MjModel.from_xml_path(absolute_xml_file_path)

# # print out some mujoco data
# print("Model Info:")
# # print("Model name: ", model.names)
# print("\tModel nq: ", model.nq)
# print("\tModel nv: ", model.nv)
# print("\tModel nu: ", model.nu)
# print("\tModel nbody: ", model.nbody)
# print("\tModel njnt: ", model.njnt)
# print("\tModel ngeom: ", model.ngeom)

# # create data from the model
# data = mj.MjData(model)

# # enable joint visualization option:
# scene_option = mj.MjvOption()
# scene_option.flags[mj.mjtVisFlag.mjVIS_JOINT] = True

# # Make renderer, render and show the pixels
# renderer = mj.Renderer(model)
# duration = 3.8  # (seconds)
# framerate = 60  # (Hz)

# # Simulate and display video.
# frames = []
# mj.mj_resetData(model, data)  # Reset state and time.
# while data.time < duration:
#   mj.mj_step(model, data)
#   if len(frames) < data.time * framerate:
#     renderer.update_scene(data)
#     pixels = renderer.render()
#     frames.append(pixels)

model = mj.MjModel.from_xml_string(absolute_xml_file_path)
renderer = mj.Renderer(model)
data = mj.MjData(model)
mj.mj_forward(model, data)
renderer.update_scene(data, camera="closeup")
# media.show_image(renderer.render())



