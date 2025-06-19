##
#
# Programatically convert all mesh .STL files in the model directory to .OBJ files
# Need: pip install pymeshlab
# 
# WARNING: This script will overwrite existing OBJ files
##

import os
import pymeshlab

# choose the robot model to convert
robot = "titan"

# path to the robot's model files
model_directory = "./models/{}/meshes".format(robot)

# get list of all files in the model directory
files = [f for f in os.listdir(model_directory) if os.path.isfile(os.path.join(model_directory, f))]

# extract only the STL files
files = [f for f in files if (f.endswith(".STL") or f.endswith(".stl"))]

# print the number of files to convert
print("Attempting to convert {} files...\n".format(len(files)))
for f in files:
    print("- {}".format(f))

# convert all STL files to OBJ
ms = pymeshlab.MeshSet()
num_converted_files = 0
for f in files:
    try:
        ms.load_new_mesh(os.path.join(model_directory, f))
        ms.save_current_mesh(os.path.join(model_directory, f.replace(".STL", ".OBJ").replace(".stl", ".obj")))
        print("- Converted {} to {}.OBJ".format(f, f.replace(".STL", "").replace(".stl", "")))
        num_converted_files += 1
    except Exception as e:
        print("\nError converting {} to OBJ: {}\n".format(f, e))
        continue

print("\nSuccessfully converted {} files to OBJ".format(num_converted_files))
