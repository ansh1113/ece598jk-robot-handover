import mujoco
import mujoco.viewer
import os

# Change to model directory
os.chdir("src/robotis_mujoco_menagerie/robotis_ffw")

# Load the scene with object
model = mujoco.MjModel.from_xml_path("scene_with_object.xml")
data = mujoco.MjData(model)

# Launch interactive viewer
mujoco.viewer.launch(model, data)