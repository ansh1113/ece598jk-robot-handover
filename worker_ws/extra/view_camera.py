import mujoco
import mujoco.viewer
import numpy as np
import cv2
import os

os.chdir("src/robotis_mujoco_menagerie/robotis_ffw")

model = mujoco.MjModel.from_xml_path("scene_with_object.xml")
data = mujoco.MjData(model)

# Create renderer
renderer = mujoco.Renderer(model, height=480, width=640)

# Get camera IDs
rgb_cam_id = model.camera("head_rgb").id
depth_cam_id = model.camera("head_depth").id

print(f"RGB Camera ID: {rgb_cam_id}")
print(f"Depth Camera ID: {depth_cam_id}")

# Simulation loop
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)
        
        # Render RGB
        renderer.update_scene(data, camera=rgb_cam_id)
        rgb_img = renderer.render()
        
        # Render Depth
        renderer.enable_depth_rendering()
        renderer.update_scene(data, camera=depth_cam_id)
        depth_img = renderer.render()
        renderer.disable_depth_rendering()
        
        # Normalize depth for visualization
        depth_normalized = np.clip(depth_img, 0, 10) / 10.0  # Clip to 10m max
        
        # Display
        cv2.imshow("RGB Camera", cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR))
        cv2.imshow("Depth Camera", depth_normalized)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        viewer.sync()

cv2.destroyAllWindows()