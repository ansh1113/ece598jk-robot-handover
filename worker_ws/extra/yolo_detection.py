import mujoco
import mujoco.viewer
import numpy as np
import cv2
import os
from ultralytics import YOLO

os.chdir("src/robotis_mujoco_menagerie/robotis_ffw")

# Load YOLO model
model = YOLO('yolov8n.pt')  # Will auto-download on first run

# Load MuJoCo model
mj_model = mujoco.MjModel.from_xml_path("scene_with_object.xml")
data = mujoco.MjData(mj_model)

# Create renderer
renderer = mujoco.Renderer(mj_model, height=480, width=640)
rgb_cam_id = mj_model.camera("head_rgb").id
depth_cam_id = mj_model.camera("head_depth").id

print("YOLO model loaded. Press 'q' to quit.")

# Simulation loop
with mujoco.viewer.launch_passive(mj_model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(mj_model, data)
        
        # Render RGB
        renderer.update_scene(data, camera=rgb_cam_id)
        rgb_img = renderer.render()
        
        # Run YOLO detection
        results = model(rgb_img, verbose=False)
        
        # Draw detections
        annotated_frame = results[0].plot()
        
        # Render Depth
        renderer.enable_depth_rendering()
        renderer.update_scene(data, camera=depth_cam_id)
        depth_img = renderer.render()
        renderer.disable_depth_rendering()
        
        depth_normalized = np.clip(depth_img, 0, 10) / 10.0
        
        # Display
        cv2.imshow("YOLO Detection", cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR))
        cv2.imshow("Depth Camera", depth_normalized)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        viewer.sync()

cv2.destroyAllWindows()