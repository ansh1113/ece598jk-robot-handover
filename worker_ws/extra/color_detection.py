import mujoco
import mujoco.viewer
import numpy as np
import cv2
import os

os.chdir("src/robotis_mujoco_menagerie/robotis_ffw")

mj_model = mujoco.MjModel.from_xml_path("scene_with_object.xml")
data = mujoco.MjData(mj_model)

renderer = mujoco.Renderer(mj_model, height=480, width=640)
rgb_cam_id = mj_model.camera("head_rgb").id
depth_cam_id = mj_model.camera("head_depth").id

with mujoco.viewer.launch_passive(mj_model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(mj_model, data)
        
        # Render RGB
        renderer.update_scene(data, camera=rgb_cam_id)
        rgb_img = renderer.render()
        
        # Convert to HSV for red detection
        hsv = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV)
        
        # Red color mask (two ranges because red wraps around in HSV)
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = mask1 | mask2
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detection_img = rgb_img.copy()
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 100:  # Filter small noise
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(detection_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(detection_img, "Red Box", (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Render Depth
        renderer.enable_depth_rendering()
        renderer.update_scene(data, camera=depth_cam_id)
        depth_img = renderer.render()
        renderer.disable_depth_rendering()
        
        depth_normalized = np.clip(depth_img, 0, 10) / 10.0
        
        cv2.imshow("Red Box Detection", cv2.cvtColor(detection_img, cv2.COLOR_RGB2BGR))
        cv2.imshow("Depth Camera", depth_normalized)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        viewer.sync()

cv2.destroyAllWindows()