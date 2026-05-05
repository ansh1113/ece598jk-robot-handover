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

# Control parameters
SAFE_DISTANCE = 1.5
TARGET_SPEED = 0.8  # Very slow
MAX_ACCEL = 0.05

current_speed = 0.0

def detect_red_box(rgb_img):
    hsv = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV)
    
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])
    
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = mask1 | mask2
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        if area > 100:
            x, y, w, h = cv2.boundingRect(largest_contour)
            return (x, y, w, h)
    return None

def get_depth_at_point(depth_img, x, y):
    h, w = depth_img.shape
    if 0 <= x < w and 0 <= y < h:
        return depth_img[y, x]
    return float('inf')

print("Autonomous Navigation Started")

with mujoco.viewer.launch_passive(mj_model, data) as viewer:
    while viewer.is_running():
        renderer.update_scene(data, camera=rgb_cam_id)
        rgb_img = renderer.render()
        
        bbox = detect_red_box(rgb_img)
        
        renderer.enable_depth_rendering()
        renderer.update_scene(data, camera=depth_cam_id)
        depth_img = renderer.render()
        renderer.disable_depth_rendering()
        
        detection_img = rgb_img.copy()
        target_speed = 0.0
        
        if bbox is not None:
            x, y, w, h = bbox
            center_x, center_y = x + w // 2, y + h // 2
            depth = get_depth_at_point(depth_img, center_x, center_y)
            
            cv2.rectangle(detection_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(detection_img, f"Depth: {depth:.2f}m", (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            if depth > SAFE_DISTANCE:
                distance_error = depth - SAFE_DISTANCE
                target_speed = min(TARGET_SPEED, distance_error * 0.15)
                status = "MOVING FORWARD"
                color = (0, 255, 0)
            else:
                target_speed = 0.0
                status = "STOPPED - SAFE DISTANCE"
                color = (0, 255, 255)
            
            cv2.putText(detection_img, status, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(detection_img, f"Pos: {data.qpos[0]:.2f}m", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        else:
            target_speed = 0.0
            cv2.putText(detection_img, "NO TARGET", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Smooth acceleration
        speed_diff = target_speed - current_speed
        if abs(speed_diff) > MAX_ACCEL:
            current_speed += MAX_ACCEL * np.sign(speed_diff)
        else:
            current_speed = target_speed
        
        # Apply velocity
        data.qvel[0] = current_speed  # X velocity
        data.qvel[1] = 0              # Y velocity  
        data.qvel[2] = 0              # Z velocity
        
        # Stabilize orientation - prevent tipping
        # qvel[3:6] are angular velocities (roll, pitch, yaw)
        data.qvel[3] *= 0.8   # Damp roll
        data.qvel[4] *= 0.8   # Damp pitch (prevent forward tipping)
        data.qvel[5] *= 0.8   # Damp yaw
        
        # Keep robot upright - reset orientation if tipping too much
        # qpos[3:7] is quaternion for orientation
        quat = data.qpos[3:7]
        # Extract pitch angle
        pitch = 2 * np.arcsin(np.clip(2 * (quat[0] * quat[2] - quat[3] * quat[1]), -1, 1))
        
        if abs(pitch) > 0.3:  # If tilting more than ~17 degrees
            # Stop immediately
            data.qvel[0] = 0
            current_speed = 0
            print(f"WARNING: Excessive tilt detected: {np.degrees(pitch):.1f} degrees")
        
        mujoco.mj_step(mj_model, data)
        
        depth_normalized = np.clip(depth_img, 0, 10) / 10.0
        cv2.imshow("Autonomous Navigation", cv2.cvtColor(detection_img, cv2.COLOR_RGB2BGR))
        cv2.imshow("Depth Camera", depth_normalized)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        viewer.sync()

cv2.destroyAllWindows()