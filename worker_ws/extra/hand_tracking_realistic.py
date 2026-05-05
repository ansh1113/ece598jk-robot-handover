import mujoco
import mujoco.viewer
import numpy as np
import cv2
import os

os.chdir("src/robotis_mujoco_menagerie/robotis_ffw")

mj_model = mujoco.MjModel.from_xml_path("scene_with_object.xml")
data = mujoco.MjData(mj_model)

# Renderers
rgb_renderer = mujoco.Renderer(mj_model, height=480, width=640)
depth_renderer = mujoco.Renderer(mj_model, height=480, width=640)
rgb_cam_id = mj_model.camera("head_rgb").id

# IDs
palm_site_id = mj_model.site("palm_center").id
robot_arm_joints = [
    "arm_r_joint1", "arm_r_joint2", "arm_r_joint3", 
    "arm_r_joint4", "arm_r_joint5", "arm_r_joint6", "arm_r_joint7"
]
robot_arm_qpos_addrs = []
for jname in robot_arm_joints:
    jnt_id = mj_model.joint(jname).id
    qpos_addr = mj_model.jnt_qposadr[jnt_id]
    robot_arm_qpos_addrs.append(qpos_addr)

ee_body_id = mj_model.body("arm_r_link7").id

human_joints = [
    "human_shoulder_pitch", "human_shoulder_roll", "human_elbow", 
    "human_wrist_pitch", "human_wrist_yaw"
]
human_joint_names = [
    "Shoulder Pitch", "Shoulder Roll", "Elbow", 
    "Wrist Pitch", "Wrist Yaw"
]
human_qpos_addrs = []
for jname in human_joints:
    jnt_id = mj_model.joint(jname).id
    qpos_addr = mj_model.jnt_qposadr[jnt_id]
    human_qpos_addrs.append(qpos_addr)

print("\n=== TRULY REALISTIC Hand Tracking Pipeline ===")
print("NO CHEATING - Only using camera RGB + Depth")
print("Steps:")
print("1. Detect hand in RGB image (skin color detection)")
print("2. Get depth at hand centroid pixel")
print("3. Deproject pixel+depth → 3D camera frame")
print("4. Transform camera frame → robot base frame")
print("5. IK to move robot arm")
print("\n")

def get_camera_intrinsics():
    """Camera intrinsics from MuJoCo camera parameters"""
    fovy = mj_model.cam_fovy[rgb_cam_id] * np.pi / 180
    height, width = 480, 640
    
    fy = (height / 2) / np.tan(fovy / 2)
    fx = fy
    cx = width / 2
    cy = height / 2
    
    return fx, fy, cx, cy

def detect_hand_in_image(rgb_img):
    """
    STEP 1: Detect hand using COLOR detection (simulates mediapipe)
    In real world: mediapipe would give you palm landmark pixel coordinates
    Here: we detect skin color to find hand region
    
    Returns: (pixel_x, pixel_y) of hand centroid, or None
    """
    # Convert to HSV for skin detection
    hsv = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV)
    
    # Skin color range (for our simulated tan hand)
    lower_skin = np.array([0, 20, 70])
    upper_skin = np.array([20, 255, 255])
    
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    
    # Find largest contour (the hand)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Get largest contour
        largest = max(contours, key=cv2.contourArea)
        
        # Filter small noise
        if cv2.contourArea(largest) > 500:
            # Get centroid (simulates palm center)
            M = cv2.moments(largest)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                
                return cx, cy, mask, largest
    
    return None, None, mask, None

def get_depth_at_pixel(depth_img, pixel_x, pixel_y):
    """
    STEP 2: Get depth value at specific pixel from depth camera
    This is what an RGB-D camera would give you
    """
    h, w = depth_img.shape
    if 0 <= pixel_x < w and 0 <= pixel_y < h:
        return depth_img[pixel_y, pixel_x]
    return None

def pixel_to_camera_frame(pixel_x, pixel_y, depth):
    """
    STEP 3: Deproject 2D pixel + depth → 3D point in camera frame
    This uses camera intrinsics (calibrated or from specs)
    """
    fx, fy, cx, cy = get_camera_intrinsics()
    
    # Standard pinhole camera deprojection
    cam_x = depth
    cam_y = (pixel_x - cx) * depth / fx
    cam_z = -(pixel_y - cy) * depth / fy
    
    return np.array([cam_x, cam_y, cam_z])

def camera_to_robot_base_frame(point_in_camera):
    """
    STEP 4: Transform 3D point from camera frame → robot base frame
    Uses camera extrinsics (from URDF/calibration)
    """
    # Get camera pose in world
    cam_pos = data.cam_xpos[rgb_cam_id].copy()
    cam_mat = data.cam_xmat[rgb_cam_id].reshape(3, 3).copy()
    
    # Transform camera frame → world frame
    point_in_world = cam_pos + cam_mat @ point_in_camera
    
    # Get robot base pose
    base_body_id = mj_model.body("robot_base").id
    base_pos = data.xpos[base_body_id].copy()
    base_mat = data.xmat[base_body_id].reshape(3, 3).copy()
    
    # Transform world frame → base frame
    point_in_base = base_mat.T @ (point_in_world - base_pos)
    
    return point_in_base, point_in_world

def compute_jacobian_ik(target_pos, current_q, max_iter=50):
    """
    STEP 5: Inverse kinematics using Jacobian
    """
    q = current_q.copy()
    
    for iteration in range(max_iter):
        for i, addr in enumerate(robot_arm_qpos_addrs):
            data.qpos[addr] = q[i]
        
        mujoco.mj_forward(mj_model, data)
        
        ee_pos = data.xpos[ee_body_id].copy()
        error = target_pos - ee_pos
        error_mag = np.linalg.norm(error)
        
        if error_mag < 0.01:
            break
        
        jacp = np.zeros((3, mj_model.nv))
        jacr = np.zeros((3, mj_model.nv))
        mujoco.mj_jacBody(mj_model, data, jacp, jacr, ee_body_id)
        
        J = np.zeros((3, 7))
        for i, addr in enumerate(robot_arm_qpos_addrs):
            jnt_id = None
            for j_id in range(mj_model.njnt):
                if mj_model.jnt_qposadr[j_id] == addr:
                    jnt_id = j_id
                    break
            
            if jnt_id is not None:
                dof_addr = mj_model.jnt_dofadr[jnt_id]
                J[:, i] = jacp[:, dof_addr]
        
        lambda_damping = 0.01
        J_pinv = J.T @ np.linalg.inv(J @ J.T + lambda_damping * np.eye(3))
        dq = J_pinv @ error
        
        alpha = 0.5
        q = q + alpha * dq
        
        for i, jname in enumerate(robot_arm_joints):
            jnt_id = mj_model.joint(jname).id
            q[i] = np.clip(q[i], 
                          mj_model.jnt_range[jnt_id][0], 
                          mj_model.jnt_range[jnt_id][1])
    
    return q

# Initialize
selected_joint = 0
human_arm_pose = np.array([0.3, -0.2, -1.5, 0.2, 0.0])
robot_arm_pose = np.array([0.0, 0.5, 0.0, -0.5, 0.0, 0.0, 0.0])

for i, addr in enumerate(human_qpos_addrs):
    data.qpos[addr] = human_arm_pose[i]

for i, addr in enumerate(robot_arm_qpos_addrs):
    data.qpos[addr] = robot_arm_pose[i]

mujoco.mj_forward(mj_model, data)

cv2.namedWindow("Robot Camera View - NO CHEATING!")
cv2.namedWindow("Hand Detection Mask")

tracking_active = False
palm_target_world = None

with mujoco.viewer.launch_passive(mj_model, data) as viewer:
    while viewer.is_running():
        # Set human arm
        for i, addr in enumerate(human_qpos_addrs):
            data.qpos[addr] = human_arm_pose[i]
        
        mujoco.mj_forward(mj_model, data)
        
        # === TRULY REALISTIC PIPELINE - NO CHEATING ===
        
        # Render RGB image (what camera sees)
        rgb_renderer.update_scene(data, camera=rgb_cam_id)
        rgb_img = rgb_renderer.render()
        
        # Render Depth image (from depth camera)
        depth_renderer.enable_depth_rendering()
        depth_renderer.update_scene(data, camera=rgb_cam_id)
        depth_img = depth_renderer.render()
        depth_renderer.disable_depth_rendering()
        
        # STEP 1: Detect hand in RGB image (simulates mediapipe)
        hand_pixel_x, hand_pixel_y, mask, contour = detect_hand_in_image(rgb_img)
        
        pipeline_success = False
        palm_in_camera = None
        palm_in_base = None
        
        if hand_pixel_x is not None:
            # STEP 2: Get depth at hand pixel
            depth_at_hand = get_depth_at_pixel(depth_img, hand_pixel_x, hand_pixel_y)
            
            if depth_at_hand is not None and depth_at_hand > 0:
                # STEP 3: Deproject to 3D camera frame
                palm_in_camera = pixel_to_camera_frame(hand_pixel_x, hand_pixel_y, depth_at_hand)
                
                # STEP 4: Transform to robot base frame
                palm_in_base, palm_target_world = camera_to_robot_base_frame(palm_in_camera)
                
                pipeline_success = True
        
        # STEP 5: IK only if we successfully detected hand
        if pipeline_success and palm_target_world is not None:
            robot_arm_pose = compute_jacobian_ik(palm_target_world, robot_arm_pose)
            tracking_active = True
        else:
            tracking_active = False
        
        # Apply robot arm
        for i, addr in enumerate(robot_arm_qpos_addrs):
            data.qpos[addr] = robot_arm_pose[i]
        
        mujoco.mj_step(mj_model, data)
        
        # === VISUALIZATION ===
        display_img = rgb_img.copy()
        
        # Draw detected hand
        if hand_pixel_x is not None:
            cv2.circle(display_img, (hand_pixel_x, hand_pixel_y), 10, (0, 255, 0), -1)
            cv2.putText(display_img, "HAND DETECTED", (hand_pixel_x + 15, hand_pixel_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            if contour is not None:
                cv2.drawContours(display_img, [contour], -1, (0, 255, 0), 2)
        
        # Info overlay
        cv2.rectangle(display_img, (5, 5), (635, 240), (0, 0, 0), -1)
        cv2.rectangle(display_img, (5, 5), (635, 240), (100, 100, 100), 2)
        
        y = 25
        status_color = (0, 255, 0) if tracking_active else (0, 0, 255)
        status_text = "TRACKING ACTIVE" if tracking_active else "NO HAND DETECTED"
        cv2.putText(display_img, f"Status: {status_text}", (15, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        y += 30
        
        cv2.putText(display_img, "REALISTIC PIPELINE (NO CHEATING):", (15, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        y += 25
        
        if hand_pixel_x is not None:
            cv2.putText(display_img, f"1. Detected at pixel: ({hand_pixel_x}, {hand_pixel_y})", 
                       (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            y += 22
            
            if depth_at_hand is not None:
                cv2.putText(display_img, f"2. Depth: {depth_at_hand:.3f}m", 
                           (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
                y += 22
                
                if palm_in_camera is not None:
                    cv2.putText(display_img, f"3. Camera: ({palm_in_camera[0]:.2f}, {palm_in_camera[1]:.2f}, {palm_in_camera[2]:.2f})", 
                               (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
                    y += 22
                    
                    if palm_in_base is not None:
                        cv2.putText(display_img, f"4. Base: ({palm_in_base[0]:.2f}, {palm_in_base[1]:.2f}, {palm_in_base[2]:.2f})", 
                                   (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
                        y += 22
                        
                        cv2.putText(display_img, f"5. IK: Moving arm to target", 
                                   (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
        else:
            cv2.putText(display_img, "1-5. Waiting for hand detection...", 
                       (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 100, 100), 1)
        
        y += 30
        cv2.putText(display_img, f"Control: {human_joint_names[selected_joint]} = {human_arm_pose[selected_joint]:.2f}", 
                   (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        cv2.imshow("Robot Camera View - NO CHEATING!", cv2.cvtColor(display_img, cv2.COLOR_RGB2BGR))
        cv2.imshow("Hand Detection Mask", mask)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == 82:  # UP
            jnt_id = mj_model.joint(human_joints[selected_joint]).id
            human_arm_pose[selected_joint] = min(
                human_arm_pose[selected_joint] + 0.05,
                mj_model.jnt_range[jnt_id][1]
            )
        elif key == 84:  # DOWN
            jnt_id = mj_model.joint(human_joints[selected_joint]).id
            human_arm_pose[selected_joint] = max(
                human_arm_pose[selected_joint] - 0.05,
                mj_model.jnt_range[jnt_id][0]
            )
        elif key == 83:  # RIGHT
            selected_joint = (selected_joint + 1) % len(human_joints)
        elif key == 81:  # LEFT
            selected_joint = (selected_joint - 1) % len(human_joints)
        
        viewer.sync()

cv2.destroyAllWindows()