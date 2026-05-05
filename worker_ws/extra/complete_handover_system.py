import mujoco
import mujoco.viewer
import numpy as np
import cv2
import os

os.chdir("src/robotis_mujoco_menagerie/robotis_ffw")

mj_model = mujoco.MjModel.from_xml_path("scene_with_object.xml")
data = mujoco.MjData(mj_model)

rgb_renderer = mujoco.Renderer(mj_model, height=480, width=640)
depth_renderer = mujoco.Renderer(mj_model, height=480, width=640)
rgb_cam_id = mj_model.camera("head_rgb").id

# IDs
palm_site_id = mj_model.site("palm_center").id

# Robot arm joints
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

# Human arm joints
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

print("\n" + "="*60)
print("COMPLETE HANDOVER SYSTEM")
print("="*60)
print("\nWorkflow:")
print("1. Human arm starts 3m away from robot")
print("2. Robot detects hand using camera + depth")
print("3. Robot approaches to ~0.5m (base movement)")
print("4. Robot locks base and tracks palm with arm (handover)")
print("\nControls: Arrow keys to move human arm | Q to quit")
print("="*60 + "\n")

# System states
STATE_SEARCHING = 0
STATE_APPROACHING = 1
STATE_HANDOVER = 2

current_state = STATE_SEARCHING

# Parameters
HANDOVER_DISTANCE = 0.5  # Distance to stop and start handover
APPROACH_SPEED = 0.5    # Base movement speed
MAX_ACCEL = 0.02

current_base_speed = 0.0
selected_joint = 0

# Initialize poses
human_arm_pose = np.array([0.3, -0.2, -1.5, 0.2, 0.0])
robot_arm_pose = np.array([0.0, 0.5, 0.0, -0.5, 0.0, 0.0, 0.0])

for i, addr in enumerate(human_qpos_addrs):
    data.qpos[addr] = human_arm_pose[i]

for i, addr in enumerate(robot_arm_qpos_addrs):
    data.qpos[addr] = robot_arm_pose[i]

mujoco.mj_forward(mj_model, data)

def get_camera_intrinsics():
    fovy = mj_model.cam_fovy[rgb_cam_id] * np.pi / 180
    height, width = 480, 640
    fy = (height / 2) / np.tan(fovy / 2)
    fx = fy
    cx, cy = width / 2, height / 2
    return fx, fy, cx, cy

def detect_hand_in_image(rgb_img):
    """Detect hand using skin color (simulates mediapipe)"""
    hsv = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV)
    lower_skin = np.array([0, 20, 70])
    upper_skin = np.array([20, 255, 255])
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) > 100:
            M = cv2.moments(largest)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                return cx, cy, mask, largest
    
    return None, None, mask, None

def get_depth_at_pixel(depth_img, px, py):
    h, w = depth_img.shape
    if 0 <= px < w and 0 <= py < h:
        return depth_img[py, px]
    return None

def pixel_to_camera_frame(px, py, depth):
    fx, fy, cx, cy = get_camera_intrinsics()
    return np.array([
        depth,
        (px - cx) * depth / fx,
        -(py - cy) * depth / fy
    ])

def camera_to_base_frame(point_cam):
    # Camera to world
    cam_pos = data.cam_xpos[rgb_cam_id].copy()
    cam_mat = data.cam_xmat[rgb_cam_id].reshape(3, 3)
    point_world = cam_pos + cam_mat @ point_cam
    
    # World to base
    base_id = mj_model.body("robot_base").id
    base_pos = data.xpos[base_id].copy()
    base_mat = data.xmat[base_id].reshape(3, 3)
    point_base = base_mat.T @ (point_world - base_pos)
    
    return point_base, point_world

def compute_ik(target_world, current_q, max_iter=50):
    q = current_q.copy()
    
    for _ in range(max_iter):
        for i, addr in enumerate(robot_arm_qpos_addrs):
            data.qpos[addr] = q[i]
        mujoco.mj_forward(mj_model, data)
        
        ee_pos = data.xpos[ee_body_id].copy()
        error = target_world - ee_pos
        
        if np.linalg.norm(error) < 0.01:
            break
        
        jacp = np.zeros((3, mj_model.nv))
        jacr = np.zeros((3, mj_model.nv))
        mujoco.mj_jacBody(mj_model, data, jacp, jacr, ee_body_id)
        
        J = np.zeros((3, 7))
        for i, addr in enumerate(robot_arm_qpos_addrs):
            for jid in range(mj_model.njnt):
                if mj_model.jnt_qposadr[jid] == addr:
                    J[:, i] = jacp[:, mj_model.jnt_dofadr[jid]]
                    break
        
        J_pinv = J.T @ np.linalg.inv(J @ J.T + 0.01 * np.eye(3))
        q += 0.5 * (J_pinv @ error)
        
        for i, jname in enumerate(robot_arm_joints):
            jid = mj_model.joint(jname).id
            q[i] = np.clip(q[i], mj_model.jnt_range[jid][0], mj_model.jnt_range[jid][1])
    
    return q

cv2.namedWindow("Handover System")

with mujoco.viewer.launch_passive(mj_model, data) as viewer:
    while viewer.is_running():
        # Update human arm
        for i, addr in enumerate(human_qpos_addrs):
            data.qpos[addr] = human_arm_pose[i]
        mujoco.mj_forward(mj_model, data)
        
        # Render
        rgb_renderer.update_scene(data, camera=rgb_cam_id)
        rgb_img = rgb_renderer.render()
        
        depth_renderer.enable_depth_rendering()
        depth_renderer.update_scene(data, camera=rgb_cam_id)
        depth_img = depth_renderer.render()
        depth_renderer.disable_depth_rendering()
        
        # === DETECTION PIPELINE ===
        hand_px, hand_py, mask, contour = detect_hand_in_image(rgb_img)
        
        hand_detected = False
        hand_distance = None
        palm_world = None
        palm_base = None
        
        if hand_px is not None:
            depth = get_depth_at_pixel(depth_img, hand_px, hand_py)
            if depth and depth > 0:
                hand_detected = True
                hand_distance = depth
                palm_cam = pixel_to_camera_frame(hand_px, hand_py, depth)
                palm_base, palm_world = camera_to_base_frame(palm_cam)
        
        # === STATE MACHINE ===
        target_speed = 0.0
        base_locked = False
        
        if not hand_detected:
            current_state = STATE_SEARCHING
            base_locked = True
            state_text = "SEARCHING"
            state_color = (0, 0, 255)
            
        elif hand_distance > HANDOVER_DISTANCE:
            current_state = STATE_APPROACHING
            # Proportional control
            error = hand_distance - HANDOVER_DISTANCE
            target_speed = min(APPROACH_SPEED, error * 0.15)
            base_locked = False
            state_text = f"APPROACHING ({hand_distance:.2f}m)"
            state_color = (255, 165, 0)
            
        else:
            current_state = STATE_HANDOVER
            base_locked = True
            state_text = "HANDOVER"
            state_color = (0, 255, 0)
            
            # Track palm with arm
            if palm_world is not None:
                robot_arm_pose = compute_ik(palm_world, robot_arm_pose)
        
        # === BASE CONTROL ===
        if base_locked:
            data.qvel[0] = 0
            data.qvel[1] = 0
            data.qvel[2] = 0
            current_base_speed = 0
        else:
            # Smooth acceleration
            speed_diff = target_speed - current_base_speed
            if abs(speed_diff) > MAX_ACCEL:
                current_base_speed += MAX_ACCEL * np.sign(speed_diff)
            else:
                current_base_speed = target_speed
            
            data.qvel[0] = current_base_speed
            data.qvel[1] = 0
            data.qvel[2] = 0

        # CRITICAL: Strong stabilization to prevent tipping
        data.qvel[3] *= 0.7  # Roll - prevent sideways tipping
        data.qvel[4] *= 0.7  # Pitch - prevent forward/back tipping
        data.qvel[5] *= 0.7  # Yaw - prevent spinning

        # Extra stability: If tilting too much, stop immediately
        quat = data.qpos[3:7]
        # Simple tilt check using quaternion
        tilt_magnitude = abs(quat[1]) + abs(quat[2])  # Roll and pitch components
        if tilt_magnitude > 0.3:
            print("WARNING: Excessive tilt detected, stopping base")
            data.qvel[0] = 0
            data.qvel[1] = 0
            data.qvel[2] = 0
            current_base_speed = 0
        
        # Update arm
        for i, addr in enumerate(robot_arm_qpos_addrs):
            data.qpos[addr] = robot_arm_pose[i]
        
        mujoco.mj_step(mj_model, data)
        
        # === VISUALIZATION ===
        display = rgb_img.copy()
        
        if hand_px:
            cv2.circle(display, (hand_px, hand_py), 10, (0, 255, 0), -1)
            if contour is not None:
                cv2.drawContours(display, [contour], -1, (0, 255, 0), 2)
        
        # Info panel
        cv2.rectangle(display, (5, 5), (635, 210), (0, 0, 0), -1)
        cv2.rectangle(display, (5, 5), (635, 210), (100, 100, 100), 2)
        
        y = 30
        cv2.putText(display, f"STATE: {state_text}", (15, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, state_color, 2)
        y += 40
        
        lock_txt = "LOCKED" if base_locked else "MOVING"
        lock_col = (255, 0, 0) if base_locked else (0, 255, 0)
        cv2.putText(display, f"BASE: {lock_txt}", (15, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, lock_col, 2)
        y += 35
        
        if hand_detected:
            cv2.putText(display, f"Hand: {hand_distance:.2f}m", (15, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y += 25
            cv2.putText(display, f"Robot: {data.qpos[0]:.2f}m", (15, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y += 25
            
            if current_state == STATE_HANDOVER:
                ee = data.xpos[ee_body_id]
                err = np.linalg.norm(palm_world - ee) if palm_world is not None else 0
                cv2.putText(display, f"Track err: {err*100:.1f}cm", (15, y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        y += 30
        cv2.putText(display, f"Human: {human_joint_names[selected_joint]}", (15, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        cv2.imshow("Handover System", cv2.cvtColor(display, cv2.COLOR_RGB2BGR))
        # Show mask for debugging
        cv2.imshow("Hand Detection Mask", mask)
        
        # Keyboard
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == 82:  # UP
            jid = mj_model.joint(human_joints[selected_joint]).id
            human_arm_pose[selected_joint] = min(
                human_arm_pose[selected_joint] + 0.05,
                mj_model.jnt_range[jid][1]
            )
        elif key == 84:  # DOWN
            jid = mj_model.joint(human_joints[selected_joint]).id
            human_arm_pose[selected_joint] = max(
                human_arm_pose[selected_joint] - 0.05,
                mj_model.jnt_range[jid][0]
            )
        elif key == 83:  # RIGHT
            selected_joint = (selected_joint + 1) % len(human_joints)
        elif key == 81:  # LEFT
            selected_joint = (selected_joint - 1) % len(human_joints)
        
        viewer.sync()

cv2.destroyAllWindows()