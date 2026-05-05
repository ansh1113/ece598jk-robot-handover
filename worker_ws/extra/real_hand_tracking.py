import mujoco
import mujoco.viewer
import numpy as np
import cv2
import mediapipe as mp
import os

os.chdir("src/robotis_mujoco_menagerie/robotis_ffw")

mj_model = mujoco.MjModel.from_xml_path("scene_with_object.xml")
data = mujoco.MjData(mj_model)

# Robot arm setup
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

# Initialize mediapipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Open webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("\n" + "="*60)
print("REAL HAND TRACKING - TRUE MIRROR MODE")
print("="*60)
print("\nShow your RIGHT hand to the camera")
print("Robot will MIRROR like a reflection:")
print("  Your hand RIGHT → Robot arm LEFT")
print("  Your hand LEFT → Robot arm RIGHT")
print("  Your hand UP → Robot reaches FAR")
print("  Your hand DOWN → Robot retracts CLOSE")
print("  Your hand HORIZONTAL position → Robot arm HEIGHT")
print("\nPress Q to quit")
print("="*60 + "\n")

robot_arm_pose = np.array([0.0, 0.5, 0.0, -0.5, 0.0, 0.0, 0.0])

for i, addr in enumerate(robot_arm_qpos_addrs):
    data.qpos[addr] = robot_arm_pose[i]

mujoco.mj_forward(mj_model, data)

def map_hand_to_robot_workspace(palm_x, palm_y):
    """
    Map hand position to robot workspace with TRUE MIRRORING
    
    Your hand position → Robot position:
    - Hand X (left/right) → MIRRORED → Robot Y (left/right from robot's view)
    - Hand Y (up/down) → Robot X (forward/back reach)
    - Hand middle position → Robot Z (height)
    """
    
    # CRITICAL: Mirror the X axis by flipping
    # When you move right (palm_x increases), robot should move left
    x_mirrored = 1.0 - palm_x
    
    # Normalize to centered coordinates
    x_norm = (x_mirrored - 0.5)  # -0.5 to 0.5 (mirrored!)
    y_norm = (palm_y - 0.5)       # -0.5 to 0.5
    
    # Robot workspace mapping:
    
    # X (forward/backward): Controlled by YOUR hand's vertical position
    # Hand high (y_norm < 0) = reach far forward
    # Hand low (y_norm > 0) = retract back
    robot_x = 0.5 - y_norm * 0.4  # Range: 0.3m (low) to 0.7m (high)
    
    # Y (left/right from robot's perspective): Controlled by YOUR hand's horizontal position (MIRRORED)
    # Your right = robot's left (negative Y)
    # Your left = robot's right (less negative Y)
    robot_y = -0.35 - x_norm * 0.4  # Range: -0.55m to -0.15m
    
    # Z (height): Keep relatively constant, slight variation based on center
    robot_z = 1.3 + x_norm * 0.15  # Range: 1.15m to 1.45m
    
    return np.array([robot_x, robot_y, robot_z])

def compute_ik(target_world, current_q, max_iter=30):
    """Inverse kinematics"""
    q = current_q.copy()
    
    for _ in range(max_iter):
        for i, addr in enumerate(robot_arm_qpos_addrs):
            data.qpos[addr] = q[i]
        mujoco.mj_forward(mj_model, data)
        
        ee_pos = data.xpos[ee_body_id].copy()
        error = target_world - ee_pos
        
        if np.linalg.norm(error) < 0.02:
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
        q += 0.4 * (J_pinv @ error)
        
        for i, jname in enumerate(robot_arm_joints):
            jid = mj_model.joint(jname).id
            q[i] = np.clip(q[i], mj_model.jnt_range[jid][0], mj_model.jnt_range[jid][1])
    
    return q

with mujoco.viewer.launch_passive(mj_model, data) as viewer:
    while viewer.is_running() and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # DON'T flip frame - we'll do mirroring in the mapping
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process with mediapipe
        results = hands.process(frame_rgb)
        
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            
            # Draw hand landmarks
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2))
            
            # Get palm center (landmark 9)
            palm = hand_landmarks.landmark[9]
            palm_x = palm.x  # 0 = left, 1 = right (from YOUR perspective)
            palm_y = palm.y  # 0 = top, 1 = bottom
            
            # Map to robot workspace (WITH TRUE MIRRORING)
            target_pos = map_hand_to_robot_workspace(palm_x, palm_y)
            
            # Draw palm position
            palm_px = int(palm_x * frame.shape[1])
            palm_py = int(palm_y * frame.shape[0])
            cv2.circle(frame, (palm_px, palm_py), 10, (255, 0, 0), -1)
            
            # Draw guide lines
            cv2.line(frame, (320, 0), (320, 480), (255, 255, 0), 1)  # Vertical center
            cv2.line(frame, (0, 240), (640, 240), (255, 255, 0), 1)  # Horizontal center
            
            # Info overlay
            cv2.rectangle(frame, (5, 5), (635, 180), (0, 0, 0), -1)
            cv2.rectangle(frame, (5, 5), (635, 180), (100, 100, 100), 2)
            
            cv2.putText(frame, "MIRROR MODE ACTIVE", (15, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Show which direction you're moving
            direction_x = "CENTER"
            if palm_x < 0.4:
                direction_x = "LEFT → Robot moves RIGHT"
            elif palm_x > 0.6:
                direction_x = "RIGHT → Robot moves LEFT"
            
            direction_y = "CENTER"
            if palm_y < 0.4:
                direction_y = "HIGH → Robot reaches FAR"
            elif palm_y > 0.6:
                direction_y = "LOW → Robot retracts CLOSE"
            
            cv2.putText(frame, f"Hand: {direction_x}", 
                       (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, f"Hand: {direction_y}", 
                       (15, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.putText(frame, f"Palm coords: ({palm_x:.2f}, {palm_y:.2f})", 
                       (15, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.putText(frame, f"Robot target: X:{target_pos[0]:.2f} Y:{target_pos[1]:.2f} Z:{target_pos[2]:.2f}", 
                       (15, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            ee_pos = data.xpos[ee_body_id].copy()
            error = np.linalg.norm(target_pos - ee_pos)
            cv2.putText(frame, f"Error: {error*100:.1f}cm", 
                       (15, 165), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
            # Compute IK
            robot_arm_pose = compute_ik(target_pos, robot_arm_pose)
            
        else:
            cv2.putText(frame, "NO HAND DETECTED - Show your RIGHT hand", (15, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Update robot
        for i, addr in enumerate(robot_arm_qpos_addrs):
            data.qpos[addr] = robot_arm_pose[i]
        
        mujoco.mj_step(mj_model, data)
        viewer.sync()
        
        # Show webcam
        cv2.imshow("Hand Tracking - True Mirror (Q to quit)", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()