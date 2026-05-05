"""
TEST SCRIPT: Hand Tracking Robot Arm Control
=============================================
EXACTLY like the working MediaPipe code, but tracking marker instead.

Original logic:
- hand_x from MediaPipe: 0.0 to 1.0
- hand_offset = hand_x - 0.5  → -0.5 to +0.5
- joint2_delta = hand_offset * MIRROR_SCALE * 2.0
- Only joint 2 moves, others locked to base pose

Now with marker:
- Get marker world X position
- Map it to 0.0 to 1.0 range
- Apply EXACT same formula
"""

import mujoco
import mujoco.viewer
import numpy as np
import cv2
import os

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("WARNING: MediaPipe not installed")


# ============================================================================
# CONFIG - Same as original
# ============================================================================

HAND_MIRROR_SCALE = 1.5
RELEASE_HOLD_TIME = 0.2


# ============================================================================
# SIMULATED HAND CONTROLLER
# ============================================================================

class SimulatedHandController:
    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.valid = True
        try:
            self.act_swing = model.actuator("act_shoulder_swing").id
            self.fingers = []
            for name in ["index", "middle", "ring", "pinky", "thumb"]:
                base = f"act_{name}_"
                p2 = "mcp" if name == "thumb" else "pip"
                p3 = "ip" if name == "thumb" else "dip"
                self.fingers.append([
                    model.actuator(base + "mcp").id,
                    model.actuator(base + p2).id,
                    model.actuator(base + p3).id,
                ])
            self.marker_geom_id = model.geom("palm_marker").id
            print(f"  Marker geom ID: {self.marker_geom_id}")
        except KeyError as e:
            print(f"SimHand Init Error: {e}")
            self.valid = False
            
        self.smooth_swing = 0.0
        self.smooth_curl = 0.0
        
        # Track the range of marker X positions to calibrate
        self.marker_x_min = None
        self.marker_x_max = None

    def update(self, norm_x, target_curl):
        if not self.valid: return
        
        # Map webcam X (0-1) to swing angle
        # When hand is on LEFT of webcam (x=0), swing positive (hand goes to one side)
        # When hand is on RIGHT of webcam (x=1), swing negative (hand goes to other side)
        target_swing = np.interp(norm_x, [0, 1], [0.5, -0.5])
        
        self.smooth_swing += (target_swing - self.smooth_swing) * 0.5
        self.data.ctrl[self.act_swing] = self.smooth_swing
        
        self.smooth_curl += (target_curl - self.smooth_curl) * 0.2
        for f in self.fingers:
            self.data.ctrl[f[0]] = self.smooth_curl * 1.5
            self.data.ctrl[f[1]] = self.smooth_curl * 1.5
            self.data.ctrl[f[2]] = self.smooth_curl * 1.0
            
    def get_marker_world_position(self):
        if not self.valid:
            return None
        pos = self.data.geom_xpos[self.marker_geom_id].copy()
        
        # Track min/max for calibration
        if self.marker_x_min is None:
            self.marker_x_min = pos[0]
            self.marker_x_max = pos[0]
        else:
            self.marker_x_min = min(self.marker_x_min, pos[0])
            self.marker_x_max = max(self.marker_x_max, pos[0])
        
        return pos
    
    def get_marker_x_normalized(self):
        """
        Get marker X position normalized to 0.0 - 1.0 range.
        This is the equivalent of MediaPipe's hand_x.
        """
        pos = self.get_marker_world_position()
        if pos is None:
            return 0.5
        
        # The marker swings based on shoulder_swing joint
        # When swing = +1.5, marker is at one X extreme
        # When swing = -1.5, marker is at other X extreme
        # We need to map this to 0-1
        
        # Based on the hand position at (0.8, 2.5) with euler -1.57,
        # the marker X range is roughly 0.3 to 1.3 (about 1m range centered at 0.8)
        marker_x = pos[0]
        
        # Map to 0-1: left side of range = 0, right side = 1
        # Marker at X=0.3 → 0.0, Marker at X=1.3 → 1.0
        x_min = 0.3
        x_max = 1.3
        
        normalized = (marker_x - x_min) / (x_max - x_min)
        normalized = np.clip(normalized, 0.0, 1.0)
        
        return normalized


# ============================================================================
# HAND TRACKER (WEBCAM) - For gesture detection only
# ============================================================================

class HandTracker:
    def __init__(self):
        if not MEDIAPIPE_AVAILABLE:
            raise RuntimeError("MediaPipe not available")
        
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Could not open webcam")
        
        print("✓ Webcam opened")
        
        self.hand_detected = False
        self.gesture = "NONE"
        self.hand_x = 0.5
        self.smoothed_x = 0.5
        self.frame = None
        self.curl = 0.0

    def calculate_3d_angle(self, a, b, c):
        p1 = np.array([a.x, a.y, a.z])
        p2 = np.array([b.x, b.y, b.z])
        p3 = np.array([c.x, c.y, c.z])
        v1, v2 = p1 - p2, p3 - p2 
        norm1, norm2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0: return 0
        return np.degrees(np.arccos(np.clip(np.dot(v1, v2)/(norm1*norm2), -1.0, 1.0)))
    
    def update(self) -> dict:
        ret, frame = self.cap.read()
        if not ret: 
            return {'detected': False, 'x': 0.5, 'curl': 0.0, 'gesture': "NONE"}
        
        frame = cv2.flip(frame, 1)
        self.frame = frame.copy()
        
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)
        
        self.hand_detected = False
        self.gesture = "NONE"
        
        if results.multi_hand_landmarks:
            lm = results.multi_hand_landmarks[0]
            self.mp_draw.draw_landmarks(self.frame, lm, self.mp_hands.HAND_CONNECTIONS)
            
            wrist = lm.landmark[self.mp_hands.HandLandmark.WRIST]
            self.hand_x = wrist.x
            self.smoothed_x = 0.4 * self.hand_x + 0.6 * self.smoothed_x
            
            mp_lm = self.mp_hands.HandLandmark
            fingers = [
                (mp_lm.INDEX_FINGER_MCP, mp_lm.INDEX_FINGER_PIP, mp_lm.INDEX_FINGER_TIP),
                (mp_lm.MIDDLE_FINGER_MCP, mp_lm.MIDDLE_FINGER_PIP, mp_lm.MIDDLE_FINGER_TIP),
                (mp_lm.RING_FINGER_MCP, mp_lm.RING_FINGER_PIP, mp_lm.RING_FINGER_TIP),
                (mp_lm.PINKY_MCP, mp_lm.PINKY_PIP, mp_lm.PINKY_TIP)
            ]
            
            total_ang = 0
            for m, p, t in fingers:
                total_ang += self.calculate_3d_angle(lm.landmark[m], lm.landmark[p], lm.landmark[t])
            avg_ang = total_ang / 4.0
            
            if avg_ang > 140:
                self.gesture = "PALM_OPEN"
                self.curl = 0.0
            elif avg_ang < 90:
                self.gesture = "GRASP_C"
                self.curl = 1.0
            else:
                self.gesture = "TRANSITION"
                self.curl = np.interp(avg_ang, [90, 140], [1.0, 0.0])
            
            self.hand_detected = True
            
            h, w = self.frame.shape[:2]
            color = (0, 255, 0) if self.gesture == "PALM_OPEN" else (0, 165, 255) if self.gesture == "GRASP_C" else (200, 200, 200)
            cv2.putText(self.frame, f"Gesture: {self.gesture}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.putText(self.frame, f"X: {self.smoothed_x:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cx = int(self.smoothed_x * w)
            cv2.line(self.frame, (cx, 0), (cx, h), color, 2)
        
        return {
            'detected': self.hand_detected,
            'x': self.smoothed_x,
            'curl': self.curl,
            'gesture': self.gesture
        }
    
    def close(self):
        if self.cap: 
            self.cap.release()


# ============================================================================
# ARM CONTROLLER - Simple, just like original
# ============================================================================

class ArmController:
    def __init__(self, model, data):
        self.model = model
        self.data = data
        
        self.joint_names = [f"arm_r_joint{i}" for i in range(1, 8)]
        self.joint_ids = [model.joint(name).id for name in self.joint_names]
        self.qpos_addrs = [model.jnt_qposadr[jid] for jid in self.joint_ids]
        self.actuator_ids = [model.actuator(f"actuator_{name}").id for name in self.joint_names]
        self.joint_limits = [(model.jnt_range[jid, 0], model.jnt_range[jid, 1]) for jid in self.joint_ids]
        self.gripper_actuator_ids = [model.actuator(f"actuator_gripper_r_joint{i}").id for i in range(1, 5)]
        
        print(f"  Joint 2 limits: {self.joint_limits[2]}")
    
    def get_joint_positions(self):
        return np.array([self.data.qpos[addr] for addr in self.qpos_addrs])
    
    def set_joint_positions(self, q):
        for i, aid in enumerate(self.actuator_ids):
            self.data.ctrl[aid] = np.clip(q[i], self.joint_limits[i][0], self.joint_limits[i][1])
    
    def set_gripper(self, closed):
        value = 0.70 if closed else 0.0
        for aid in self.gripper_actuator_ids:
            self.data.ctrl[aid] = value

    def set_gripper_value(self, value):
        for aid in self.gripper_actuator_ids:
            self.data.ctrl[aid] = value


# ============================================================================
# TEST CONTROLLER - EXACTLY like original _state_handover
# ============================================================================

class TestController:
    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.arm = ArmController(model, data)
        self.sim_hand = SimulatedHandController(model, data)
        
        # Start webcam for gesture + controlling sim hand
        self.hand_tracker = None
        if MEDIAPIPE_AVAILABLE:
            try:
                self.hand_tracker = HandTracker()
            except Exception as e:
                print(f"Could not start hand tracker: {e}")
        
        # Save the initial arm pose - this is our "handover_base_pose"
        self.handover_base_pose = self.arm.get_joint_positions().copy()
        self.smooth_joint2_target = self.handover_base_pose[2]
        
        print(f"  Base pose joint 2: {self.handover_base_pose[2]:.3f}")
        
        # State
        self.grasp_gesture_timer = 0.0
        self.gripper_release_value = 0.70
        self.releasing = False
        self.locked_release_joints = None
        
        self.arm.set_gripper(True)
        
        print("\n" + "="*60)
        print("TEST: Marker Tracking (Same as MediaPipe logic)")
        print("="*60)
        print("Move hand LEFT/RIGHT → sim hand moves → robot arm follows")
        print("Make FIST for 1 second → gripper opens")
        print("Press 'Q' to quit")
        print("="*60 + "\n")
    
    def update(self, dt):
        # Stop base
        self.data.qvel[0:6] = 0
        
        # Update webcam tracking
        hand_result = {'detected': False, 'x': 0.5, 'curl': 0.0, 'gesture': 'NONE'}
        if self.hand_tracker:
            hand_result = self.hand_tracker.update()
            
            # Use webcam to control simulated hand position AND curl
            if hand_result['detected']:
                self.sim_hand.update(hand_result['x'], hand_result['curl'])
        
        gesture = hand_result['gesture']
        
        # Get marker X normalized (equivalent to hand_result['x'] from MediaPipe)
        marker_x = self.sim_hand.get_marker_x_normalized()
        
        # === EXACT SAME LOGIC AS ORIGINAL _state_handover ===
        
        # Control parameters (V6 constants)
        target_smoothing = 0.4
        move_smoothing = 0.3
        
        # Default: keep current target
        raw_target = self.smooth_joint2_target
        
        if self.releasing:
            # RELEASING state - hard lock joints, open gripper
            if self.locked_release_joints is not None:
                self.arm.set_joint_positions(self.locked_release_joints)
            
            self.gripper_release_value = max(0.0, self.gripper_release_value - (2 * dt))
            self.arm.set_gripper_value(self.gripper_release_value)
            
            if self.gripper_release_value <= 0.01:
                print("✓ Released!")
                self.releasing = False
                self.gripper_release_value = 0.70
            
            return marker_x
        
        if gesture == "PALM_OPEN":
            # 1. Reset Release Timer
            self.grasp_gesture_timer = 0.0
            
            # 2. Tracking Logic - EXACTLY like original
            # marker_x is 0.0 to 1.0, just like hand_result['x'] was
            hand_offset = 0.5 - marker_x  # -0.5 to +0.5
            joint2_delta = hand_offset * HAND_MIRROR_SCALE * 2.0
            
            # Calculate Target
            raw_target = self.handover_base_pose[2] + joint2_delta
            
            # Overlay
            if self.hand_tracker and self.hand_tracker.frame is not None:
                cv2.putText(self.hand_tracker.frame, "MODE: TRACKING", (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(self.hand_tracker.frame, f"Marker X: {marker_x:.2f}", (10, 120), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

        elif gesture == "GRASP_C":
            # 1. Increment Timer
            self.grasp_gesture_timer += dt
            
            # 2. FREEZE TARGET
            raw_target = self.smooth_joint2_target
            
            print(f"  RELEASE TIMER: {self.grasp_gesture_timer:.1f} / {RELEASE_HOLD_TIME:.1f}")
            
            if self.grasp_gesture_timer > RELEASE_HOLD_TIME:
                print("✓ TIMER COMPLETE - RELEASING!")
                self.locked_release_joints = self.arm.get_joint_positions().copy()
                self.releasing = True
                
            # Overlay
            if self.hand_tracker and self.hand_tracker.frame is not None:
                bar_width = int((self.grasp_gesture_timer / RELEASE_HOLD_TIME) * 200)
                cv2.rectangle(self.hand_tracker.frame, (10, 100), (210, 120), (50, 50, 50), -1)
                cv2.rectangle(self.hand_tracker.frame, (10, 100), (10 + bar_width, 120), (0, 165, 255), -1)
                cv2.putText(self.hand_tracker.frame, "HOLD TO RELEASE", (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
        
        else:
            # NONE or TRANSITION - decay timer, return to center
            self.grasp_gesture_timer = max(0.0, self.grasp_gesture_timer - dt)
            raw_target = self.handover_base_pose[2]
            
            if self.hand_tracker and self.hand_tracker.frame is not None:
                cv2.putText(self.hand_tracker.frame, "WAITING...", (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 2)

        # === APPLY SMOOTHING (V6 LOGIC) ===
        
        # 1. Clamp Target
        raw_target = np.clip(raw_target, self.arm.joint_limits[2][0], self.arm.joint_limits[2][1])
        
        # 2. Smooth the Target
        self.smooth_joint2_target = (target_smoothing * raw_target + 
                                     (1 - target_smoothing) * self.smooth_joint2_target)
        
        # 3. Move Current Joint toward Smoothed Target
        current_q = self.arm.get_joint_positions()
        current_joint2 = current_q[2]
        
        new_joint2 = current_joint2 + (self.smooth_joint2_target - current_joint2) * move_smoothing
        
        # 4. Set Joints - LOCK ALL EXCEPT JOINT 2
        new_q = self.handover_base_pose.copy()
        new_q[2] = new_joint2
        self.arm.set_joint_positions(new_q)
        
        # Keep gripper closed
        self.arm.set_gripper(True)
        
        return marker_x
    
    def get_hand_frame(self):
        if self.hand_tracker:
            return self.hand_tracker.frame
        return None
    
    def close(self):
        if self.hand_tracker:
            self.hand_tracker.close()


# ============================================================================
# MAIN
# ============================================================================

def main():
    possible_paths = [
        "src/robotis_mujoco_menagerie/robotis_ffw/test_tracking_scene.xml",
        "pick_and_place_scene.xml",
    ]
    scene_path = next((p for p in possible_paths if os.path.exists(p)), None)
    
    if scene_path is None:
        print("ERROR: Cannot find scene XML")
        return
    
    full_path = os.path.abspath(scene_path)
    os.chdir(os.path.dirname(full_path))
    
    print(f"Loading: {os.path.basename(scene_path)}")
    model = mujoco.MjModel.from_xml_path(os.path.basename(scene_path))
    data = mujoco.MjData(model)
    
    # Sticky gripper
    for i in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)
        if name and ("gripper" in name or "finger" in name):
            model.geom_friction[i, 0] = 5.0
            model.geom_friction[i, 1] = 5.0
    
    # Set initial robot position: facing +Y, in front of hand
    print("\nSetting initial position...")
    data.qpos[0] = 0.67   # X
    data.qpos[1] = 1.35   # Y
    data.qpos[2] = 0.05  # Z
    data.qpos[3] = 0.7071  # qw (90 deg rotation)
    data.qpos[4] = 0.0
    data.qpos[5] = 0.0
    data.qpos[6] = 0.7071  # qz
    
    # Set arm pose (extended forward)
    arm_joint_names = [f"arm_r_joint{i}" for i in range(1, 8)]
    arm_pose = [0.0, -0.3, 0.0, -1.2, 0.0, -0.5, 0.0]
    
    for i, name in enumerate(arm_joint_names):
        try:
            joint_id = model.joint(name).id
            addr = model.jnt_qposadr[joint_id]
            data.qpos[addr] = arm_pose[i]
        except:
            pass
    
    mujoco.mj_forward(model, data)
    
    controller = TestController(model, data)
    viewer = mujoco.viewer.launch_passive(model, data)
    
    dt = model.opt.timestep
    
    print("\nRunning. Press 'Q' to quit.\n")
    
    while viewer.is_running():
        marker_x = controller.update(dt)
        mujoco.mj_step(model, data)
        
        # Show webcam with overlays
        hand_frame = controller.get_hand_frame()
        if hand_frame is not None:
            # Add marker X display
            cv2.putText(hand_frame, f"Marker norm X: {marker_x:.2f}", (10, 150), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            cv2.imshow("Webcam + Tracking", hand_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        viewer.sync()
    
    controller.close()
    cv2.destroyAllWindows()
    print("\nDone.")


if __name__ == "__main__":
    main()