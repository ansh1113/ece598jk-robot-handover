"""
Human Hand Teleop Controller V2.1 (Bugfix)
===============================================
- Fixed MediaPipe naming bug (PINKY_FINGER_MCP -> PINKY_MCP)
- Robust 3D angle calculation
- "Active Zone" mapping
- Movement Filtering
"""

import mujoco
import mujoco.viewer
import numpy as np
import cv2
import time

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("ERROR: pip install mediapipe")

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Camera X-axis "Active Zone" (0.0 to 1.0)
ACTIVE_ZONE_MIN = 0.0
ACTIVE_ZONE_MAX = 1.0

# Robot Swing Limits (Matches XML)
ROBOT_SWING_MIN = -10.0  # Right (robot perspective)
ROBOT_SWING_MAX = 10.0   # Left (robot perspective)

# Finger Curl Thresholds (3D Angle based)
CURL_TRIGGER_THRESHOLD = 0.7  # 70% curled
CURL_RELEASE_THRESHOLD = 0.4  # 40% curled (hysteresis)

# ==============================================================================
# ROBUST HAND TRACKER
# ==============================================================================

class RobustHandTracker:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            max_num_hands=1,
            model_complexity=1, # Higher accuracy
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        
        self.cap = cv2.VideoCapture(0)
        
        # Tracking State
        self.raw_x = 0.5
        self.smoothed_x = 0.5
        self.last_valid_curl = 0.0 # 0=Open, 1=Closed
        self.velocity = 0.0
        self.last_time = time.time()
        
        self.hand_detected = False
        self.frame = None

    def calculate_3d_angle(self, a, b, c):
        """Calculates 3D angle at joint b (degrees)."""
        p1 = np.array([a.x, a.y, a.z])
        p2 = np.array([b.x, b.y, b.z])
        p3 = np.array([c.x, c.y, c.z])
        
        v1 = p1 - p2 
        v2 = p3 - p2 
        
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        
        if norm1 == 0 or norm2 == 0: return 0
            
        dot_product = np.dot(v1, v2)
        cosine_angle = max(-1.0, min(1.0, dot_product / (norm1 * norm2)))
        return np.degrees(np.arccos(cosine_angle))

    def get_finger_curl_state(self, landmarks):
        """
        Returns average curl (0.0=Straight, 1.0=Fully Bent).
        Uses 3D angles, independent of camera rotation.
        """
        mp_lm = self.mp_hands.HandLandmark
        lm = landmarks.landmark
        
        # FIXED: MediaPipe names for Pinky do NOT have "_FINGER_" in them
        fingers = [
            (mp_lm.INDEX_FINGER_MCP, mp_lm.INDEX_FINGER_PIP, mp_lm.INDEX_FINGER_TIP),
            (mp_lm.MIDDLE_FINGER_MCP, mp_lm.MIDDLE_FINGER_PIP, mp_lm.MIDDLE_FINGER_TIP),
            (mp_lm.RING_FINGER_MCP, mp_lm.RING_FINGER_PIP, mp_lm.RING_FINGER_TIP),
            (mp_lm.PINKY_MCP, mp_lm.PINKY_PIP, mp_lm.PINKY_TIP), # FIXED THIS LINE
        ]
        
        total_curl = 0
        
        for mcp, pip, tip in fingers:
            angle = self.calculate_3d_angle(lm[mcp], lm[pip], lm[tip])
            # Angle 180 = Straight, Angle 50 = Fully curled
            # Map 170->50 to 0.0->1.0
            curl = np.interp(angle, [50, 170], [1.0, 0.0])
            total_curl += curl
            
        avg_curl = total_curl / 4.0
        return avg_curl

    def update(self):
        ret, frame = self.cap.read()
        if not ret: return 0.5, 0.0
        
        # Mirror flip for intuitive control
        frame = cv2.flip(frame, 1)
        self.frame = frame.copy()
        h, w = self.frame.shape[:2]
        
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)
        
        dt = time.time() - self.last_time
        self.last_time = time.time()
        
        self.hand_detected = False
        target_curl = self.last_valid_curl # Default to holding last state
        
        if results.multi_hand_landmarks:
            landmarks = results.multi_hand_landmarks[0]
            self.mp_draw.draw_landmarks(self.frame, landmarks, self.mp_hands.HAND_CONNECTIONS)
            
            # 1. POSITION TRACKING
            wrist = landmarks.landmark[self.mp_hands.HandLandmark.WRIST]
            self.raw_x = wrist.x
            
            # Calc velocity (units per second)
            dx = abs(self.raw_x - self.smoothed_x)
            self.velocity = dx / (dt + 1e-6)
            
            # Smooth Position (Adaptive: fast movement = less smoothing)
            alpha = 0.2 if self.velocity < 0.5 else 0.6
            self.smoothed_x = (alpha * self.raw_x) + ((1 - alpha) * self.smoothed_x)
            
            # 2. CURL DETECTION (With Velocity Filter)
            if self.velocity < 1.5:
                raw_curl = self.get_finger_curl_state(landmarks)
                
                # Apply Hysteresis (Snap to Open/Close)
                if self.last_valid_curl < 0.5: # Currently Open
                    if raw_curl > CURL_TRIGGER_THRESHOLD:
                        target_curl = 1.0
                    else:
                        target_curl = 0.0
                else: # Currently Closed
                    if raw_curl < CURL_RELEASE_THRESHOLD:
                        target_curl = 0.0
                    else:
                        target_curl = 1.0
                
                self.last_valid_curl = target_curl
            else:
                # High velocity: Draw Warning
                cv2.putText(self.frame, "MOVING FAST - GRIP LOCKED", (10, 150),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

            self.hand_detected = True

        # Draw UI
        self._draw_ui(w, h, target_curl)
        
        return self.smoothed_x, self.last_valid_curl

    def _draw_ui(self, w, h, curl_state):
        # Draw Active Zone Box
        x1 = int(ACTIVE_ZONE_MIN * w)
        x2 = int(ACTIVE_ZONE_MAX * w)
        
        cv2.rectangle(self.frame, (x1, 10), (x2, h-10), (50, 50, 50), 2)
        cv2.putText(self.frame, "ACTIVE ZONE", (x1+10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
        
        if self.hand_detected:
            # Draw Stick
            px = int(self.smoothed_x * w)
            color = (0, 255, 0) if ACTIVE_ZONE_MIN < self.smoothed_x < ACTIVE_ZONE_MAX else (0, 0, 255)
            cv2.line(self.frame, (px, 0), (px, h), color, 2)
            
            # Draw Status
            status = "FIST (CLOSED)" if curl_state > 0.5 else "OPEN"
            col_status = (0, 165, 255) if curl_state > 0.5 else (0, 255, 0)
            cv2.putText(self.frame, f"STATUS: {status}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, col_status, 2)

    def close(self):
        self.cap.release()

# ==============================================================================
# MUJOCO CONTROLLER
# ==============================================================================

class SimController:
    def __init__(self, model, data):
        self.model = model
        self.data = data
        
        # Actuators
        self.act_swing = model.actuator("act_shoulder_swing").id
        
        self.fingers = []
        for name in ["index", "middle", "ring", "pinky", "thumb"]:
            self.fingers.append([
                model.actuator(f"act_{name}_mcp").id,
                model.actuator(f"act_{name}_pip" if name != "thumb" else f"act_{name}_mcp").id,
                model.actuator(f"act_{name}_dip" if name != "thumb" else f"act_{name}_ip").id,
            ])

        self.smooth_curl = 0.0
        self.smooth_swing = 0.0

    def update(self, hand_x_norm, curl_target):
        # 1. Map Position (With Active Zone Scaling)
        rel_x = (hand_x_norm - ACTIVE_ZONE_MIN) / (ACTIVE_ZONE_MAX - ACTIVE_ZONE_MIN)
        rel_x = np.clip(rel_x, 0.0, 1.0)
        
        # Map 0.0 (Left) -> +2.0 (Swing Left)
        # Map 1.0 (Right) -> -2.0 (Swing Right)
        target_swing = np.interp(rel_x, [0, 1], [ROBOT_SWING_MAX, ROBOT_SWING_MIN])
        
        # Exponential smoothing
        self.smooth_swing += (target_swing - self.smooth_swing) * 0.15
        self.data.ctrl[self.act_swing] = self.smooth_swing
        
        # 2. Map Fingers
        self.smooth_curl += (curl_target - self.smooth_curl) * 0.2
        
        for f_joints in self.fingers:
            self.data.ctrl[f_joints[0]] = self.smooth_curl * 1.5
            self.data.ctrl[f_joints[1]] = self.smooth_curl * 1.5
            self.data.ctrl[f_joints[2]] = self.smooth_curl * 1.0

# ==============================================================================
# MAIN LOOP
# ==============================================================================

def main():
    if not MEDIAPIPE_AVAILABLE: return

    xml_path = "human_hand.xml"
    print(f"Loading: {xml_path}")
    
    try:
        model = mujoco.MjModel.from_xml_path(xml_path)
        data = mujoco.MjData(model)
    except Exception as e:
        print(f"Failed to load XML. Make sure 'human_hand.xml' is updated.\nError: {e}")
        return

    tracker = RobustHandTracker()
    controller = SimController(model, data)
    
    viewer = mujoco.viewer.launch_passive(model, data)
    
    print("\n" + "="*60)
    print("ROBUST HAND TELEOP V2.1")
    print("="*60)
    print("1. Keep hand inside the GREY BOX (Active Zone) on screen.")
    print("2. Move Left/Right to swing arm.")
    print("3. Making a FIST will snap the robot hand closed.")
    print("4. Fast movements will LOCK the gripper to prevent glitches.")
    print("="*60)

    while viewer.is_running():
        x, curl = tracker.update()
        
        if tracker.hand_detected:
            controller.update(x, curl)
        
        mujoco.mj_step(model, data)
        
        if tracker.frame is not None:
            cv2.imshow("Hand Tracker V2", tracker.frame)
            
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
        viewer.sync()

    tracker.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()