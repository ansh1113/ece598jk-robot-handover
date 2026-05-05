import cv2
import mediapipe as mp
import numpy as np
import math
import time

# ============================================================================
# CONFIGURATION
# ============================================================================
WIDTH, HEIGHT = 640, 480

# 3D ANGLE THRESHOLDS (Degrees)
# A straight finger is ~180 degrees. A fist is ~45-90 degrees.
# We set a strict threshold for "Open" to avoid false positives.
ANGLE_OPEN_THRESHOLD = 150.0  
ANGLE_THUMB_OPEN_THRESHOLD = 140.0 # Thumb is naturally slightly curved

# ============================================================================
# GESTURE DETECTOR CLASS
# ============================================================================
class HandGestureProcessor:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        # State Simulation
        self.robot_x_pos = 0.5     # Virtual robot position (0.0 to 1.0)
        self.gripper_state = 0.0   # 0.0 (Closed) to 1.0 (Open)
        
    def calculate_3d_angle(self, a, b, c):
        """
        Calculates the angle at joint 'b' formed by points a, b, c in 3D space.
        Returns degrees (0 to 180).
        180 means straight line a-b-c.
        90 means right angle.
        """
        # Convert landmarks to numpy arrays (x, y, z)
        # MediaPipe z is relative depth.
        p1 = np.array([a.x, a.y, a.z])
        p2 = np.array([b.x, b.y, b.z])
        p3 = np.array([c.x, c.y, c.z])
        
        # Vectors: BA and BC
        # We want the angle inside the joint, so we define vectors pointing OUT from the joint
        v1 = p1 - p2 
        v2 = p3 - p2 
        
        # Normalize vectors
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        
        if norm1 == 0 or norm2 == 0:
            return 0
            
        # Dot product
        dot_product = np.dot(v1, v2)
        
        # Clamp value to [-1, 1] to handle float precision errors for arccos
        cosine_angle = max(-1.0, min(1.0, dot_product / (norm1 * norm2)))
        
        angle_rad = np.arccos(cosine_angle)
        return np.degrees(angle_rad)

    def is_finger_open_3d(self, hand_landmarks, finger_name):
        """
        Checks if finger is open using 3D joint angles.
        Robust to camera perspective (pointing at camera).
        """
        lm = hand_landmarks.landmark
        mp_lm = self.mp_hands.HandLandmark
        
        if finger_name == "THUMB":
            # Thumb: Angle at IP joint (MCP - IP - TIP)
            # Or Angle at MCP (CMC - MCP - IP) - let's use MCP for better "sticking out" detection
            angle = self.calculate_3d_angle(lm[mp_lm.THUMB_CMC], lm[mp_lm.THUMB_MCP], lm[mp_lm.THUMB_IP])
            return angle > ANGLE_THUMB_OPEN_THRESHOLD, angle
            
        elif finger_name == "INDEX":
            # Angle at PIP (MCP - PIP - TIP)
            angle = self.calculate_3d_angle(lm[mp_lm.INDEX_FINGER_MCP], lm[mp_lm.INDEX_FINGER_PIP], lm[mp_lm.INDEX_FINGER_TIP])
            return angle > ANGLE_OPEN_THRESHOLD, angle
            
        elif finger_name == "MIDDLE":
            angle = self.calculate_3d_angle(lm[mp_lm.MIDDLE_FINGER_MCP], lm[mp_lm.MIDDLE_FINGER_PIP], lm[mp_lm.MIDDLE_FINGER_TIP])
            return angle > ANGLE_OPEN_THRESHOLD, angle

        elif finger_name == "RING":
            angle = self.calculate_3d_angle(lm[mp_lm.RING_FINGER_MCP], lm[mp_lm.RING_FINGER_PIP], lm[mp_lm.RING_FINGER_TIP])
            return angle > ANGLE_OPEN_THRESHOLD, angle

        elif finger_name == "PINKY":
            angle = self.calculate_3d_angle(lm[mp_lm.PINKY_MCP], lm[mp_lm.PINKY_PIP], lm[mp_lm.PINKY_TIP])
            return angle > ANGLE_OPEN_THRESHOLD, angle
            
        return False, 0.0

    def process(self, frame):
        # Flip frame for mirror effect
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)
        
        gesture_detected = "NONE"
        fingers_status = [] # Booleans
        finger_angles = []  # Float degrees
        wrist_x = 0.5
        
        if results.multi_hand_landmarks:
            landmarks = results.multi_hand_landmarks[0]
            self.mp_draw.draw_landmarks(frame, landmarks, self.mp_hands.HAND_CONNECTIONS)
            
            # 1. Analyze Fingers (3D Angles)
            fingers = ["THUMB", "INDEX", "MIDDLE", "RING", "PINKY"]
            open_count = 0
            
            for f in fingers:
                is_open, angle = self.is_finger_open_3d(landmarks, f)
                fingers_status.append(is_open)
                finger_angles.append(angle)
                if is_open:
                    open_count += 1
            
            # 2. Determine Gesture
            # PALM_OPEN: Strict check. At least 5 fingers must be very straight.
            # We relax it to 4 in case thumb is slightly bent.
            if open_count >= 4:
                gesture_detected = "PALM_OPEN"
            
            # GRASP_C:
            # - NOT Palm Open (so hands are curved)
            # - But NOT a Fist (0 or 1 finger open)
            # - Specifically: "Half curling". 
            # - A natural grasp usually has angles around 90-140 degrees.
            # - Let's check if fingers are in "Grasp Range" (e.g., angle < 140 but > 60)
            elif open_count <= 2: 
                # This catches the condition where fingers are curled significantly
                gesture_detected = "GRASP_C"
            else:
                gesture_detected = "TRANSITION"

            # 3. Get Wrist Position
            wrist = landmarks.landmark[self.mp_hands.HandLandmark.WRIST]
            wrist_x = wrist.x 

        return frame, gesture_detected, wrist_x, fingers_status, finger_angles

# ============================================================================
# MAIN LOOP
# ============================================================================
def main():
    cap = cv2.VideoCapture(0)
    processor = HandGestureProcessor()
    
    print("------------------------------------------------")
    print("GESTURE TEST SCRIPT (3D ANGLE V2)")
    print("------------------------------------------------")

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        frame = cv2.resize(frame, (WIDTH, HEIGHT))
        
        # Process Hand
        frame, gesture, wrist_x, fingers_status, finger_angles = processor.process(frame)
        
        overlay = frame.copy()
        
        # ==========================================================
        # LOGIC
        # ==========================================================
        status_text = "MODE: WAITING"
        status_color = (100, 100, 100)

        if gesture == "PALM_OPEN":
            processor.robot_x_pos = wrist_x
            processor.gripper_state = max(0.0, processor.gripper_state - 0.05)
            status_text = "MODE: MOVING ARM (TRACKING)"
            status_color = (0, 255, 0) # Green

        elif gesture == "GRASP_C":
            # Freeze position
            processor.gripper_state = min(1.0, processor.gripper_state + 0.02)
            status_text = "MODE: RELEASING (GRASP DETECTED)"
            status_color = (0, 165, 255) # Orange
            
        # ==========================================================
        # VISUALIZATION
        # ==========================================================
        
        # Draw Robot Slider
        cv2.rectangle(overlay, (50, 400), (590, 420), (50, 50, 50), -1)
        robot_px = int(50 + processor.robot_x_pos * 540)
        cv2.circle(overlay, (robot_px, 410), 15, status_color, -1)
        
        # Draw Gripper Bar
        cv2.rectangle(overlay, (600, 100), (620, 300), (50, 50, 50), -1)
        fill_height = int(200 * processor.gripper_state)
        cv2.rectangle(overlay, (600, 300 - fill_height), (620, 300), (0, 255, 255), -1)

        # Header
        cv2.rectangle(overlay, (0, 0), (640, 80), (0, 0, 0), -1)
        cv2.putText(overlay, f"GESTURE: {gesture}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(overlay, status_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        
        # Detailed Debug for Angles
        if finger_angles:
            # Display angle values for calibration
            f_names = ["Thmb", "Indx", "Midl", "Ring", "Pink"]
            start_y = 100
            cv2.putText(overlay, "3D ANGLES:", (10, start_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            for i, angle in enumerate(finger_angles):
                # Color code: Green = Straight (>150), Red = Curled
                col = (0, 255, 0) if fingers_status[i] else (0, 0, 255)
                text = f"{f_names[i]}: {int(angle)} deg"
                cv2.putText(overlay, text, (10, start_y + 20 + i*20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)

        alpha = 0.8
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

        cv2.imshow("Gesture Test", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()