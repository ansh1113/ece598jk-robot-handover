"""
VISION-BASED PICK AND PLACE - V17
=================================
V16 + LLM integration for task planning using Anthropic Claude API (switched from xAI).
V16 + Adjusted YOLO conf to 0.1 for better sim detection.
V16 + Added domain randomization in renderer for sim-to-real.
V16 + Replaced FSM with LLM-guided planning.
Requires: ANTHROPIC_API_KEY env var for Claude API.
"""

import mujoco
import mujoco.viewer
import numpy as np
import cv2
import os
import math
import time
import requests
import json
from enum import Enum, auto
from dataclasses import dataclass
from typing import Optional, Tuple, List

# MediaPipe for hand tracking
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("WARNING: MediaPipe not installed. Run: pip install mediapipe")

# YOLO for object detection
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("WARNING: Ultralytics not installed. Run: pip install ultralytics")


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class Config:
    # Base movement
    STANDOFF_DISTANCE: float = 0.40
    BASE_SPEED: float = 0.60
    
    # Gripper positioning  
    HOVER_HEIGHT: float = 0.10
    GRASP_DEPTH: float = 0.09
    
    # World-frame offset
    POSITION_X_OFFSET: float = -0.04
    POSITION_Y_OFFSET: float = 0.04
    
    # Grasp center offset from link7
    GRASP_CENTER_OFFSET: np.ndarray = None
    
    # Timeouts
    POSITIONING_TIMEOUT: float = 5.0
    DESCENDING_TIMEOUT: float = 4.0
    GRASPING_TIME: float = 0.2
    
    # Handover settings
    HAND_MIRROR_SCALE: float = 1.5
    RELEASE_HOLD_TIME: float = 0.2
    
    # Gesture Thresholds
    ANGLE_OPEN_THRESHOLD: float = 150.0  
    ANGLE_THUMB_OPEN_THRESHOLD: float = 140.0
    
    # YOLO thresholds - lowered conf for sim
    YOLO_CONFIDENCE: float = 0.1
    YOLO_AREA_MIN: int = 50
    YOLO_AREA_MAX: int = 20000
    YOLO_CUP_CLASS: int = 41  # COCO 'cup' - may detect as bottle (39), adjust if needed
    
    # Domain randomization ranges
    RAND_LIGHT_INTENSITY: Tuple[float, float] = (0.5, 1.5)
    RAND_LIGHT_DIR: float = 0.2
    RAND_COLOR_PERTURB: float = 0.1
    
    # LLM (Claude)
    CLAUDE_MODEL: str = "claude-3-5-sonnet-20240620"
    MAX_TOKENS: int = 512
    TEMPERATURE: float = 0.7
    
    def __post_init__(self):
        if self.GRASP_CENTER_OFFSET is None:
            self.GRASP_CENTER_OFFSET = np.array([0.0, 0.0, -0.10])
        self.api_key = os.getenv('ANTHROPIC_API_KEY')
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set")


CONFIG = Config()


# ============================================================================
# LLM PLANNER (Using Claude API)
# ============================================================================

def query_llm(prompt: str) -> str:
    """Query Anthropic Claude API for planning."""
    url = "https://api.anthropic.com/v1/messages"
    headers = {
        "Content-Type": "application/json",
        "x-api-key": CONFIG.api_key,
        "anthropic-version": "2023-06-01"
    }
    data = {
        "model": CONFIG.CLAUDE_MODEL,
        "max_tokens": CONFIG.MAX_TOKENS,
        "temperature": CONFIG.TEMPERATURE,
        "messages": [{"role": "user", "content": prompt}]
    }
    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        return response.json()['content'][0]['text']
    else:
        raise RuntimeError(f"Claude API error: {response.text}")


# ============================================================================
# SIMULATED HAND CONTROLLER
# ============================================================================

class SimulatedHandController:
    """Controls the MuJoCo 'human_hand' body based on Real Hand inputs."""
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
        except KeyError as e:
            print(f"SimHand Init Error: {e}. Hand will be static.")
            self.valid = False
            
        self.smooth_swing = 0.0
        self.smooth_curl = 0.0

    def update(self, norm_x, target_curl):
        if not self.valid: return
        
        target_swing = np.interp(norm_x, [0, 1], [1.0, -1.0])
        
        self.smooth_swing += (target_swing - self.smooth_swing) * 0.5
        self.data.ctrl[self.act_swing] = self.smooth_swing
        
        self.smooth_curl += (target_curl - self.smooth_curl) * 0.2
        
        for f in self.fingers:
            self.data.ctrl[f[0]] = self.smooth_curl * 1.5
            self.data.ctrl[f[1]] = self.smooth_curl * 1.5
            self.data.ctrl[f[2]] = self.smooth_curl * 1.0
            
    def get_curl_value(self):
        return self.smooth_curl
    
    def get_marker_position(self):
        if not self.valid:
            return None
        return self.data.geom_xpos[self.marker_geom_id].copy()
    
    def get_marker_x_normalized(self):
        pos = self.get_marker_position()
        if pos is None:
            return 0.5
        
        marker_x = pos[0]
        
        x_min = 0.3
        x_max = 1.3
        
        normalized = (marker_x - x_min) / (x_max - x_min)
        normalized = np.clip(normalized, 0.0, 1.0)
        
        return normalized


# ============================================================================
# MEDIAPIPE HAND TRACKER
# ============================================================================

class HandTracker:
    """Tracks hand using laptop webcam via MediaPipe."""
    
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
            for i in [1, 2]:
                self.cap = cv2.VideoCapture(i)
                if self.cap.isOpened():
                    break
        
        if not self.cap.isOpened():
            raise RuntimeError("Could not open webcam")
        
        print("✓ Webcam opened for hand tracking")
        
        self.hand_detected = False
        self.gesture = "NONE"
        self.hand_x = 0.5
        self.smoothed_x = 0.5
        self.frame = None
        self.last_valid_curl = 0.0

    def calculate_3d_angle(self, a, b, c):
        v1 = np.array([a.x - b.x, a.y - b.y, a.z - b.z])
        v2 = np.array([c.x - b.x, c.y - b.y, c.z - b.z])
        
        norm1, norm2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0: return 0
        
        dot_product = np.dot(v1, v2)
        return np.degrees(np.arccos(np.clip(dot_product/(norm1*norm2), -1.0, 1.0)))
    
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
            elif avg_ang < 90:
                self.gesture = "GRASP_C"
            else:
                self.gesture = "TRANSITION"
            
            raw_curl = np.interp(avg_ang, [60, 160], [1.0, 0.0])
            
            if self.last_valid_curl < 0.5:
                self.last_valid_curl = 1.0 if raw_curl > 0.7 else 0.0
            else:
                self.last_valid_curl = 0.0 if raw_curl < 0.3 else 1.0
                
            self.hand_detected = True
            
            h, w = self.frame.shape[:2]
            color = (0, 255, 0) if self.gesture == "PALM_OPEN" else (0, 165, 255)
            cv2.putText(self.frame, f"GESTURE: {self.gesture}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cx = int(self.smoothed_x * w)
            cv2.line(self.frame, (cx, 0), (cx, h), color, 2)
        else:
            cv2.putText(self.frame, "NO HAND DETECTED", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        return {
            'detected': self.hand_detected,
            'x': self.smoothed_x,
            'curl': self.last_valid_curl,
            'gesture': self.gesture
        }
    
    def close(self):
        if self.cap: self.cap.release()


# ============================================================================
# VISION SYSTEM
# ============================================================================

class VisionSystem:
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        self.model = model
        self.data = data
        self.width = 640
        self.height = 480
        self.fovy = 90.0
        self.rgb_renderer = mujoco.Renderer(model, self.height, self.width)
        self.depth_renderer = mujoco.Renderer(model, self.height, self.width)
        self.depth_renderer.enable_depth_rendering()
        self.cam_id = model.camera("head_rgb").id
        self.cup_detected = False
        self.table_detected = False
        self.cup_position = None
        self.cup_top_position = None
        self.table_position = None
        self._position_history = []
        self.debug_image = None
        
        self.last_known_cup_position = None
        self.last_known_cup_top = None
        self.last_known_table_position = None
        
        if YOLO_AVAILABLE:
            self.yolo_model = YOLO('yolov8n.pt')
            print("✓ YOLO model loaded")
        else:
            self.yolo_model = None
            print("WARNING: YOLO not available")
        
        self._hsv_green_lower = np.array([40, 100, 100])
        self._hsv_green_upper = np.array([80, 255, 255])
        self._morph_kernel = np.ones((3, 3), np.uint8)
    
    def _apply_domain_randomization(self):
        # Randomize light intensity and direction
        light_id = 0  # Assume first light
        self.model.light_diffuse[light_id] *= np.random.uniform(CONFIG.RAND_LIGHT_INTENSITY[0], CONFIG.RAND_LIGHT_INTENSITY[1])
        self.model.light_dir[light_id] += np.random.normal(0, CONFIG.RAND_LIGHT_DIR, 3)
        self.model.light_dir[light_id] = self.model.light_dir[light_id] / np.linalg.norm(self.model.light_dir[light_id])
        
        # Randomize material colors
        for mat_id in range(self.model.nmat):
            self.model.mat_rgba[mat_id][:3] += np.random.normal(0, CONFIG.RAND_COLOR_PERTURB, 3)
            self.model.mat_rgba[mat_id][:3] = np.clip(self.model.mat_rgba[mat_id][:3], 0, 1)
    
    def _detect_objects_yolo(self, rgb: np.ndarray):
        if not YOLO_AVAILABLE or self.yolo_model is None:
            return False, None, None, False, None
        
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        results = self.yolo_model(bgr, conf=CONFIG.YOLO_CONFIDENCE, verbose=False)
        
        if not results or not results[0].boxes:
            return False, None, None, False, None
        
        boxes = results[0].boxes
        cup_center = None
        cup_bbox = None
        table_center = None
        cup_area = 0
        table_area = 0
        
        for box in boxes:
            cls = int(box.cls)
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            area = (x2 - x1) * (y2 - y1)
            if CONFIG.YOLO_AREA_MIN < area < CONFIG.YOLO_AREA_MAX:
                if cls == CONFIG.YOLO_CUP_CLASS:  # cup
                    if area > cup_area:
                        cup_area = area
                        cup_center = ((x1 + x2) // 2, (y1 + y2) // 2)
                        cup_bbox = (x1, y1, x2, y2)
                elif cls == 60:  # dining table (COCO class 60)
                    if area > table_area:
                        table_area = area
                        table_center = ((x1 + x2) // 2, (y1 + y2) // 2)
        
        cup_found = cup_center is not None
        table_found = table_center is not None
        
        return cup_found, cup_center, cup_bbox, table_found, table_center

    def _detect_green_marker(self, rgb: np.ndarray):
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(hsv, self._hsv_green_lower, self._hsv_green_upper)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            best = max(contours, key=cv2.contourArea)
            if cv2.contourArea(best) > 10:
                M = cv2.moments(best)
                if M["m00"] > 0:
                    cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
                    return (cx, cy), best
        return None, None
    
    def _get_depth_at_pixel(self, depth_img: np.ndarray, u: int, v: int) -> float:
        half = 3
        v1, v2 = max(0, v-half), min(self.height, v+half+1)
        u1, u2 = max(0, u-half), min(self.width, u+half+1)
        window = depth_img[v1:v2, u1:u2]
        valid = window[window > 0]
        return np.median(valid) if len(valid) > 0 else -1
    
    def _pixel_to_world(self, u: int, v: int, depth: float) -> np.ndarray:
        f = 0.5 * self.height / np.tan(np.deg2rad(self.fovy) / 2)
        cx, cy = self.width / 2, self.height / 2
        x_cam = (u - cx) * depth / f
        y_cam = (v - cy) * depth / f
        z_cam = depth
        cam_pos = self.data.cam_xpos[self.cam_id].copy()
        cam_rot = self.data.cam_xmat[self.cam_id].reshape(3, 3)
        point_cam_mujoco = np.array([x_cam, -y_cam, -z_cam])
        return cam_pos + cam_rot @ point_cam_mujoco
    
    def _smooth_position(self, new_pos: np.ndarray) -> np.ndarray:
        self._position_history.append(new_pos.copy())
        if len(self._position_history) > 10:
            self._position_history.pop(0)
        if len(self._position_history) < 3:
            return new_pos
        return np.median(self._position_history, axis=0)
    
    def update(self) -> dict:
        # Apply domain randomization before rendering
        self._apply_domain_randomization()
        
        self.rgb_renderer.update_scene(self.data, camera=self.cam_id)
        rgb = self.rgb_renderer.render()
        self.depth_renderer.update_scene(self.data, camera=self.cam_id)
        depth = self.depth_renderer.render()
        
        cup_found, cup_center, cup_bbox, table_found, table_center = self._detect_objects_yolo(rgb)
        self.cup_detected = cup_found
        self.table_detected = table_found
        self.debug_image = rgb.copy()
        
        if cup_found and cup_center is not None:
            u, v = cup_center
            d = self._get_depth_at_pixel(depth, u, v)
            if 0.2 < d < 5.0:
                pos_world = self._pixel_to_world(u, v, d)
                pos_smoothed = self._smooth_position(pos_world)
                self.cup_position = pos_smoothed
                self.cup_top_position = pos_smoothed.copy()
                self.cup_top_position[2] += 0.04
                
                self.last_known_cup_position = self.cup_position.copy()
                self.last_known_cup_top = self.cup_top_position.copy()
                
                if cup_bbox:
                    x1, y1, x2, y2 = cup_bbox
                    cv2.rectangle(self.debug_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(self.debug_image, cup_center, 5, (255, 255, 0), -1)
        
        if table_found and table_center is not None:
            u, v = table_center
            d = self._get_depth_at_pixel(depth, u, v)
            if 0.2 < d < 5.0:
                pos_world = self._pixel_to_world(u, v, d)
                self.table_position = self._smooth_position(pos_world)
                self.last_known_table_position = self.table_position.copy()
        
        if not self.cup_detected and self.last_known_cup_position is not None:
            self.cup_position = self.last_known_cup_position
            self.cup_top_position = self.last_known_cup_top
        
        if not self.table_detected and self.last_known_table_position is not None:
            self.table_position = self.last_known_table_position
        
        # Status text
        status = []
        if self.cup_detected:
            status.append("CUP DETECTED (YOLO)")
        elif self.last_known_cup_position is not None:
            status.append("CUP: LAST KNOWN")
        else:
            status.append("CUP NOT DETECTED")
        
        if self.table_detected:
            status.append("TABLE DETECTED")
        elif self.last_known_table_position is not None:
            status.append("TABLE: LAST KNOWN")
        
        cv2.putText(self.debug_image, " | ".join(status), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Detect green marker
        marker_uv, marker_contour = self._detect_green_marker(rgb)
        if marker_uv:
            cv2.drawContours(self.debug_image, [marker_contour], -1, (0, 255, 255), 2)
            cv2.circle(self.debug_image, marker_uv, 5, (0, 255, 255), -1)
        
        return {
            'cup_detected': self.cup_detected or self.last_known_cup_position is not None,
            'table_detected': self.table_detected or self.last_known_table_position is not None,
            'cup_position': self.cup_position,
            'cup_top_position': self.cup_top_position,
            'table_position': self.table_position,
            'marker_uv': marker_uv,
            'debug_image': self.debug_image
        }


# ============================================================================
# BASE CONTROLLER
# ============================================================================

class BaseController:
    """Mobile base controller with velocity control."""
    def __init__(self, model, data):
        self.model = model
        self.data = data
        # Assuming freejoint for base: qpos[0:3] = x,y,z; qpos[3:6] = quat; but for mobile base, assume planar
        # Adjust if needed; assuming qvel[0] = vx, qvel[1] = vy, qvel[5] = wz (yaw rate)
        self.vel_indices = [0, 1, 5]  # vx, vy, wz

    def get_position(self):
        return self.data.qpos[0:2].copy()

    def get_yaw(self):
        # Simplify; assume qpos[5] or compute from quat
        quat = self.data.qpos[3:7]
        yaw = np.arctan2(2*(quat[0]*quat[3] + quat[1]*quat[2]), 1 - 2*(quat[2]**2 + quat[3]**2))
        return yaw

    def set_velocity(self, vel: np.ndarray):
        # vel = [vx, vy, wz]
        self.data.qvel[self.vel_indices[0]] = vel[0]
        self.data.qvel[self.vel_indices[1]] = vel[1]
        self.data.qvel[self.vel_indices[2]] = vel[2]

    def rotate_gentle(self, speed: float):
        self.set_velocity(np.array([0.0, 0.0, speed]))

    def stop(self):
        self.set_velocity(np.array([0.0, 0.0, 0.0]))


# ============================================================================
# ARM CONTROLLER WITH IK
# ============================================================================

class ArmController:
    """Arm controller with numerical IK."""
    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.joint_names = [f"arm_l_joint{i}" for i in range(1, 8)]
        self.joint_ids = [model.joint(name).id for name in self.joint_names]
        self.gripper_ids = [model.actuator(f"actuator_gripper_l_joint{i}").id for i in range(1, 5)]
        self.link7_body_id = model.body("arm_l_link7").id
        self.joint_limits = [model.jnt_range[jid] for jid in self.joint_ids]
        
        # For IK: Use Jacobian-based with damping
        self.ik_damping = 0.01

    def get_joint_positions(self):
        return np.array([self.data.qpos[jid] for jid in self.joint_ids])

    def set_joint_positions(self, q: np.ndarray):
        for i, jid in enumerate(self.joint_ids):
            self.data.qpos[jid] = q[i]

    def get_end_effector_pose(self):
        pos = self.data.body(self.link7_body_id).xpos + CONFIG.GRASP_CENTER_OFFSET
        mat = self.data.body(self.link7_body_id).xmat.reshape(3,3)
        return pos.copy(), mat.copy()

    def compute_ik(self, target_pos: np.ndarray, target_quat: np.ndarray = None, max_iter: int = 100, tol: float = 1e-4):
        q = self.get_joint_positions().copy()
        for _ in range(max_iter):
            pos, mat = self.get_end_effector_pose()
            pos_err = target_pos - pos
            if target_quat is not None:
                # Simple orientation error (axis-angle approximation)
                rot_err = np.cross(mat[:,2], target_quat[:,2])  # Assume z-axis alignment for gripper
                err = np.concatenate([pos_err, rot_err * 0.5])
            else:
                err = np.concatenate([pos_err, np.zeros(3)])
            
            if np.linalg.norm(err) < tol:
                return q, True
            
            # Full 6xN Jacobian
            jacp = np.zeros((3, self.model.nv))
            jacr = np.zeros((3, self.model.nv))
            mujoco.mj_jacBody(self.model, self.data, jacp, jacr, self.link7_body_id)
            jac = np.vstack((jacp, jacr))[:, [self.data.jnt_dofadr[jid] for jid in self.joint_ids]]
            
            dq = np.linalg.solve(jac.T @ jac + self.ik_damping * np.eye(len(q)), jac.T @ err)
            q += dq
            q = np.clip(q, [lim[0] for lim in self.joint_limits], [lim[1] for lim in self.joint_limits])
            self.set_joint_positions(q)
            mujoco.mj_forward(self.model, self.data)
        
        return q, False

    def move_to_pose(self, target_pos: np.ndarray, target_quat: np.ndarray = None, speed: float = 0.2):
        q_target, success = self.compute_ik(target_pos, target_quat)
        if success:
            current_q = self.get_joint_positions()
            delta = q_target - current_q
            step = np.clip(delta * speed, -0.1, 0.1)  # Limit step size
            new_q = current_q + step
            self.set_joint_positions(new_q)
            return np.linalg.norm(delta)
        return float('inf')

    def descend_with_ik(self, target: np.ndarray, dt: float, damping: float = 0.04, gain: float = 5.0):
        self.ik_damping = damping
        current_pos, _ = self.get_end_effector_pose()
        target_pos = target.copy()
        target_pos[2] = current_pos[2] - gain * dt  # Gradual descend
        self.move_to_pose(target_pos)
        z_err = current_pos[2] - target[2]
        return abs(z_err)

    def lift_simple(self, target: np.ndarray, dt: float, damping: float = 0.05, gain: float = 6.0):
        self.ik_damping = damping
        current_pos, _ = self.get_end_effector_pose()
        target_pos = target.copy()
        target_pos[2] = current_pos[2] + gain * dt
        self.move_to_pose(target_pos)

    def get_grasp_center_world(self):
        return self.get_end_effector_pose()[0]

    def set_gripper(self, closed: bool):
        val = 1.0 if closed else 0.0
        for gid in self.gripper_ids:
            self.data.ctrl[gid] = val

    def set_gripper_value(self, val: float):
        for gid in self.gripper_ids:
            self.data.ctrl[gid] = val

    def save_positioned_joints(self):
        pass  # Not needed with IK


# ============================================================================
# PICK AND PLACE CONTROLLER WITH LLM PLANNING
# ============================================================================

class PickAndPlaceController:
    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.base = BaseController(model, data)
        self.arm = ArmController(model, data)
        self.vision = VisionSystem(model, data)
        self.sim_hand = SimulatedHandController(model, data)
        self.hand_tracker = None
        self.current_task = "Bring the red cup from the table to the human"
        self.plan = []
        self.current_step = 0
        self.timer = 0.0
        self.gripper_release_value = 0.0
        self.handover_in_progress = False
        self.grasp_gesture_timer = 0.0
        self.handover_base_pose = None
        self.smooth_joint2_target = 0.0
        self.marker_x_center = 0.5

    def generate_plan(self):
        scene_desc = self.describe_scene()
        prompt = f"""
Robot Task: {self.current_task}

Current Scene: {scene_desc}

Constraints:
- Do not move backwards while holding cup (inertia causes slip)
- Gripper mouth faces forward; plan movements accordingly
- Use left arm
- Mobile base for navigation
- Detect cup and table
- For handover, approach human, track hand, release on grasp gesture

Provide a step-by-step plan as a numbered list of actions.
Each action should be executable: e.g., 'Move base to [x,y]', 'Set arm to pose [pos]', 'Close gripper', etc.
Include checks like 'If cup detected' 
"""
        plan_text = query_llm(prompt)
        self.plan = self.parse_plan(plan_text)
        print("Generated Plan:")
        for step in self.plan:
            print(step)
        self.current_step = 0

    def describe_scene(self) -> str:
        vr = self.vision.update()
        desc = f"Robot position: {self.base.get_position()}, yaw: {self.base.get_yaw()}\n"
        if vr['cup_position'] is not None:
            desc += f"Cup at {vr['cup_position']}\n"
        if vr['table_position'] is not None:
            desc += f"Table at {vr['table_position']}\n"
        desc += f"Arm joints: {self.arm.get_joint_positions()}\n"
        desc += f"Gripper: {'closed' if self.data.ctrl[self.arm.gripper_ids[0]] > 0.5 else 'open'}\n"
        if self.hand_tracker and self.hand_tracker.hand_detected:
            desc += f"Human hand detected, gesture: {self.hand_tracker.gesture}\n"
        return desc

    def parse_plan(self, plan_text: str) -> List[str]:
        lines = plan_text.strip().split('\n')
        steps = []
        for line in lines:
            if line.strip().startswith(tuple('123456789')):
                steps.append(line.strip()[line.find(' ')+1:].strip())
        return steps

    def execute_step(self, step: str, dt: float, vr: dict) -> bool:
        vr = self.vision.update()  # Fresh vision
        if "move base to" in step.lower():
            target = self.extract_target_pos(step)
            if target is not None:
                err = self.move_base_to(target, dt)
                return err < 0.05
        elif "set arm to" in step.lower() or "move arm" in step.lower():
            target = self.extract_target_pos(step)
            if target is not None:
                err = self.arm.move_to_pose(target)
                return err < 0.02
        elif "close gripper" in step.lower():
            self.arm.set_gripper(True)
            time.sleep(CONFIG.GRASPING_TIME)
            return True
        elif "open gripper" in step.lower():
            self.arm.set_gripper(False)
            return True
        elif "approach human" in step.lower():
            self.start_handover(vr)
            return True
        elif "track handover" in step.lower():
            self.handover_in_progress = True
            return False  # Ongoing
        # Add more parsers as needed
        else:
            print(f"Unknown action: {step}")
            return True  # Skip
        
        return False

    def extract_target_pos(self, step: str) -> Optional[np.ndarray]:
        # Simple parse, e.g., [x,y,z]
        import re
        match = re.search(r'\[([\d.-]+),\s*([\d.-]+),\s*([\d.-]+)\]', step)
        if match:
            return np.array([float(match.group(1)), float(match.group(2)), float(match.group(3))])
        return None

    def move_base_to(self, target: np.ndarray, dt: float) -> float:
        current = self.base.get_position()
        delta = target - current
        dist = np.linalg.norm(delta)
        if dist < 0.05:
            self.base.stop()
            return 0.0
        dir = delta / dist
        vel = dir * CONFIG.BASE_SPEED
        self.base.set_velocity(np.append(vel, 0.0))  # No rotation, assume aligned
        return dist

    def start_handover(self, vr: dict):
        self.hand_tracker = HandTracker()
        self.handover_base_pose = self.arm.get_joint_positions().copy()
        self.smooth_joint2_target = self.handover_base_pose[2]
        self.marker_x_center = self.sim_hand.get_marker_x_normalized()
        self.grasp_gesture_timer = 0.0

    def update_handover(self, dt: float, vr: dict):
        if not self.hand_tracker:
            return False
        
        tracker_result = self.hand_tracker.update()
        gesture = tracker_result['gesture']
        marker_x = self.sim_hand.get_marker_x_normalized()
        
        target_smoothing = 0.7
        move_smoothing = 0.6
        
        raw_target = self.smooth_joint2_target
        
        if gesture == "PALM_OPEN":
            self.grasp_gesture_timer = 0.0
            marker_delta = self.marker_x_center - marker_x
            joint2_delta = marker_delta * CONFIG.HAND_MIRROR_SCALE * 2.0
            raw_target = self.handover_base_pose[2] + joint2_delta
        elif gesture == "GRASP_C":
            self.grasp_gesture_timer += dt
            raw_target = self.smooth_joint2_target
            if self.grasp_gesture_timer > CONFIG.RELEASE_HOLD_TIME:
                self.arm.set_gripper(False)
                return True  # Released
        else:
            self.grasp_gesture_timer = max(0.0, self.grasp_gesture_timer - dt)
            raw_target = self.handover_base_pose[2]
        
        raw_target = np.clip(raw_target, self.arm.joint_limits[2][0], self.arm.joint_limits[2][1])
        self.smooth_joint2_target = target_smoothing * raw_target + (1 - target_smoothing) * self.smooth_joint2_target
        
        current_q = self.arm.get_joint_positions()
        new_joint2 = current_q[2] + (self.smooth_joint2_target - current_q[2]) * move_smoothing
        
        new_q = self.handover_base_pose.copy()
        new_q[2] = new_joint2
        self.arm.set_joint_positions(new_q)
        
        return False  # Continue handover

    def update(self, dt: float):
        vr = self.vision.update()
        self.sim_hand.update(0.5, 0.0)  # Default
        
        if not self.plan:
            self.generate_plan()
        
        if self.current_step < len(self.plan):
            done = self.execute_step(self.plan[self.current_step], dt, vr)
            if done:
                self.current_step += 1
        elif self.handover_in_progress:
            released = self.update_handover(dt, vr)
            if released:
                print("MISSION COMPLETE")
                if self.hand_tracker:
                    self.hand_tracker.close()
                    self.hand_tracker = None
        else:
            print("Plan complete")
        
        return vr['debug_image']

    def get_hand_frame(self):
        if self.hand_tracker is not None:
            return self.hand_tracker.frame
        return None


# ============================================================================
# MAIN
# ============================================================================

def make_gripper_sticky(model):
    print("  Patching gripper physics for better grip...")
    for i in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)
        if name and ("gripper" in name or "finger" in name):
            model.geom_friction[i, 0] = 5.0
            model.geom_friction[i, 1] = 5.0
            model.geom_friction[i, 2] = 0.5

def main():
    if not MEDIAPIPE_AVAILABLE:
        print("ERROR: MediaPipe is required.")
        return
    
    possible_paths = ["pick_and_place_scene.xml", "src/robotis_mujoco_menagerie/robotis_ffw/pick_and_place_scene.xml"]
    scene_path = next((p for p in possible_paths if os.path.exists(p)), None)
    if scene_path is None:
        print("ERROR: Cannot find pick_and_place_scene.xml")
        return
    
    full_path = os.path.abspath(scene_path)
    os.chdir(os.path.dirname(full_path))
    
    print(f"Loading: {scene_path}")
    model = mujoco.MjModel.from_xml_path(os.path.basename(scene_path))
    data = mujoco.MjData(model)
    make_gripper_sticky(model)
    
    controller = PickAndPlaceController(model, data)
    viewer = mujoco.viewer.launch_passive(model, data)
    
    dt = model.opt.timestep
    cv_available = True
    
    print("\nSimulation running...")
    
    while viewer.is_running():
        debug_img = controller.update(dt)
        mujoco.mj_step(model, data)
        
        if debug_img is not None and cv_available:
            try:
                cv2.imshow("Robot Camera", cv2.cvtColor(debug_img, cv2.COLOR_RGB2BGR))
            except cv2.error: cv_available = False
        
        hand_frame = controller.get_hand_frame()
        if hand_frame is not None:
            cv2.imshow("Human Hand (Webcam)", hand_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'): break
        viewer.sync()
    
    cv2.destroyAllWindows()
    print("\nSimulation ended.")

if __name__ == "__main__":
    main()