"""
VISION-BASED PICK AND PLACE - V13 (INTEGRATED HAND TRACKING)
=============================================================
Integrates the working test_hand_tracking.py logic:
1. Marker world X position normalized to 0-1 range
2. Mirrored movement: hand_offset = 0.5 - marker_x
3. Faster smoothing: target=0.4, move=0.3
4. HAND_MIRROR_SCALE = 1.5 for larger arm movement
5. RELEASE_HOLD_TIME = 0.2 for quick release
"""

import mujoco
import mujoco.viewer
import numpy as np
import cv2
import os
import math
import time
from enum import Enum, auto
from dataclasses import dataclass
from typing import Optional, Tuple

# MediaPipe for hand tracking
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("WARNING: MediaPipe not installed. Run: pip install mediapipe")


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
    
    # Handover settings - FROM WORKING TEST CODE
    HAND_MIRROR_SCALE: float = 1.5   # Increased for larger arm movement
    RELEASE_HOLD_TIME: float = 0.2   # Quick release
    
    # Gesture Thresholds (Degrees)
    ANGLE_OPEN_THRESHOLD: float = 150.0  
    ANGLE_THUMB_OPEN_THRESHOLD: float = 140.0
    
    def __post_init__(self):
        if self.GRASP_CENTER_OFFSET is None:
            self.GRASP_CENTER_OFFSET = np.array([0.0, 0.0, -0.10])


CONFIG = Config()


# ============================================================================
# SIMULATED HAND CONTROLLER - FROM WORKING TEST CODE
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
        
        # Map webcam X (0-1) to swing angle - FROM WORKING TEST CODE
        target_swing = np.interp(norm_x, [0, 1], [1.0, -1.0])
        
        self.smooth_swing += (target_swing - self.smooth_swing) * 0.5  # Faster response
        self.data.ctrl[self.act_swing] = self.smooth_swing
        
        # Map Curl
        self.smooth_curl += (target_curl - self.smooth_curl) * 0.2
        
        for f in self.fingers:
            self.data.ctrl[f[0]] = self.smooth_curl * 1.5
            self.data.ctrl[f[1]] = self.smooth_curl * 1.5
            self.data.ctrl[f[2]] = self.smooth_curl * 1.0
            
    def get_curl_value(self):
        return self.smooth_curl
    
    def get_marker_position(self):
        """Get world position of palm marker for stopping logic."""
        if not self.valid:
            return None
        return self.data.geom_xpos[self.marker_geom_id].copy()
    
    def get_marker_x_normalized(self):
        """
        Get marker X position normalized to 0.0 - 1.0 range.
        This is the equivalent of MediaPipe's hand_x.
        FROM WORKING TEST CODE.
        """
        pos = self.get_marker_position()
        if pos is None:
            return 0.5
        
        marker_x = pos[0]
        
        # Map to 0-1 based on marker swing range
        # Marker at X=0.3 → 0.0, Marker at X=1.3 → 1.0
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
            
            # Wrist Position
            wrist = lm.landmark[self.mp_hands.HandLandmark.WRIST]
            self.hand_x = wrist.x
            self.smoothed_x = 0.4 * self.hand_x + 0.6 * self.smoothed_x
            
            # Finger angles for gesture
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
            
            # Determine gesture
            if avg_ang > 140:
                self.gesture = "PALM_OPEN"
            elif avg_ang < 90:
                self.gesture = "GRASP_C"
            else:
                self.gesture = "TRANSITION"
            
            # Map to curl value
            raw_curl = np.interp(avg_ang, [60, 160], [1.0, 0.0])
            
            # Hysteresis
            if self.last_valid_curl < 0.5:
                self.last_valid_curl = 1.0 if raw_curl > 0.7 else 0.0
            else:
                self.last_valid_curl = 0.0 if raw_curl < 0.3 else 1.0
                
            self.hand_detected = True
            
            # Draw HUD
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
        self.cup_position = None
        self.cup_top_position = None
        self._position_history = []
        self.debug_image = None
    
    def _detect_red_cup(self, rgb: np.ndarray):
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        mask1 = cv2.inRange(hsv, np.array([0, 100, 50]), np.array([8, 255, 255]))
        mask2 = cv2.inRange(hsv, np.array([172, 100, 50]), np.array([180, 255, 255]))
        mask = mask1 | mask2
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return False, None, None
        best = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(best)
        if area < 50 or area > 10000:
            return False, None, None
        M = cv2.moments(best)
        if M["m00"] == 0:
            return False, None, None
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        return True, (cx, cy), best

    def _detect_green_marker(self, rgb: np.ndarray):
        """Detects the green marker on simulated hand."""
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(hsv, np.array([40, 100, 100]), np.array([80, 255, 255]))
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
        self.rgb_renderer.update_scene(self.data, camera=self.cam_id)
        rgb = self.rgb_renderer.render()
        self.depth_renderer.update_scene(self.data, camera=self.cam_id)
        depth = self.depth_renderer.render()
        
        found, center, contour = self._detect_red_cup(rgb)
        self.cup_detected = False
        self.debug_image = rgb.copy()
        
        if found and center is not None:
            u, v = center
            d = self._get_depth_at_pixel(depth, u, v)
            if 0.2 < d < 5.0:
                pos_world = self._pixel_to_world(u, v, d)
                pos_smoothed = self._smooth_position(pos_world)
                self.cup_position = pos_smoothed
                self.cup_top_position = pos_smoothed.copy()
                self.cup_top_position[2] += 0.04
                self.cup_detected = True
                cv2.drawContours(self.debug_image, [contour], -1, (0, 255, 0), 2)
                cv2.circle(self.debug_image, center, 5, (255, 255, 0), -1)
        
        if self.cup_detected:
            cv2.putText(self.debug_image, "CUP DETECTED", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        else:
            cv2.putText(self.debug_image, "SEARCHING...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 100, 255), 2)
        
        # Marker Detection
        marker_center, marker_cnt = self._detect_green_marker(rgb)
        if marker_center:
            cv2.drawContours(self.debug_image, [marker_cnt], -1, (255, 0, 0), 2)
            cv2.circle(self.debug_image, marker_center, 5, (255, 0, 0), -1)

        return {
            'detected': self.cup_detected, 
            'cup_top': self.cup_top_position,
            'marker_uv': marker_center
        }


# ============================================================================
# ARM CONTROLLER
# ============================================================================

class ArmController:
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        self.model = model
        self.data = data
        self.link7_id = model.body("arm_r_link7").id
        self.joint_names = [f"arm_r_joint{i}" for i in range(1, 8)]
        self.joint_ids = [model.joint(name).id for name in self.joint_names]
        self.qpos_addrs = [model.jnt_qposadr[jid] for jid in self.joint_ids]
        self.qvel_addrs = [model.jnt_dofadr[jid] for jid in self.joint_ids]
        self.actuator_ids = [model.actuator(f"actuator_{name}").id for name in self.joint_names]
        self.joint_limits = [(model.jnt_range[jid, 0], model.jnt_range[jid, 1]) for jid in self.joint_ids]
        self.gripper_actuator_ids = [model.actuator(f"actuator_gripper_r_joint{i}").id for i in range(1, 5)]
        
        self.POSE_READY = np.array([0.0, -1.0, 0.0, -1.57, 0.0, 0.0, 0.0])
        
        self._last_q_cmd = None
        self._positioned_joints = None
    
    def get_joint_positions(self) -> np.ndarray:
        return np.array([self.data.qpos[addr] for addr in self.qpos_addrs])
    
    def get_link7_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        pos = self.data.xpos[self.link7_id].copy()
        rot = self.data.xmat[self.link7_id].reshape(3, 3).copy()
        return pos, rot
    
    def get_grasp_center_world(self) -> np.ndarray:
        link7_pos, link7_rot = self.get_link7_pose()
        return link7_pos + link7_rot @ CONFIG.GRASP_CENTER_OFFSET
    
    def set_joint_positions(self, q: np.ndarray):
        for i, aid in enumerate(self.actuator_ids):
            q_clamped = np.clip(q[i], self.joint_limits[i][0], self.joint_limits[i][1])
            self.data.ctrl[aid] = q_clamped
    
    def set_gripper(self, closed: bool):
        value = 0.70 if closed else 0.0
        for aid in self.gripper_actuator_ids:
            self.data.ctrl[aid] = value

    def set_gripper_value(self, value: float):
        for aid in self.gripper_actuator_ids:
            self.data.ctrl[aid] = value
    
    def move_to_pose(self, target_pose: np.ndarray, speed: float = 0.1) -> float:
        current = self.get_joint_positions()
        error = target_pose - current
        step = np.clip(error * speed, -0.15, 0.15)
        new_q = current + step
        self.set_joint_positions(new_q)
        return np.linalg.norm(error)
    
    def ik_to_position_horizontal(self, target_pos: np.ndarray, dt: float, 
                                   damping: float = 0.05, gain: float = 5.0) -> float:
        current_grasp = self.get_grasp_center_world()
        _, link7_rot = self.get_link7_pose()
        
        pos_error = target_pos - current_grasp
        pos_dist = np.linalg.norm(pos_error)
        
        current_x = link7_rot @ np.array([1, 0, 0])
        target_x = np.array([0, 0, 1])
        rot_error = np.cross(current_x, target_x) * 1.5
        
        error_6d = np.concatenate([pos_error, rot_error])
        
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacBody(self.model, self.data, jacp, jacr, self.link7_id)
        J_pos = jacp[:, self.qvel_addrs]
        J_rot = jacr[:, self.qvel_addrs]
        J = np.vstack([J_pos, J_rot])
        
        lam = damping
        JJT = J @ J.T + lam**2 * np.eye(6)
        dq = J.T @ np.linalg.solve(JJT, error_6d) * gain
        
        max_vel = np.array([3.0, 3.0, 3.0, 3.0, 2.0, 2.0, 2.0])
        dq = np.clip(dq, -max_vel, max_vel)
        
        current_q = self.get_joint_positions()
        new_q = current_q + dq * dt * 8.0
        
        if self._last_q_cmd is not None:
            new_q = 0.6 * new_q + 0.4 * self._last_q_cmd
        self._last_q_cmd = new_q.copy()
        
        self.set_joint_positions(new_q)
        return pos_dist
    
    def save_positioned_joints(self):
        self._positioned_joints = self.get_joint_positions().copy()
        print(f"    Saved joint positions")
    
    def descend_with_ik(self, target_pos: np.ndarray, dt: float, 
                        damping: float = 0.04, gain: float = 6.0) -> float:
        current_grasp = self.get_grasp_center_world()
        _, link7_rot = self.get_link7_pose()
        
        pos_error = target_pos - current_grasp
        z_error = abs(target_pos[2] - current_grasp[2])
        
        current_x = link7_rot @ np.array([1, 0, 0])
        target_x = np.array([0, 0, 1])
        rot_error = np.cross(current_x, target_x) * 1.5
        
        error_6d = np.concatenate([pos_error, rot_error])
        
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacBody(self.model, self.data, jacp, jacr, self.link7_id)
        J_pos = jacp[:, self.qvel_addrs]
        J_rot = jacr[:, self.qvel_addrs]
        J = np.vstack([J_pos, J_rot])
        
        lam = damping
        JJT = J @ J.T + lam**2 * np.eye(6)
        dq = J.T @ np.linalg.solve(JJT, error_6d) * gain
        
        max_vel = np.array([2.0, 2.0, 2.0, 2.0, 1.5, 1.5, 1.5])
        dq = np.clip(dq, -max_vel, max_vel)
        
        current_q = self.get_joint_positions()
        new_q = current_q + dq * dt * 10.0
        
        if self._last_q_cmd is not None:
            new_q = 0.5 * new_q + 0.5 * self._last_q_cmd
        self._last_q_cmd = new_q.copy()
        
        self.set_joint_positions(new_q)
        return z_error
    
    def lift_simple(self, target_pos: np.ndarray, dt: float, 
                     damping: float = 0.05, gain: float = 6.0) -> float:
        current_grasp = self.get_grasp_center_world()
        error = target_pos - current_grasp
        dist = np.linalg.norm(error)
        
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacBody(self.model, self.data, jacp, jacr, self.link7_id)
        J = jacp[:, self.qvel_addrs]
        
        lam = damping
        JJT = J @ J.T + lam**2 * np.eye(3)
        dq = J.T @ np.linalg.solve(JJT, error) * gain
        
        max_vel = np.array([2.0, 2.0, 2.0, 2.0, 0.5, 0.3, 0.2])
        for i in range(7):
            dq[i] = np.clip(dq[i], -max_vel[i], max_vel[i])
        
        current_q = self.get_joint_positions()
        new_q = current_q + dq * dt * 8.0
        
        if self._last_q_cmd is not None:
            new_q = 0.7 * new_q + 0.3 * self._last_q_cmd
        self._last_q_cmd = new_q.copy()
        
        self.set_joint_positions(new_q)
        return dist


# ============================================================================
# BASE CONTROLLER
# ============================================================================

class BaseController:
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        self.model = model
        self.data = data
        self.base_id = model.body("robot_base").id
    
    def get_position(self) -> np.ndarray:
        return self.data.xpos[self.base_id].copy()

    def get_yaw(self) -> float:
        quat = self.data.xquat[self.base_id]
        return np.arctan2(2*(quat[0]*quat[3] + quat[1]*quat[2]), 1 - 2*(quat[2]**2 + quat[3]**2))
    
    def set_velocity(self, vel: np.ndarray):
        self.data.qvel[0:3] = vel
        self.data.qvel[3:6] *= 0.5
    
    def move_toward(self, target_xy: np.ndarray, speed: float = 0.2) -> float:
        current = self.get_position()
        error_xy = target_xy[:2] - current[:2]
        dist = np.linalg.norm(error_xy)
        if dist < 0.02:
            self.set_velocity(np.zeros(3))
            return dist
        direction = error_xy / dist
        self.set_velocity(np.array([direction[0] * speed, direction[1] * speed, 0.0]))
        return dist

    def rotate(self, speed_rot: float):
        self.data.qvel[0] = 0
        self.data.qvel[1] = 0
        self.data.qvel[5] = speed_rot

    def rotate_gentle(self, target_rot_speed: float):
        """Gentle rotation with smooth acceleration."""
        self.data.qvel[0] *= 0.9
        self.data.qvel[1] *= 0.9
        
        current_rot = self.data.qvel[5]
        max_accel = 0.12  # Faster acceleration
        diff = target_rot_speed - current_rot
        diff = np.clip(diff, -max_accel, max_accel)
        self.data.qvel[5] = current_rot + diff
        
        self.data.qvel[3] *= 0.85
        self.data.qvel[4] *= 0.85

    def stop(self):
        self.set_velocity(np.zeros(3))
        self.data.qvel[5] = 0


# ============================================================================
# STATE MACHINE
# ============================================================================

class State(Enum):
    SEARCHING = auto()
    APPROACHING = auto()
    POSITIONING = auto()
    DESCENDING = auto()
    GRASPING = auto()
    LIFTING = auto()
    TILTING_ARM = auto()
    STRAFING = auto()
    ROTATING = auto()
    APPROACHING_HAND = auto()
    TRACKING_HANDOVER = auto()
    RELEASING = auto()
    COMPLETE = auto()


class PickAndPlaceController:
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        self.model = model
        self.data = data
        self.vision = VisionSystem(model, data)
        self.arm = ArmController(model, data)
        self.base = BaseController(model, data)
        
        self.sim_hand = SimulatedHandController(model, data)
        self.hand_tracker = None  # Deferred initialization
        
        self.state = State.SEARCHING
        self.timer = 0.0
        
        self.locked_cup_top = None
        self.locked_hover_pos = None
        self.locked_grasp_pos = None
        self.target_base_pos = None
        
        self.gripper_release_value = 0.70
        self.lifted_arm_pose = None
        self.travel_arm_pose = None
        
        # For handover control (from reference V9)
        self.handover_base_pose = None
        self.smooth_joint2_target = 0.0
        self.grasp_gesture_timer = 0.0
        self.locked_release_joints = None
        self.marker_x_center = 0.5  # Initial marker position when tracking starts
        
        self.arm.set_gripper(False)
        self._print_header()
    
    def _print_header(self):
        print("\n" + "="*70)
        print("PICK & PLACE V12 - FINAL")
        print("="*70)
        print("Phase 1: Auto Pick")
        print("Phase 2: Tilt -> Strafe -> Rotate -> Approach Hand")
        print("Phase 3: Webcam control of Sim Hand")
        print("Phase 4: Robot mirrors marker (MIRRORED: marker right -> arm left)")
        print("Phase 5: Grasp gesture -> Release cup")
        print("="*70 + "\n")

    def update(self, dt: float):
        self.timer += dt
        
        # Update Sim Hand from Real Hand (only when tracker exists)
        if self.hand_tracker is not None:
            real_result = self.hand_tracker.update()
            if real_result['detected']:
                self.sim_hand.update(real_result['x'], real_result['curl'])
        
        vision_result = self.vision.update()
        
        if self.state == State.SEARCHING:
            self._state_searching(dt, vision_result)
        elif self.state == State.APPROACHING:
            self._state_approaching(dt)
        elif self.state == State.POSITIONING:
            self._state_positioning(dt)
        elif self.state == State.DESCENDING:
            self._state_descending(dt)
        elif self.state == State.GRASPING:
            self._state_grasping(dt)
        elif self.state == State.LIFTING:
            self._state_lifting(dt)
        elif self.state == State.TILTING_ARM:
            self._state_tilting_arm(dt)
        elif self.state == State.STRAFING:
            self._state_strafing(dt)
        elif self.state == State.ROTATING:
            self._state_rotating(dt)
        elif self.state == State.APPROACHING_HAND:
            self._state_approaching_hand(dt, vision_result)
        elif self.state == State.TRACKING_HANDOVER:
            self._state_tracking_handover(dt, vision_result)
        elif self.state == State.RELEASING:
            self._state_releasing(dt)
        elif self.state == State.COMPLETE:
            self._state_complete(dt)
        
        return self.vision.debug_image
    
    def _state_searching(self, dt: float, vision_result: dict):
        self.arm.move_to_pose(self.arm.POSE_READY, speed=0.2)
        self.base.stop()
        
        if vision_result['detected'] and vision_result['cup_top'] is not None:
            self.locked_cup_top = vision_result['cup_top'].copy()
            
            self.locked_hover_pos = self.locked_cup_top.copy()
            self.locked_hover_pos[2] += CONFIG.HOVER_HEIGHT
            self.locked_hover_pos[0] += CONFIG.POSITION_X_OFFSET
            self.locked_hover_pos[1] += CONFIG.POSITION_Y_OFFSET
            
            self.locked_grasp_pos = self.locked_hover_pos.copy()
            self.locked_grasp_pos[2] = self.locked_cup_top[2] - CONFIG.GRASP_DEPTH
            
            base_pos = self.base.get_position()
            direction = self.locked_cup_top[:2] - base_pos[:2]
            direction /= (np.linalg.norm(direction) + 1e-6)
            self.target_base_pos = self.locked_cup_top.copy()
            self.target_base_pos[:2] -= direction * CONFIG.STANDOFF_DISTANCE
            
            print(f"✓ CUP LOCKED")
            self._transition_to(State.APPROACHING)
    
    def _state_approaching(self, dt: float):
        base_dist = self.base.move_toward(self.target_base_pos[:2], CONFIG.BASE_SPEED)
        self.arm.move_to_pose(self.arm.POSE_READY, speed=0.2)
        
        if base_dist < 0.05:
            self.base.stop()
            print(f"✓ At standoff position")
            self._transition_to(State.POSITIONING)
    
    def _state_positioning(self, dt: float):
        self.base.stop()
        self.arm.set_gripper(False)
        dist = self.arm.ik_to_position_horizontal(self.locked_hover_pos, dt, damping=0.05, gain=6.0)
        grasp_pos = self.arm.get_grasp_center_world()
        xy_err = np.linalg.norm(grasp_pos[:2] - self.locked_hover_pos[:2])
        z_err = abs(grasp_pos[2] - self.locked_hover_pos[2])
        
        if xy_err < 0.03 and z_err < 0.04:
            print(f"✓ Aligned")
            self.arm.save_positioned_joints()
            self._transition_to(State.DESCENDING)
        if self.timer > CONFIG.POSITIONING_TIMEOUT:
            self.arm.save_positioned_joints()
            self._transition_to(State.DESCENDING)
    
    def _state_descending(self, dt: float):
        self.base.stop()
        self.arm.set_gripper(False)
        z_err = self.arm.descend_with_ik(self.locked_grasp_pos, dt, damping=0.04, gain=5.0)
        if z_err < 0.02 or self.timer > CONFIG.DESCENDING_TIMEOUT:
            print(f"✓ At grasp height")
            self._transition_to(State.GRASPING)
    
    def _state_grasping(self, dt: float):
        self.base.stop()
        self.arm.set_gripper(True)
        current_q = self.arm.get_joint_positions()
        self.arm.set_joint_positions(current_q)
        if self.timer > CONFIG.GRASPING_TIME:
            print(f"✓ Gripper closed")
            self._transition_to(State.LIFTING)
    
    def _state_lifting(self, dt: float):
        self.base.stop()
        self.arm.set_gripper(True)
        lift_target = self.locked_grasp_pos.copy()
        lift_target[2] = self.locked_cup_top[2] + 0.25
        self.arm.lift_simple(lift_target, dt, damping=0.05, gain=6.0)
        
        grasp_pos = self.arm.get_grasp_center_world()
        if grasp_pos[2] > self.locked_cup_top[2] + 0.18 or self.timer > 3.0:
            print(f"✓ Lifted - TILTING ARM")
            self.lifted_arm_pose = self.arm.get_joint_positions().copy()
            self._transition_to(State.TILTING_ARM)

    def _state_tilting_arm(self, dt: float):
        """Tilt gripper backward to prevent cup slippage."""
        self.base.stop()
        self.arm.set_gripper(True)
        
        target_pose = self.lifted_arm_pose.copy()
        target_pose[5] = -0.4  # Tilt wrist back
        
        error = self.arm.move_to_pose(target_pose, speed=0.2)
        
        if error < 0.1 or self.timer > 1.5:
            self.travel_arm_pose = self.arm.get_joint_positions().copy()
            print("✓ Arm tilted - STRAFING")
            self._transition_to(State.STRAFING)

    def _state_strafing(self, dt: float):
        """Move sideways to clear table."""
        self.arm.set_joint_positions(self.travel_arm_pose)
        self.arm.set_gripper(True)
        
        current_pos = self.base.get_position()
        
        if not hasattr(self, 'strafe_target'):
            self.strafe_target = current_pos[1] + 0.6
        
        if current_pos[1] < self.strafe_target:
            # FASTER strafe
            target_vel = 1.0
            current_vel = self.data.qvel[1]
            new_vel = current_vel + (target_vel - current_vel) * 0.2
            self.base.set_velocity(np.array([0.0, new_vel, 0.0]))
            
            self.data.qvel[3] *= 0.85
            self.data.qvel[4] *= 0.85
        else:
            self.data.qvel[1] *= 0.8
            
            if abs(self.data.qvel[1]) < 0.03:
                self.base.stop()
                print("✓ Cleared table - ROTATING")
                del self.strafe_target
                self._transition_to(State.ROTATING)

    def _state_rotating(self, dt: float):
        """Turn 90 degrees left."""
        self.arm.set_joint_positions(self.travel_arm_pose)
        self.arm.set_gripper(True)
        
        current_yaw = self.base.get_yaw()
        target_yaw = 1.57
        error = target_yaw - current_yaw
        
        while error > np.pi:
            error -= 2 * np.pi
        while error < -np.pi:
            error += 2 * np.pi
        
        if abs(error) > 0.05:
            # FASTER rotation
            rot_speed = np.clip(error * 1.5, -1.5, 1.5)
            self.base.rotate_gentle(rot_speed)
        else:
            self.data.qvel[5] *= 0.85
            
            if abs(self.data.qvel[5]) < 0.02:
                self.base.stop()
                print("✓ Rotated - APPROACHING HAND")
                self._transition_to(State.APPROACHING_HAND)

    def _state_approaching_hand(self, dt: float, vision_result: dict):
        """
        Move toward hand but STOP BEFORE reaching it.
        Use marker position to know when to stop.
        """
        self.arm.set_joint_positions(self.travel_arm_pose)
        self.arm.set_gripper(True)
        
        current_pos = self.base.get_position()
        grasp_pos = self.arm.get_grasp_center_world()
        
        # Get marker position in world
        marker_world_pos = self.sim_hand.get_marker_position()
        
        # Check if we should stop: cup should not go past the hand
        # Stop when gripper Y is close to marker Y (with some margin)
        should_stop = False
        
        if marker_world_pos is not None:
            # Gripper Y position vs Marker Y position
            # We want to stop BEFORE the cup passes the marker
            # Since we're moving in +Y direction after rotating 90 deg
            distance_to_marker = marker_world_pos[1] - grasp_pos[1]
            
            # Stop when gripper is about 0.15m before the marker
            if distance_to_marker < 0.15:
                should_stop = True
                print(f"  Stopping: gripper close to marker (dist={distance_to_marker:.2f}m)")
        
        # Also check marker in camera view
        marker_uv = vision_result.get('marker_uv')
        if marker_uv:
            cx, cy = marker_uv
            # If marker is in lower half of image and relatively close, we're close enough
            if cy > 300:  # Marker is low in view = we're close
                should_stop = True
                print(f"  Stopping: marker low in camera view (cy={cy})")
        
        if not should_stop:
            # FASTER approach
            target_vel = 1.0
            current_vel = self.data.qvel[1]
            new_vel = current_vel + (target_vel - current_vel) * 0.15
            self.base.set_velocity(np.array([0.0, new_vel, 0.0]))
            
            self.data.qvel[3] *= 0.9
            self.data.qvel[4] *= 0.9
        else:
            # Decelerate smoothly
            self.data.qvel[1] *= 0.8
            
            if abs(self.data.qvel[1]) < 0.03:
                self.base.stop()
                print("✓ Arrived at Human. STARTING HAND TRACKING...")
                
                # Initialize hand tracker NOW
                if MEDIAPIPE_AVAILABLE and self.hand_tracker is None:
                    try:
                        self.hand_tracker = HandTracker()
                        print("✓ Webcam hand tracking started")
                    except Exception as e:
                        print(f"WARNING: Could not start hand tracking: {e}")
                
                # Save pose for handover - IMPORTANT: save current arm position AND marker position
                self.handover_base_pose = self.arm.get_joint_positions().copy()
                self.smooth_joint2_target = self.handover_base_pose[2]
                self.grasp_gesture_timer = 0.0
                
                # Record initial marker X as the "center" reference
                # This prevents snapping - tracking starts from wherever marker currently is
                self.marker_x_center = self.sim_hand.get_marker_x_normalized()
                print(f"  Initial marker X center: {self.marker_x_center:.2f}")
                
                self._transition_to(State.TRACKING_HANDOVER)

    def _state_tracking_handover(self, dt: float, vision_result: dict):
        """
        Track marker and mirror robot arm movement.
        FIXED: Uses marker delta from INITIAL position, not from 0.5
        This prevents arm from snapping to a "home" position when tracking starts.
        """
        self.base.stop()
        self.arm.set_gripper(True)
        
        # Get current marker X normalized (0.0 to 1.0)
        marker_x = self.sim_hand.get_marker_x_normalized()
        
        # Control parameters - FASTER for responsive movement
        target_smoothing = 0.7   # Was 0.4 - even faster target update
        move_smoothing = 0.6     # Was 0.3 - even faster movement
        
        # Get gesture from webcam hand tracker
        gesture = "NONE"
        if self.hand_tracker is not None:
            gesture = self.hand_tracker.gesture
        
        # Default: keep current target
        raw_target = self.smooth_joint2_target
        
        if gesture == "PALM_OPEN":
            # 1. Reset Release Timer
            self.grasp_gesture_timer = 0.0
            
            # 2. Tracking Logic - FIXED: use delta from INITIAL marker position
            # marker moved right from start → arm goes left (mirrored)
            # marker moved left from start → arm goes right (mirrored)
            marker_delta = self.marker_x_center - marker_x  # Delta from where we started
            joint2_delta = marker_delta * CONFIG.HAND_MIRROR_SCALE * 2.0
            
            # Calculate Target - delta from where arm WAS when tracking started
            raw_target = self.handover_base_pose[2] + joint2_delta
            
            # Overlay info on webcam frame
            if self.hand_tracker and self.hand_tracker.frame is not None:
                cv2.putText(self.hand_tracker.frame, "MODE: TRACKING", (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(self.hand_tracker.frame, f"Marker X: {marker_x:.2f} (center: {self.marker_x_center:.2f})", (10, 120), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        elif gesture == "GRASP_C":
            # 1. Increment Timer
            self.grasp_gesture_timer += dt
            
            # 2. FREEZE TARGET: Keep smoothed target where it is
            raw_target = self.smooth_joint2_target
            
            # 3. Check for Release
            print(f"  RELEASE TIMER: {self.grasp_gesture_timer:.1f} / {CONFIG.RELEASE_HOLD_TIME:.1f}")
            
            if self.grasp_gesture_timer > CONFIG.RELEASE_HOLD_TIME:
                print("✓ TIMER COMPLETE - RELEASING!")
                self.locked_release_joints = self.arm.get_joint_positions().copy()
                self._transition_to(State.RELEASING)
                return
                
            # Overlay info
            if self.hand_tracker and self.hand_tracker.frame is not None:
                bar_width = int((self.grasp_gesture_timer / CONFIG.RELEASE_HOLD_TIME) * 200)
                cv2.rectangle(self.hand_tracker.frame, (10, 100), (210, 120), (50, 50, 50), -1)
                cv2.rectangle(self.hand_tracker.frame, (10, 100), (10 + bar_width, 120), (0, 165, 255), -1)
                cv2.putText(self.hand_tracker.frame, "HOLD TO RELEASE", (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
        
        else:
            # NONE or TRANSITION - decay timer and return to center
            self.grasp_gesture_timer = max(0.0, self.grasp_gesture_timer - dt)
            raw_target = self.handover_base_pose[2]
            
            if self.hand_tracker and self.hand_tracker.frame is not None:
                cv2.putText(self.hand_tracker.frame, "WAITING...", (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 2)

        # --- APPLY SMOOTHING (FROM WORKING TEST CODE) ---
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

    def _state_releasing(self, dt: float):
        """Slowly open gripper while arm is HARD LOCKED (from reference V9)."""
        self.base.stop()
        
        # HARD LOCK joints
        if self.locked_release_joints is not None:
            self.arm.set_joint_positions(self.locked_release_joints)
        
        # Ramp gripper open
        self.gripper_release_value = max(0.0, self.gripper_release_value - (dt * 1.5))
        self.arm.set_gripper_value(self.gripper_release_value)
        
        print(f"  RELEASING... {self.gripper_release_value:.2f}")
        
        if self.gripper_release_value <= 0.01:
            print("✓ Released")
            self._transition_to(State.COMPLETE)

    def _state_complete(self, dt: float):
        self.base.stop()
        if self.hand_tracker is not None:
            self.hand_tracker.close()
            self.hand_tracker = None
        
        if self.timer < 0.1:
            print("\n" + "="*70)
            print("MISSION COMPLETE")
            print("="*70)
    
    def _transition_to(self, new_state: State):
        print(f"→ {new_state.name}")
        self.state = new_state
        self.timer = 0.0
        if new_state == State.RELEASING:
            self.gripper_release_value = 0.70
    
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
                cv2.putText(debug_img, f"STATE: {controller.state.name}", (10, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                cv2.imshow("Robot Camera", cv2.cvtColor(debug_img, cv2.COLOR_RGB2BGR))
            except cv2.error: cv_available = False
        
        hand_frame = controller.get_hand_frame()
        if hand_frame is not None:
            cv2.imshow("Human Hand (Webcam)", hand_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'): break
        viewer.sync()
    
    if controller.hand_tracker: controller.hand_tracker.close()
    cv2.destroyAllWindows()
    print("\nSimulation ended.")

if __name__ == "__main__":
    main()