# ECE598JK — Vision-Based Robot Manipulation with LLM-Guided Handover

**UIUC ECE598JK Final Project**

A MuJoCo simulation of the Unitree G1 humanoid robot performing vision-based pick-and-place with autonomous LLM task planning and real-time human-robot handover via webcam gesture recognition.

## Overview

The system integrates three pipelines:

1. **Vision** — YOLOv8 detects a cup and table from the robot's onboard camera; depth rendering back-projects pixel coordinates to world-frame positions with temporal smoothing and domain randomization for sim-to-real robustness.

2. **LLM Planning** — Claude (Anthropic API) receives a natural-language task description and a live scene description, then generates a step-by-step action plan that the robot executes via a simple parser.

3. **Human Handover** — MediaPipe tracks the operator's hand in real time from a laptop webcam. The robot arm mirrors the hand's lateral position and releases the object when a sustained grasp gesture is detected.

## Structure

```
worker_ws/
├── pick_and_place.py        # Main simulation — vision + LLM + handover (V17)
├── human_hand_teleop.py     # Standalone hand teleoperation demo
├── test_hand_tracking.py    # Marker-tracking handover test
├── human_hand.xml           # MuJoCo model for simulated human hand
├── unitree_g1/              # Unitree G1 MJCF description + STL assets
└── extra/                   # Exploratory scripts (YOLO, navigation, IK, etc.)

ECE598JK_Project/
├── ECE598_Final_Presentation.pdf
└── ECE598JK_FInal_Report.pdf
```

## Dependencies

```
pip install mujoco mediapipe ultralytics opencv-python numpy requests
```

- MuJoCo ≥ 2.3.4
- Python ≥ 3.9

## Usage

Set your Anthropic API key:

```bash
export ANTHROPIC_API_KEY=your_key_here
```

Run the main simulation:

```bash
cd worker_ws
python pick_and_place.py
```

Run the hand teleoperation demo:

```bash
cd worker_ws
python human_hand_teleop.py
```

## Controls

| Action | Effect |
|---|---|
| Move hand left/right | Robot arm tracks laterally |
| Open palm | Arm follows hand position |
| Hold fist gesture | Robot releases object (handover) |
| Press `Q` | Quit |
