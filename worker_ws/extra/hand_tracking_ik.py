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

# Get site and body IDs
palm_site_id = mj_model.site("palm_center").id

# Robot right arm joints
robot_arm_joints = [
    "arm_r_joint1", "arm_r_joint2", "arm_r_joint3", 
    "arm_r_joint4", "arm_r_joint5", "arm_r_joint6", "arm_r_joint7"
]
robot_arm_qpos_addrs = []
for jname in robot_arm_joints:
    jnt_id = mj_model.joint(jname).id
    qpos_addr = mj_model.jnt_qposadr[jnt_id]
    robot_arm_qpos_addrs.append(qpos_addr)

# End effector body
ee_body_id = mj_model.body("arm_r_link7").id

# Human arm joints for interactive control
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

print(f"Robot arm DOFs: {len(robot_arm_qpos_addrs)}")
print(f"Human arm DOFs: {len(human_qpos_addrs)}")

def get_palm_position():
    """Get palm position in world frame"""
    return data.site_xpos[palm_site_id].copy()

def get_ee_position():
    """Get robot end effector position"""
    return data.xpos[ee_body_id].copy()

def compute_jacobian_ik(target_pos, current_q, max_iter=50):
    """
    Compute IK using MuJoCo's Jacobian
    """
    q = current_q.copy()
    
    for iteration in range(max_iter):
        # Set current joint positions
        for i, addr in enumerate(robot_arm_qpos_addrs):
            data.qpos[addr] = q[i]
        
        mujoco.mj_forward(mj_model, data)
        
        # Get current end effector position
        ee_pos = get_ee_position()
        
        # Position error
        error = target_pos - ee_pos
        error_mag = np.linalg.norm(error)
        
        if error_mag < 0.01:  # 1cm tolerance
            break
        
        # Compute Jacobian
        jacp = np.zeros((3, mj_model.nv))
        jacr = np.zeros((3, mj_model.nv))
        mujoco.mj_jacBody(mj_model, data, jacp, jacr, ee_body_id)
        
        # Extract Jacobian for robot arm joints only
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
        
        # Damped least squares
        lambda_damping = 0.01
        J_pinv = J.T @ np.linalg.inv(J @ J.T + lambda_damping * np.eye(3))
        
        # Compute joint velocity
        dq = J_pinv @ error
        
        # Update joints with step size
        alpha = 0.5
        q = q + alpha * dq
        
        # Clamp to joint limits
        for i, jname in enumerate(robot_arm_joints):
            jnt_id = mj_model.joint(jname).id
            q[i] = np.clip(q[i], 
                          mj_model.jnt_range[jnt_id][0], 
                          mj_model.jnt_range[jnt_id][1])
    
    return q

# Control state
selected_joint = 0

print("\n=== Hand Tracking IK Controller ===")
print("KEYBOARD CONTROLS:")
print("  W/S: Increase/Decrease selected joint")
print("  A/D: Select previous/next joint")
print("  R: Reset to initial pose")
print("  Q: Quit")
print("\nRobot will mirror human hand position in real-time")

# Initialize poses
human_arm_pose = np.array([0.3, -0.2, -1.5, 0.2, 0.0])
robot_arm_pose = np.array([0.0, 0.5, 0.0, -0.5, 0.0, 0.0, 0.0])

# Set initial human pose
for i, addr in enumerate(human_qpos_addrs):
    data.qpos[addr] = human_arm_pose[i]

# Set initial robot pose
for i, addr in enumerate(robot_arm_qpos_addrs):
    data.qpos[addr] = robot_arm_pose[i]

mujoco.mj_forward(mj_model, data)

step = 0

with mujoco.viewer.launch_passive(mj_model, data) as viewer:
    while viewer.is_running():
        # Set human arm position
        for i, addr in enumerate(human_qpos_addrs):
            data.qpos[addr] = human_arm_pose[i]
        
        mujoco.mj_forward(mj_model, data)
        
        # Get palm position
        palm_pos = get_palm_position()
        
        # Compute IK for robot arm
        robot_arm_pose = compute_jacobian_ik(palm_pos, robot_arm_pose)
        
        # Apply to robot arm
        for i, addr in enumerate(robot_arm_qpos_addrs):
            data.qpos[addr] = robot_arm_pose[i]
        
        # Step simulation
        mujoco.mj_step(mj_model, data)
        
        # Render camera view
        renderer.update_scene(data, camera=rgb_cam_id)
        rgb_img = renderer.render()
        
        # Visualize
        display_img = rgb_img.copy()
        
        # Get positions
        ee_pos = get_ee_position()
        error = np.linalg.norm(palm_pos - ee_pos)
        
        # Draw UI
        cv2.putText(display_img, f"Selected: {human_joint_names[selected_joint]}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(display_img, f"Value: {human_arm_pose[selected_joint]:.2f} rad", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        
        cv2.putText(display_img, f"Palm: ({palm_pos[0]:.2f}, {palm_pos[1]:.2f}, {palm_pos[2]:.2f})", 
                   (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(display_img, f"Error: {error*100:.1f}cm", 
                   (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        # Show controls
        cv2.putText(display_img, "W/S: +/- | A/D: Select | R: Reset", 
                   (10, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        cv2.imshow("Robot Camera - Hand Tracking", cv2.cvtColor(display_img, cv2.COLOR_RGB2BGR))
        
        # Handle keyboard
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('w'):
            # Increase joint angle
            jnt_id = mj_model.joint(human_joints[selected_joint]).id
            human_arm_pose[selected_joint] = min(
                human_arm_pose[selected_joint] + 0.05,
                mj_model.jnt_range[jnt_id][1]
            )
        elif key == ord('s'):
            # Decrease joint angle
            jnt_id = mj_model.joint(human_joints[selected_joint]).id
            human_arm_pose[selected_joint] = max(
                human_arm_pose[selected_joint] - 0.05,
                mj_model.jnt_range[jnt_id][0]
            )
        elif key == ord('d'):
            # Next joint
            selected_joint = (selected_joint + 1) % len(human_joints)
        elif key == ord('a'):
            # Previous joint
            selected_joint = (selected_joint - 1) % len(human_joints)
        elif key == ord('r'):
            # Reset
            human_arm_pose = np.array([0.3, -0.2, -1.5, 0.2, 0.0])
            robot_arm_pose = np.array([0.0, 0.5, 0.0, -0.5, 0.0, 0.0, 0.0])
            print("Reset to initial pose")
        
        viewer.sync()
        step += 1

cv2.destroyAllWindows()