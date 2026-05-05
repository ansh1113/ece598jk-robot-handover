import mujoco
import mujoco.viewer
import os

os.chdir("src/robotis_mujoco_menagerie/robotis_ffw")

mj_model = mujoco.MjModel.from_xml_path("scene_with_object.xml")
data = mujoco.MjData(mj_model)

print(f"Number of bodies: {mj_model.nbody}")
print(f"Number of DOFs: {mj_model.nv}")
print(f"Number of joints: {mj_model.njnt}")

print("\nBodies:")
for i in range(mj_model.nbody):
    print(f"{i}: {mj_model.body(i).name}")

print("\nJoints:")
for i in range(mj_model.njnt):
    jnt = mj_model.joint(i)
    print(f"{i}: {jnt.name} (type: {jnt.type})")

print("\nInitial qpos:", data.qpos[:10])
print("Initial qvel:", data.qvel[:10])

# Try to move the robot
with mujoco.viewer.launch_passive(mj_model, data) as viewer:
    step = 0
    while viewer.is_running() and step < 1000:
        # Apply forward velocity to first 6 DOFs (freejoint)
        if step < 500:
            data.qvel[0] = 0.5  # X velocity
            data.qvel[1] = 0.0  # Y velocity
            data.qvel[2] = 0.0  # Z velocity
        
        mujoco.mj_step(mj_model, data)
        
        if step % 100 == 0:
            print(f"Step {step}: pos={data.qpos[0]:.3f}, vel={data.qvel[0]:.3f}")
        
        viewer.sync()
        step += 1