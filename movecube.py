import mujoco
import mujoco.viewer
import numpy as np
import time


model_path = "scenenew.xml"
model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)

# Target configurations (joint angles) for each stage
targets = {
    'pre_grasp': np.array([1.26, -1.63, 2, 0.754, 1.67, 0.691]),  # Open gripper
    'grasp': np.array([1.26, -1.63, 2.2, 0.754, 1.92, 0.06]),      # Close gripper
    'lift': np.array([1.26, -1.63, 1, 0.754, 1.92, 0.06]),       # Lift up
    'pre_drop': np.array([-1.3, -0.848, 0, 1.81, 1.91, 0.06]),   # Above open box
    'drop': np.array([-1.3, -0.848, 0, 1.81, 1.91, 0.691])       # Release
}

# Controller parameters
kp = 100
kd = 5
tolerance = 0.08
stages = ['pre_grasp', 'grasp', 'lift', 'pre_drop', 'drop']
current_stage = 0

# Initial box position and quaternion (from XML)
box_initial_pos = [0.2, -0.1, 0.1]  # x, y, z
box_initial_quat = [1, 0, 0, 0]       # w, x, y, z

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        # Reset small box if it falls (7 DOF: 3 pos + 4 quat)
        if data.qpos[9] < 0.05:  # Check z-position (index 9 = 6+3)
            data.qpos[6:13] = [*box_initial_pos, *box_initial_quat]
            data.qvel[6:12] = [0]*6
            mujoco.mj_forward(model, data)

        # PD Control for arm joints (indices 0-5)
        target = targets[stages[current_stage]]
        errors = []
        for i in range(6):
            error = target[i] - data.qpos[i]
            data.ctrl[i] = kp * error + kd * -data.qvel[i]
            errors.append(abs(error))

        # Stage transitions
        if max(errors) < tolerance:
            print(f"Completed {stages[current_stage]}")
            current_stage += 1
            if current_stage >= len(stages):
                break
            time.sleep(0.5)

        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(model.opt.timestep)
print("Task completed successfully!")
