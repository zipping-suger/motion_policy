import numpy as np
import cv2
import time
import torch
from tqdm.auto import tqdm
from pathlib import Path
from geometrout.transform import SE3

from robofin.robots import FrankaRobot, FrankaGripper
from robofin.bullet import BulletController
from robofin.pointcloud.torch import FrankaSampler

from models.policynet import PolicyNet
from utils import normalize_franka_joints, unnormalize_franka_joints
from data_loader import PointCloudTrajectoryDataset, DatasetType
from geometry import construct_mixed_point_cloud
from geometrout.primitive import Cuboid, Cylinder, Sphere

NUM_ROBOT_POINTS = 2048
NUM_OBSTACLE_POINTS = 4096
MAX_ROLLOUT_LENGTH = 100

model_path = "./checkpoints/table_30k_optc/last.ckpt"
val_data_path = "./pretrain_data/ompl_table_30k"

def move_target_config_with_key(target_config, key, step=0.05):
    """
    Move the target joint configuration with keyboard keys.
    1-7: Increase joint angle
    Shift+1-7: Decrease joint angle
    """
    moved = False
    # Keys 1-7 increase joint angles
    for i in range(7):
        if key == ord(str(i + 1)):
            target_config[i] += step
            moved = True
        # Shift+1-7 decrease joint angles (ASCII for '!' is 33, '@' is 64, etc.)
        elif key == ord('!') + i:
            target_config[i] -= step
            moved = True
    return moved

def random_ik_with_retries(pose, eff_frame="right_gripper", retries=1000):
    """
    Try to solve IK up to `retries` times, return the first solution found.
    Returns None if no solution is found.
    """
    for _ in range(retries):
        try:
            solutions = FrankaRobot.random_ik(pose, eff_frame)
            if solutions and len(solutions) > 0:
                return solutions[0]
        except Exception:
            continue
    return None


model = PolicyNet.load_from_checkpoint(model_path).cuda()
model.eval()

cpu_fk_sampler = FrankaSampler("cpu", use_cache=True)
gpu_fk_sampler = FrankaSampler("cuda:0", use_cache=True)

sim = BulletController(hz=12, substeps=20, gui=True)
franka = sim.load_robot(FrankaRobot, hd=False, collision_free=True)

dataset = PointCloudTrajectoryDataset(
    Path(val_data_path), 
    "global_solutions", 
    NUM_ROBOT_POINTS, 
    NUM_OBSTACLE_POINTS, 
    DatasetType.VAL
)

problem_idx = 0
print(f"\n======= Visualizing problem {problem_idx} =======")
data = dataset[problem_idx]
for key in data:
    if isinstance(data[key], torch.Tensor):
        data[key] = data[key].cuda()


cuboid_centers = data["cuboid_centers"].cpu().numpy()
cuboid_dims = data["cuboid_dims"].cpu().numpy()
cuboid_quats = data["cuboid_quats"].cpu().numpy()
cylinder_centers = data["cylinder_centers"].cpu().numpy()
cylinder_radii = data["cylinder_radii"].cpu().numpy()
cylinder_heights = data["cylinder_heights"].cpu().numpy()
cylinder_quats = data["cylinder_quats"].cpu().numpy()

cuboids = [
    Cuboid(c, d, q)
    for c, d, q in zip(
        list(cuboid_centers), list(cuboid_dims), list(cuboid_quats)
    )
    if not np.all(np.isclose(d, 0))
]
cylinders = [
    Cylinder(c, r, h, q)
    for c, r, h, q in zip(
        list(cylinder_centers),
        list(cylinder_radii.squeeze(1)),
        list(cylinder_heights.squeeze(1)),
        list(cylinder_quats),
    )
    if not np.isclose(r, 0) and not np.isclose(h, 0)
]

# Load obstacles once
sim.load_primitives(cuboids + cylinders, color=[0.6, 0.6, 0.6, 1], visual_only=True)
franka.marionette(data["configuration"].cpu().numpy())

# Initial target robot configuration
target_franka = sim.load_robot(FrankaGripper, collision_free=True)
target_config = data["target_configuration"].cpu().numpy().copy()
target_pose = FrankaRobot.fk(target_config)
target_franka.marionette(target_pose)
target_position = target_pose.xyz

print("Use 1-7 to increase joint angles, Shift+1-7 to decrease. Press SPACE to plan and execute. Press ESC to quit.")

# Dummy OpenCV window (required for key input)
cv2.namedWindow("Control", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Control", 200, 100)
cv2.imshow("Control", np.zeros((100, 200), dtype=np.uint8))

policy_final_config = None  # To store the final configuration after policy execution

while True:
    key = cv2.waitKey(30) & 0xFF  # Non-blocking key press check
    moved = move_target_config_with_key(target_config, key)
    if moved:
        target_pose = FrankaRobot.fk(target_config)
        target_franka.marionette(target_pose)
        target_position = target_pose.xyz

    sim.step()
    time.sleep(0.03)

    if key == 27:  # ESC
        print("Exiting interactive session.")
        break
    elif key == 32:  # SPACE
        print("Planning and executing trajectory...")
        # Update data with new target configuration
        data["target_configuration"] = torch.tensor(target_config, dtype=torch.float32).cuda()

        # Plan and execute trajectory
        with torch.no_grad():
            xyz = data["xyz"].unsqueeze(0)
            if policy_final_config is None:
                start_config = data["configuration"]
            else:
                start_config = torch.tensor(policy_final_config, dtype=torch.float32).cuda()
            
            start_config_batched = start_config.unsqueeze(0)
            
            start_config_batched = start_config.unsqueeze(0)
            target_config_tensor = torch.tensor(target_config, dtype=torch.float32).cuda()
            target_config_batched = target_config_tensor.unsqueeze(0)
            target_input = target_config_batched

            trajectory = []
            q = start_config_batched.clone()
            trajectory.append(q.squeeze(0).cpu().numpy())

            for i in range(MAX_ROLLOUT_LENGTH):
                delta_q = model(xyz, q, target_input)
                q = q + delta_q
                trajectory.append(q.squeeze(0).cpu().numpy())
                robot_points = gpu_fk_sampler.sample(q, NUM_ROBOT_POINTS)
                xyz[:, :NUM_ROBOT_POINTS, :3] = robot_points
                current_position = gpu_fk_sampler.end_effector_pose(q)[:, :3, -1]
                distance_to_target = torch.norm(current_position - torch.tensor(target_position, device=current_position.device).unsqueeze(0), dim=1)
                if distance_to_target.item() < 0.05:
                    print(f"Reached target in {i+1} steps!")
                    break

        print(f"Generated trajectory with {len(trajectory)} steps")
        franka.marionette(trajectory[0])
        time.sleep(0.2)
        print(f"Executing policy trajectory...")
        for q in tqdm(trajectory):
            franka.control_position(q)
            sim.step()
            sim_config, _ = franka.get_joint_states()
            time.sleep(0.08)

        # Get final position of policy trajectory and calculate error
        policy_final_config = trajectory[-1]
        policy_final_ee = np.array(FrankaRobot.fk(policy_final_config).xyz)
        print(f"Policy final position error: {np.linalg.norm(policy_final_ee - np.array(target_position)):.4f} m")

        # Extra time to view final pose
        for _ in range(10):
            sim.step()
            time.sleep(0.05)