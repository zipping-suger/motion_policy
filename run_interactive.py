import numpy as np
import cv2
import time
import torch
from tqdm.auto import tqdm
from pathlib import Path
from geometrout.transform import SE3, SO3
from pyquaternion import Quaternion

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
NUM_TARGET_POINTS = 128
MAX_ROLLOUT_LENGTH = 50
GOAL_THRESHOLD = 0.01  # 1 cm threshold for goal reaching

model_path = "./checkpoints/sdrwmtfu/last.ckpt"
# model_path = "./checkpoints/dqu9herp/epoch-epoch=2-end.ckpt"
val_data_path = "./pretrain_data/ompl_table_6k"


def ensure_orthogonal_rotmat_polar(target_rotmat):
    target_rotmat = target_rotmat.reshape(3, 3)
    U, _, Vt = np.linalg.svd(target_rotmat)
    orthogonal_rotmat = U @ Vt
    
    # Ensure determinant is +1
    if np.linalg.det(orthogonal_rotmat) < 0:
        Vt[-1, :] *= -1
        orthogonal_rotmat = U @ Vt
    
    return orthogonal_rotmat


def move_target_with_key(target_pose, key, pos_step=0.02, rot_step=5.0):
    moved = False
    xyz = np.array(target_pose.xyz)
    so3 = target_pose.so3

    # Position changes
    if key == ord('w'):
        xyz = xyz + np.array([0, pos_step, 0])
        moved = True
    elif key == ord('s'):
        xyz = xyz + np.array([0, -pos_step, 0])
        moved = True
    elif key == ord('a'):
        xyz = xyz + np.array([-pos_step, 0, 0])
        moved = True
    elif key == ord('d'):
        xyz = xyz + np.array([pos_step, 0, 0])
        moved = True
    elif key == ord('q'):
        xyz = xyz + np.array([0, 0, pos_step])
        moved = True
    elif key == ord('e'):
        xyz = xyz + np.array([0, 0, -pos_step])
        moved = True

    # Orientation changes (in gripper's local frame)
    elif key in [ord('u'), ord('o'), ord('i'), ord('k'), ord('j'), ord('l')]:
        rot_step_rad = np.radians(rot_step)
        R = so3.matrix

        if key == ord('u'):  # Roll +
            dR = SO3.from_rpy(rot_step_rad, 0, 0).matrix
        elif key == ord('o'):  # Roll -
            dR = SO3.from_rpy(-rot_step_rad, 0, 0).matrix
        elif key == ord('i'):  # Pitch +
            dR = SO3.from_rpy(0, rot_step_rad, 0).matrix
        elif key == ord('k'):  # Pitch -
            dR = SO3.from_rpy(0, -rot_step_rad, 0).matrix
        elif key == ord('j'):  # Yaw +
            dR = SO3.from_rpy(0, 0, rot_step_rad).matrix
        elif key == ord('l'):  # Yaw -
            dR = SO3.from_rpy(0, 0, -rot_step_rad).matrix

        R_new = R @ dR
        R_new_ortho = ensure_orthogonal_rotmat_polar(R_new)
        so3 = SO3(Quaternion(matrix=R_new_ortho))
        moved = True

    if moved:
        target_pose = SE3(xyz=xyz, so3=so3)
    return moved, target_pose


model = PolicyNet.load_from_checkpoint(model_path).cuda()
model.eval()

cpu_fk_sampler = FrankaSampler("cpu", use_cache=True)
gpu_fk_sampler = FrankaSampler("cuda:0", use_cache=True)


sim = BulletController(hz=12, substeps=20, gui=True)
franka = sim.load_robot(FrankaRobot)
gripper = sim.load_robot(FrankaGripper, collision_free=True)

# Set the camera position using the new method signature
sim.set_camera_position(
    yaw=-90,           # degrees
    pitch=-30,        # degrees
    distance=2.5,     # meters
    target=[0.0, 0.0, 0.5]
)

dataset = PointCloudTrajectoryDataset(
    Path(val_data_path), 
    "global_solutions", 
    NUM_ROBOT_POINTS, 
    NUM_OBSTACLE_POINTS, 
    NUM_TARGET_POINTS,
    DatasetType.VAL,
    random_scale=0.0
)

problem_idx = 10
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


print("Use WASD (XY), QE (Z) to move position.")
print("Use U/O (roll), I/K (pitch), J/L (yaw) to rotate gripper.")
print("Press SPACE to plan and execute. Press ESC to quit.")

# Dummy OpenCV window (required for key input)
cv2.namedWindow("Control", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Control", 200, 100)
cv2.imshow("Control", np.zeros((100, 200), dtype=np.uint8))

policy_final_config = None  # To store the final configuration after policy execution

while True:
    key = cv2.waitKey(30) & 0xFF
    moved, target_pose = move_target_with_key(target_pose, key)
    if moved:
        target_franka.marionette(target_pose)

    sim.step()
    time.sleep(0.03)

    if key == 27:  # ESC
        print("Exiting interactive session.")
        break
    elif key == 32:  # SPACE
        print("Planning and executing trajectory...")

        # Plan and execute trajectory
        with torch.no_grad():
            xyz = data["xyz"].unsqueeze(0)
            if policy_final_config is None:
                start_config = data["configuration"]
            else:
                start_config = torch.tensor(policy_final_config, dtype=torch.float32).cuda()
            
            start_config_batched = start_config.unsqueeze(0)
            
            # Solve Inverse Kinematics to get target configuration
            # Construct the desired end-effector pose as an SE3 object
            target_xyz = torch.tensor(target_pose.xyz, dtype=torch.float32, device=xyz.device).unsqueeze(0)  # shape (1, 3)
            target_rotmat = torch.tensor(target_pose.so3.matrix, dtype=torch.float32, device=xyz.device).reshape(1, 9)
            target_input = torch.cat([target_xyz, target_rotmat], dim=-1)  # shape (1, 12)

            trajectory = []
            q = start_config_batched.clone()
            trajectory.append(q.squeeze(0).cpu().numpy())

            for i in range(MAX_ROLLOUT_LENGTH):
                delta_q = model(xyz, q, target_input)
                q = q + delta_q
                trajectory.append(q.squeeze(0).cpu().numpy())
                robot_points = gpu_fk_sampler.sample(q, NUM_ROBOT_POINTS)
                xyz[:, :NUM_ROBOT_POINTS, :3] = robot_points
                
                target_pose_mat = torch.tensor(
                    target_pose.matrix,
                    dtype=torch.float32,
                    device="cuda:0"
                        ).unsqueeze(0)
                
                target_points = gpu_fk_sampler.sample_end_effector(
                    target_pose_mat, 
                    NUM_TARGET_POINTS
                    ).squeeze(0)
                xyz[:, NUM_ROBOT_POINTS + NUM_OBSTACLE_POINTS:, :3] = target_points
                
                current_position = gpu_fk_sampler.end_effector_pose(q)[:, :3, -1]
                distance_to_target = torch.norm(current_position - target_xyz, dim=1)
                if distance_to_target.item() < GOAL_THRESHOLD:
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
        policy_final_ee = FrankaRobot.fk(policy_final_config).xyz
        policy_final_ee_np = np.array(policy_final_ee)
        target_pose_xyz_np = np.array(target_pose.xyz)
        error = np.linalg.norm(policy_final_ee_np - target_pose_xyz_np)
        print(
            f"Policy final position error: {error:.4f} m"
        )

        # Extra time to view final pose
        for _ in range(10):
            sim.step()
            time.sleep(0.05)