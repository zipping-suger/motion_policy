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

# Updated model import
from models.mpinet import MotionPolicyNetwork
from utils import normalize_franka_joints, unnormalize_franka_joints
from data_loader import PointCloudTrajectoryDataset, DatasetType
from geometry import construct_mixed_point_cloud
from geometrout.primitive import Cuboid, Cylinder

NUM_ROBOT_POINTS = 2048
NUM_OBSTACLE_POINTS = 4096
NUM_TARGET_POINTS = 128
MAX_ROLLOUT_LENGTH = 75  # Stay the same as the original code from MPInet
GOAL_THRESHOLD = 0.01  # 1 cm threshold for goal reaching

model_path = "mpinets_hybrid_expert.ckpt"
val_data_path = "./pretrain_data/ompl_table_6k"

def create_point_cloud(robot_points, obstacle_points, target_points):
    pc = torch.zeros(
        NUM_ROBOT_POINTS + NUM_OBSTACLE_POINTS + NUM_TARGET_POINTS, 
        4,  # x,y,z + segmentation mask
        device="cuda:0"
    )
    # Robot points (mask=0)
    pc[:NUM_ROBOT_POINTS, :3] = robot_points
    pc[:NUM_ROBOT_POINTS, 3] = 0
    
    # Obstacle points (mask=1)
    mid_start = NUM_ROBOT_POINTS
    mid_end = mid_start + NUM_OBSTACLE_POINTS
    pc[mid_start:mid_end, :3] = obstacle_points
    pc[mid_start:mid_end, 3] = 1
    
    # Target points (mask=2)
    pc[mid_end:, :3] = target_points
    pc[mid_end:, 3] = 2
    
    return pc.unsqueeze(0)  # Add batch dimension

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

# Load MotionPolicyNetwork
model = MotionPolicyNetwork.load_from_checkpoint(model_path).cuda()
model.eval()

cpu_fk_sampler = FrankaSampler("cpu", use_cache=True)
gpu_fk_sampler = FrankaSampler("cuda:0", use_cache=True)

sim = BulletController(hz=12, substeps=20, gui=True)
franka = sim.load_robot(FrankaRobot)
gripper = sim.load_robot(FrankaGripper, collision_free=True)

# Set camera
sim.set_camera_position(
    yaw=-90, pitch=-30, distance=2.5, target=[0.0, 0.0, 0.5]
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

problem_idx = 2
print(f"\n======= Visualizing problem {problem_idx} =======")
data = dataset[problem_idx]
for key in data:
    if isinstance(data[key], torch.Tensor):
        data[key] = data[key].cuda()

# Extract obstacles
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
        list(cuboid_centers), list(cuboid_dims), list(cuboid_quats))
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

# Precompute obstacle points once
obstacle_points = construct_mixed_point_cloud(cuboids + cylinders, NUM_OBSTACLE_POINTS)
obstacle_points_tensor = torch.tensor(
    obstacle_points[:, :3], 
    dtype=torch.float32, 
    device="cuda:0"
)

# Load obstacles
sim.load_primitives(cuboids + cylinders, color=[0.6, 0.6, 0.6, 1], visual_only=True)
franka.marionette(data["configuration"].cpu().numpy())

# Initial target pose
target_franka = sim.load_robot(FrankaGripper, collision_free=True)
target_config = data["target_configuration"].cpu().numpy().copy()
target_pose = FrankaRobot.fk(target_config)
target_franka.marionette(target_pose)

print("Use WASD (XY), QE (Z) to move position.")
print("Use U/O (roll), I/K (pitch), J/L (yaw) to rotate gripper.")
print("Press SPACE to plan and execute. Press ESC to quit.")

cv2.namedWindow("Control", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Control", 200, 100)
cv2.imshow("Control", np.zeros((100, 200), dtype=np.uint8))

policy_final_config = None

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
        
        # Get start configuration
        if policy_final_config is None:
            start_config = data["configuration"].cpu().numpy()
        else:
            start_config = policy_final_config
        
        # Convert to tensor
        current_q = torch.tensor(
            start_config, 
            dtype=torch.float32,
            device="cuda:0"
        ).unsqueeze(0)
        q_norm = normalize_franka_joints(current_q)
        
        trajectory = []
        trajectory.append(start_config.copy())
        
        for i in range(MAX_ROLLOUT_LENGTH):
            # Sample points
            robot_points = gpu_fk_sampler.sample(
                current_q, 
                NUM_ROBOT_POINTS
            ).squeeze(0)
            
            target_pose_mat = torch.tensor(
                target_pose.matrix,
                dtype=torch.float32,
                device="cuda:0"
            ).unsqueeze(0)
            target_points = gpu_fk_sampler.sample_end_effector(
                target_pose_mat, 
                NUM_TARGET_POINTS
            ).squeeze(0)
            
            # Create point cloud
            xyz = create_point_cloud(
                robot_points, 
                obstacle_points_tensor, 
                target_points
            )
            
            # Policy prediction
            delta_q = model(xyz, q_norm)
            q_norm = torch.clamp(q_norm + delta_q, min=-1, max=1)
            current_q = unnormalize_franka_joints(q_norm)
            current_config = current_q.squeeze(0).detach().cpu().numpy()
            trajectory.append(current_config.copy())
            
            # Check termination
            current_ee = FrankaRobot.fk(current_config).xyz
            distance = np.linalg.norm(np.array(current_ee) - np.array(target_pose.xyz))
            if distance < GOAL_THRESHOLD:
                print(f"Reached target in {i+1} steps!")
                break
        
        print(f"Generated trajectory with {len(trajectory)} steps")
        franka.marionette(trajectory[0])
        time.sleep(0.2)
        
        print(f"Executing policy trajectory...")
        for q in tqdm(trajectory):
            franka.control_position(q)
            sim.step()
            time.sleep(0.08)
        
        # Store final configuration
        policy_final_config = trajectory[-1]
        policy_final_ee = FrankaRobot.fk(policy_final_config).xyz
        error = np.linalg.norm(np.array(policy_final_ee) - np.array(target_pose.xyz))
        print(f"Policy final position error: {error:.4f} m")
        
        # Pause at final pose
        for _ in range(10):
            sim.step()
            time.sleep(0.05)