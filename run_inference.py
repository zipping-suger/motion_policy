import numpy as np
import time
import torch
import h5py
from tqdm.auto import tqdm
import pickle
import meshcat
import urchin
from pathlib import Path

from robofin.robots import FrankaRobot, FrankaGripper
from robofin.bullet import BulletController
from robofin.pointcloud.torch import FrankaSampler

# Changed to MotionPolicyNetwork
from models.mpinet import MotionPolicyNetwork
from utils import normalize_franka_joints, unnormalize_franka_joints
from data_loader import PointCloudTrajectoryDataset, DatasetType
from geometry import construct_mixed_point_cloud
from geometrout.primitive import Cuboid, Cylinder, Sphere
from robofin.pointcloud.torch import FrankaCollisionSampler
from geometry import TorchCuboids, TorchCylinders

NUM_ROBOT_POINTS = 2048
NUM_OBSTACLE_POINTS = 4096
NUM_TARGET_POINTS = 128
MAX_ROLLOUT_LENGTH = 150 # Stay the same as the original code from MPInet
SHOW_EXPERT_TRAJ = False
GOAL_THRESHOLD = 0.05  # 5cm threshold for goal reaching
NUM_DEMO = 10 

model_path = "mpinets_hybrid_expert.ckpt"
# val_data_path = "./pretrain_data/ompl_cubby_22k"
val_data_path = "./pretrain_data/ompl_table_6k"

# Load MotionPolicyNetwork
model = MotionPolicyNetwork.load_from_checkpoint(model_path).cuda()
model.eval()

# Setup samplers
cpu_fk_sampler = FrankaSampler("cpu", use_cache=True)
gpu_fk_sampler = FrankaSampler("cuda:0", use_cache=True)
collision_sampler = FrankaCollisionSampler("cuda:0", with_base_link=False)

# Setup simulation
sim = BulletController(hz=12, substeps=20, gui=True)

# Load meshcat visualizer
viz = meshcat.Visualizer()
print(f"MeshCat URL: {viz.url()}")

# Load URDF and setup visualization
urdf = urchin.URDF.load(FrankaRobot.urdf)
for idx, (k, v) in enumerate(urdf.visual_trimesh_fk(np.zeros(8)).items()):
    viz[f"robot/{idx}"].set_object(
        meshcat.geometry.TriangularMeshGeometry(k.vertices, k.faces),
        meshcat.geometry.MeshLambertMaterial(color=0xEEDD22, wireframe=False),
    )
    viz[f"robot/{idx}"].set_transform(v)

# Load robot and gripper
franka = sim.load_robot(FrankaRobot)
target_franka = sim.load_robot(FrankaGripper, collision_free=True)

# Load validation data
dataset = PointCloudTrajectoryDataset(
    Path(val_data_path), 
    "global_solutions", 
    NUM_ROBOT_POINTS, 
    NUM_OBSTACLE_POINTS, 
    NUM_TARGET_POINTS,
    DatasetType.VAL
)


def get_expert_trajectory(dataset, idx):
    with h5py.File(str(dataset._database), "r") as f:
        trajectory = f[dataset.trajectory_key][idx]
        return trajectory


num_problems = min(NUM_DEMO, len(dataset))
problems_to_visualize = range(num_problems)

for problem_idx in problems_to_visualize:
    print(f"\n======= Visualizing problem {problem_idx} =======")
    
    data = dataset[problem_idx]
    expert_trajectory = get_expert_trajectory(dataset, problem_idx)
    
    # Initial target robot configuration
    target_config = data["target_configuration"].cpu().numpy().copy()
    target_pose = FrankaRobot.fk(target_config)
    target_franka.marionette(target_pose)
    
    # Move tensors to GPU
    for key in data:
        if isinstance(data[key], torch.Tensor):
            data[key] = data[key].cuda()
    
    # Use XYZ only without segmentation mask
    xyz = data["xyz"].unsqueeze(0)
    
    # Use normalized configuration for model input
    q_norm = normalize_franka_joints(data["configuration"]).unsqueeze(0)
    
    # Generate rollout
    trajectory = []
    
    # Unnormalize start config for visualization
    q_unnorm = unnormalize_franka_joints(q_norm)
    trajectory.append(q_unnorm.squeeze(0).cpu().detach().numpy())
    
    for i in range(MAX_ROLLOUT_LENGTH):
        # Forward pass through model with normalized config
        delta_q = model(xyz, q_norm)
        q_norm = q_norm + delta_q
        
        # Unnormalize for visualization and FK
        q_unnorm = unnormalize_franka_joints(q_norm)
        trajectory.append(q_unnorm.squeeze(0).cpu().detach().numpy())
        
        # Update point cloud with new robot position
        robot_points = gpu_fk_sampler.sample(q_unnorm, NUM_ROBOT_POINTS)
        xyz[:, :NUM_ROBOT_POINTS, :3] = robot_points
        
        # Check if we've reached the target
        target_position = data["target_position"].unsqueeze(0)
        current_position = gpu_fk_sampler.end_effector_pose(q_unnorm)[:, :3, -1]
        
        distance_to_target = torch.norm(current_position - target_position, dim=1)
        if distance_to_target.item() < GOAL_THRESHOLD:
            print(f"Reached target in {i+1} steps!")
            break

    print(f"Generated trajectory with {len(trajectory)} steps")
    
    # Create obstacle primitives from data
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
    
    # Load obstacles in simulation
    sim.load_primitives(cuboids + cylinders, color=[0.6, 0.6, 0.6, 1], visual_only=True)
    
    # Visualize obstacle point cloud in meshcat
    obstacle_points = construct_mixed_point_cloud(cuboids + cylinders, NUM_OBSTACLE_POINTS)
    obstacle_pc = obstacle_points[:, :3]
    
    # Create color array for points (green for obstacles)
    point_cloud_colors = np.zeros((3, obstacle_pc.shape[0]))
    point_cloud_colors[1, :] = 1.0  # Green for obstacles
    
    viz["point_cloud"].set_object(
        meshcat.geometry.PointCloud(
            position=obstacle_pc.T,
            color=point_cloud_colors,
            size=0.005,
        )
    )
    
    # Visualize target position
    target_position = data["target_position"].cpu().numpy()
    viz["target"].set_object(
        meshcat.geometry.Sphere(0.03),
        meshcat.geometry.MeshLambertMaterial(color=0xFF0000)
    )
    viz["target"].set_transform(
        meshcat.transformations.translation_matrix(target_position)
    )
    
    # Calculate trajectories statistics
    print("\n=== Trajectory Comparison ===")
    print(f"Expert trajectory: {expert_trajectory.shape[0]} steps")
    print(f"Policy trajectory: {len(trajectory)} steps")
    
    # Calculate end effector positions
    expert_final_ee = FrankaRobot.fk(expert_trajectory[-1]).xyz
    policy_final_ee = FrankaRobot.fk(trajectory[-1]).xyz
    print(f"Expert final position error: {np.linalg.norm(expert_final_ee - target_position):.4f} m")
    print(f"Policy final position error: {np.linalg.norm(policy_final_ee - target_position):.4f} m")
    
    if SHOW_EXPERT_TRAJ:
        print("Executing expert trajectory...")
        franka.marionette(expert_trajectory[0])
        time.sleep(0.5)
        
        for q in tqdm(expert_trajectory):
            franka.control_position(q)
            sim.step()
            sim_config, _ = franka.get_joint_states()
            
            # Update meshcat visualization
            for idx, (k, v) in enumerate(urdf.visual_trimesh_fk(sim_config[:8]).items()):
                viz[f"robot/{idx}"].set_transform(v)
            time.sleep(0.05)
        
        print("Resetting to start configuration...")
        franka.marionette(trajectory[0])
        time.sleep(1.0)
    
    # Initialize robot to start configuration
    franka.marionette(trajectory[0])
    time.sleep(0.5)
    
    # Execute policy trajectory
    print(f"Executing policy trajectory...")
    for q in tqdm(trajectory):
        franka.control_position(q)
        sim.step()
        sim_config, _ = franka.get_joint_states()
        
        # Update meshcat visualization
        for idx, (k, v) in enumerate(urdf.visual_trimesh_fk(sim_config[:8]).items()):
            viz[f"robot/{idx}"].set_transform(v)
        time.sleep(0.05)
    
    # Collision checking for policy trajectory
    traj_tensor = torch.tensor(np.array(trajectory), dtype=torch.float32, device="cuda:0").unsqueeze(0)
    traj_flat = traj_tensor.reshape(-1, 7)

    cuboids_torch = TorchCuboids(
        data["cuboid_centers"].unsqueeze(0),
        data["cuboid_dims"].unsqueeze(0),
        data["cuboid_quats"].unsqueeze(0)
    )
    cylinders_torch = TorchCylinders(
        data["cylinder_centers"].unsqueeze(0),
        data["cylinder_radii"].unsqueeze(0),
        data["cylinder_heights"].unsqueeze(0),
        data["cylinder_quats"].unsqueeze(0)
    )

    collision_spheres = collision_sampler.compute_spheres(traj_flat)
    has_collision = False

    for radius, spheres in collision_spheres:
        num_spheres = spheres.size(-2)
        spheres_reshaped = spheres.view(1, -1, num_spheres, 3)
        sdf_cuboids = cuboids_torch.sdf_sequence(spheres_reshaped)
        sdf_cylinders = cylinders_torch.sdf_sequence(spheres_reshaped)
        sdf_values = torch.minimum(sdf_cuboids, sdf_cylinders)
        collision_mask = sdf_values <= radius
        if torch.any(collision_mask):
            has_collision = True
            break

    print(f"Policy trajectory collision: {'YES' if has_collision else 'NO'}")
    
    # Extra time to view final pose
    for _ in range(20):
        sim.step()
        sim_config, _ = franka.get_joint_states()
        for idx, (k, v) in enumerate(urdf.visual_trimesh_fk(sim_config[:8]).items()):
            viz[f"robot/{idx}"].set_transform(v)
        time.sleep(0.05)
    
    # Clear obstacles before next problem
    sim.clear_all_obstacles()