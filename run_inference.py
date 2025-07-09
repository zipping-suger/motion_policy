import numpy as np
import time
import torch
import h5py
from tqdm.auto import tqdm
import pickle
import meshcat
import urchin
from pathlib import Path

from robofin.robots import FrankaRobot, FrankaRealRobot, FrankaGripper
from robofin.bullet import BulletController
from robofin.pointcloud.torch import FrankaSampler

from models.policynet import PolicyNet
from utils import normalize_franka_joints, unnormalize_franka_joints, convert_robotB_to_robotA_torch, convert_robotA_to_robotB
from data_loader import PointCloudTrajectoryDataset, DatasetType
from geometry import construct_mixed_point_cloud
from geometrout.primitive import Cuboid, Cylinder, Sphere
from robofin.pointcloud.torch import FrankaCollisionSampler
from geometry import TorchCuboids, TorchCylinders
from geometrout.transform import SE3, SO3

NUM_ROBOT_POINTS = 2048
NUM_OBSTACLE_POINTS = 4096
NUM_TARGET_POINTS = 128
MAX_ROLLOUT_LENGTH = 50
# Set this flag to True to always show expert trajectory, False to skip
SHOW_EXPERT_TRAJ = False
GOAL_THRESHOLD = 0.05  # 5cm threshold for goal reaching
NUM_DMEO = 10 
ACTION_SCALE = 0.1  # Scale for the action space

# model_path = "./checkpoints/697q7qbd/last.ckpt"
# val_data_path = "./pretrain_data/ompl_cubby_6k"

# model_path = "./checkpoints/n15mdb71/last.ckpt"
# val_data_path = "./pretrain_data/single_cubby_17k"

model_path = "./checkpoints/whfoy34g/last.ckpt"
val_data_path = "./pretrain_data/single_cubby_pretrain_8k"


# model = PolicyNet().to("cuda:0")
model = PolicyNet.load_from_checkpoint(model_path).cuda()
model.eval()

# Setup samplers
cpu_fk_sampler = FrankaSampler("cpu", use_cache=True)
gpu_fk_sampler = FrankaSampler("cuda:0", use_cache=True)
collision_sampler = FrankaCollisionSampler("cuda:0", with_base_link=False)

# Setup simulation
sim = BulletController(hz=12, substeps=20, gui=True)

# Load meshcat visualizer for point cloud visualization
viz = meshcat.Visualizer()
print(f"MeshCat URL: {viz.url()}")

# Load the URDF and setup visualization
urdf = urchin.URDF.load(FrankaRobot.urdf)
# Preload robot meshes in meshcat at neutral position
for idx, (k, v) in enumerate(urdf.visual_trimesh_fk(np.zeros(8)).items()):
    viz[f"robot/{idx}"].set_object(
        meshcat.geometry.TriangularMeshGeometry(k.vertices, k.faces),
        meshcat.geometry.MeshLambertMaterial(color=0xEEDD22, wireframe=False),
    )
    viz[f"robot/{idx}"].set_transform(v)

# Load the robot and gripper in the simulation
franka = sim.load_robot(FrankaRobot)
gripper = sim.load_robot(FrankaGripper, collision_free=True)

# Load validation data
dataset = PointCloudTrajectoryDataset(
    Path(val_data_path), 
    "global_solutions", 
    NUM_ROBOT_POINTS, 
    NUM_OBSTACLE_POINTS, 
    NUM_TARGET_POINTS,
    DatasetType.VAL,
    random_scale=0.0
)


def get_expert_trajectory(dataset, idx):
    """Extract the expert trajectory for a given problem index from the dataset."""
    with h5py.File(str(dataset._database), "r") as f:
        # Get the full trajectory of joint positions
        trajectory = f[dataset.trajectory_key][idx]
        
        return trajectory

# Select how many problems to visualize
num_problems = min(NUM_DMEO, len(dataset))
problems_to_visualize = range(num_problems)

for problem_idx in problems_to_visualize:
    print(f"\n======= Visualizing problem {problem_idx} =======")
    
    # Get data for this problem
    data = dataset[problem_idx]
    
    # Extract expert trajectory
    expert_trajectory = get_expert_trajectory(dataset, problem_idx)
    print(f"Expert trajectory length: {expert_trajectory.shape[0]} steps")
    
    # Target configuration of the expert trajectory
    target_config = expert_trajectory[-1]
    print(f"Target configuration of the expert: {target_config}")
    
    # Move tensors to GPU
    for key in data:
        if isinstance(data[key], torch.Tensor):
            data[key] = data[key].cuda()
    
    # Extract start configuration and target
    start_config = data["configuration"]  # This is normalized
    target_config = data["target_configuration"]
    target_pose = data["target_pose"]
    
    # Create point cloud for obstacles
    cuboid_centers = data["cuboid_centers"].cpu().numpy()
    cuboid_dims = data["cuboid_dims"].cpu().numpy()
    cuboid_quats = data["cuboid_quats"].cpu().numpy()
    
    cylinder_centers = data["cylinder_centers"].cpu().numpy()
    cylinder_radii = data["cylinder_radii"].cpu().numpy()
    cylinder_heights = data["cylinder_heights"].cpu().numpy()
    cylinder_quats = data["cylinder_quats"].cpu().numpy()
    
    # Create obstacle primitives
    cuboids = [
        Cuboid(c, d, q)
        for c, d, q in zip(
            list(cuboid_centers), list(cuboid_dims), list(cuboid_quats)
        )
        if not np.all(np.isclose(d, 0))  # Filter out zero-volume cuboids
    ]
    
    cylinders = [
        Cylinder(c, r, h, q)
        for c, r, h, q in zip(
            list(cylinder_centers),
            list(cylinder_radii.squeeze(1)),
            list(cylinder_heights.squeeze(1)),
            list(cylinder_quats),
        )
        if not np.isclose(r, 0) and not np.isclose(h, 0)  # Filter out zero-volume cylinders
    ]
    
    # Run model to get trajectory
    with torch.no_grad():
        # Add batch dimension for model input
        start_config_batched = start_config.unsqueeze(0)
        target_config_batched = target_config.unsqueeze(0)
        target_pose_batched = target_pose.unsqueeze(0)
        
        # Choose using config or eff-pose as target
        target_input = target_pose_batched
        
        # Generate rollout
        trajectory = []
        q = start_config_batched.clone()
        
        print(f"Start config: {q}")
        print(f"Target config: {target_config_batched}")
        
        # Unnormalize start config for visualization
        # unnorm_q = unnormalize_franka_joints(q)
        # trajectory.append(unnorm_q.squeeze(0).cpu().numpy())
        
        trajectory.append(q.squeeze(0).cpu().numpy())
        
        for i in range(MAX_ROLLOUT_LENGTH):
            # Forward pass through the model
            delta_q = model(q, target_input)*ACTION_SCALE
            q = q + delta_q
            
            # Unnormalize for visualization
            # unnorm_q = unnormalize_franka_joints(q)
            # trajectory.append(unnorm_q.squeeze(0).cpu().numpy())
            trajectory.append(q.squeeze(0).cpu().numpy())
            
            # Update point cloud with new robot position
            # robot_points = gpu_fk_sampler.sample(unnorm_q, NUM_ROBOT_POINTS)
            robot_points = gpu_fk_sampler.sample(q, NUM_ROBOT_POINTS)
            
            # Check if we've reached the target
            target_position = data["target_position"].unsqueeze(0)
            # current_position = gpu_fk_sampler.end_effector_pose(unnorm_q)[:, :3, -1]
            eff = gpu_fk_sampler.end_effector_pose(q)
            # Convert to Robot A tool frame
            eff = convert_robotB_to_robotA_torch(eff)
            current_position = eff[:, :3, -1]
            
            distance_to_target = torch.norm(current_position - target_position, dim=1)
            # print(f"Step {i+1}: Distance to target: {distance_to_target.item():.4f} m")
            if distance_to_target.item() < GOAL_THRESHOLD:  # 5cm threshold
                print(f"Reached target in {i+1} steps!")
                break

    print(f"Generated trajectory with {len(trajectory)} steps")
    
    # Load Obstacles in the simulation
    sim.load_primitives(cuboids + cylinders, color=[0.6, 0.6, 0.6, 1], # gray color for obstacles
                        visual_only=True)
    
    # Visualize obstacle point cloud in meshcat
    obstacle_points = construct_mixed_point_cloud(cuboids + cylinders, NUM_OBSTACLE_POINTS)
    obstacle_pc = obstacle_points[:, :3]  # Extract XYZ
    
    # Visualize target position with a small red sphere 
    # target_position = data["target_position"].cpu().numpy()
    # target_marker = Sphere(center=target_position, radius=0.05)
    # sim.load_sphere(target_marker, color=[1, 0, 0, 1], visual_only=True)
    
    target_config = data["target_configuration"].cpu().numpy().copy()
    target_pose = FrankaRealRobot.fk(target_config)
    mat = convert_robotA_to_robotB(target_pose.matrix)
    target_pose_se3 = SE3(matrix=mat)
    target_marker = Sphere(center=target_pose_se3.xyz, radius=0.05)
    sim.load_sphere(target_marker, color=[1, 0, 0, 1], visual_only=True)
    
    
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
    
    # Visualize target position with a small red sphere
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
    
    # Calculate end effector positions for expert final state
    expert_final_ee = FrankaRealRobot.fk(expert_trajectory[-1]).xyz
    print(f"Expert final position error: {np.linalg.norm(expert_final_ee - target_position):.4f} m")
    
    # Ask user if they want to see expert trajectory first
    # view_expert = input("View expert trajectory first? (y/n): ").strip().lower() == 'y'
    view_expert = SHOW_EXPERT_TRAJ
    
    if view_expert:
        print("Executing expert trajectory...")
        # Reset to start position
        franka.marionette(expert_trajectory[0])
        time.sleep(0.5)  # Give time to visualize initial state
        
        for q in tqdm(expert_trajectory):
            franka.control_position(q)
            sim.step()
            sim_config, _ = franka.get_joint_states()
            
            # Update meshcat visualization
            for idx, (k, v) in enumerate(urdf.visual_trimesh_fk(sim_config[:8]).items()):
                viz[f"robot/{idx}"].set_transform(v)
            time.sleep(0.05)
        
        # Reset robot to start configuration for policy trajectory
        print("Resetting to start configuration...")
        franka.marionette(trajectory[0])
        time.sleep(1.0)  # Give time to visualize initial state
        
        # Ask user if they want to continue to policy trajectory
        # input("Press Enter to continue to policy trajectory...")
    
    # Initialize robot to start configuration
    franka.marionette(trajectory[0])
    time.sleep(0.5)  # Give time to visualize initial state
    
    # Execute trajectory
    print(f"Executing policy trajectory...")
    for q in tqdm(trajectory):
        franka.control_position(q)
        sim.step()
        sim_config, _ = franka.get_joint_states()
        
        # Update meshcat visualization
        for idx, (k, v) in enumerate(urdf.visual_trimesh_fk(sim_config[:8]).items()):
            viz[f"robot/{idx}"].set_transform(v)
        time.sleep(0.05)
    
    # Get final position of policy trajectory and calculate error
    policy_final_config = trajectory[-1]
    policy_final_ee = FrankaRealRobot.fk(policy_final_config).xyz
    print(f"Policy final position error: {np.linalg.norm(policy_final_ee - target_position):.4f} m")
    
    # --- Collision checking for policy trajectory ---
    traj_tensor = torch.tensor(np.array(trajectory), dtype=torch.float32, device="cuda:0").unsqueeze(0)  # (1, T, 7)
    traj_flat = traj_tensor.reshape(-1, 7)  # (T, 7)

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
        spheres_reshaped = spheres.view(1, -1, num_spheres, 3)  # (1, T, S, 3)
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
    
    # Wait for user to press Enter to continue
    # input("Press Enter to continue to next problem...")