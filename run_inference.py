import numpy as np
import time
import torch
import h5py
from tqdm.auto import tqdm
import pickle
import meshcat
import urchin
from pathlib import Path
from typing import List, Union
from robofin.robots import FrankaRobot, FrankaGripper
from robofin.bullet import BulletController
from robofin.pointcloud.torch import FrankaSampler, FrankaCollisionSampler

from models.mpinet import MotionPolicyNetwork
from utils import normalize_franka_joints, unnormalize_franka_joints
from data_loader import PointCloudTrajectoryDataset, DatasetType
from geometry import construct_mixed_point_cloud
from geometrout.primitive import Cuboid, Cylinder
from geometry import TorchCuboids, TorchCylinders
from geometrout.transform import SE3

NUM_ROBOT_POINTS = 2048
NUM_OBSTACLE_POINTS = 4096
NUM_TARGET_POINTS = 128
MAX_ROLLOUT_LENGTH = 150
GOAL_THRESHOLD = 0.01  # 1cm threshold for goal reaching
ANGLE_THRESHOLD = 15  # 15 degrees threshold for orientation
NUM_DEMO = 10 

model_path = "mpinets_hybrid_expert.ckpt"
val_data_path = "./pretrain_data/ompl_cubby_6k"

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


def make_point_cloud_from_primitives(
    q0: torch.Tensor,
    target: SE3,
    obstacles: List[Union[Cuboid, Cylinder]],
    fk_sampler: FrankaSampler,
) -> torch.Tensor:
    """
    Creates point cloud from primitives instead of using pre-sampled points
    """
    obstacle_points = construct_mixed_point_cloud(obstacles, NUM_OBSTACLE_POINTS)
    robot_points = fk_sampler.sample(q0, NUM_ROBOT_POINTS)
    
    target_points = fk_sampler.sample_end_effector(
        torch.as_tensor(target.matrix).type_as(robot_points).unsqueeze(0),
        num_points=NUM_TARGET_POINTS,
    )
    
    xyz = torch.cat(
        (
            torch.zeros(NUM_ROBOT_POINTS, 4),
            torch.ones(NUM_OBSTACLE_POINTS, 4),
            2 * torch.ones(NUM_TARGET_POINTS, 4),
        ),
        dim=0,
    )
    xyz[:NUM_ROBOT_POINTS, :3] = robot_points.float()
    xyz[NUM_ROBOT_POINTS:NUM_ROBOT_POINTS+NUM_OBSTACLE_POINTS, :3] = torch.as_tensor(obstacle_points[:, :3]).float()
    xyz[NUM_ROBOT_POINTS+NUM_OBSTACLE_POINTS:, :3] = target_points.float()
    
    return xyz


def rollout_until_success(
    model: MotionPolicyNetwork,
    q0: np.ndarray,
    target: SE3,
    obstacles: List[Union[Cuboid, Cylinder]],
    fk_sampler: FrankaSampler,
) -> np.ndarray:
    """
    Improved rollout logic with both position and orientation checking
    """
    q = torch.as_tensor(q0).unsqueeze(0).float().cuda()
    trajectory = [q]
    q_norm = normalize_franka_joints(q)
    
    # Create initial point cloud from primitives
    point_cloud = make_point_cloud_from_primitives(q, target, obstacles, fk_sampler).unsqueeze(0).cuda()
    
    def sampler(config):
        return fk_sampler.sample(config, NUM_ROBOT_POINTS)

    for i in range(MAX_ROLLOUT_LENGTH):
        q_norm = torch.clamp(q_norm + model(point_cloud, q_norm), min=-1, max=1)
        qt = unnormalize_franka_joints(q_norm)
        trajectory.append(qt)
        
        # Get current end effector pose
        eff_pose = FrankaRobot.fk(qt.squeeze().detach().cpu().numpy(), eff_frame="right_gripper")
        
        # Check position and orientation thresholds
        position_error = np.linalg.norm(eff_pose._xyz - target._xyz)
        orientation_error = np.abs(np.degrees((eff_pose.so3._quat * target.so3._quat.conjugate).radians))
        
        if position_error < GOAL_THRESHOLD and orientation_error < ANGLE_THRESHOLD:
            print(f"Reached target in {i+1} steps! (Position error: {position_error:.4f}m, Orientation error: {orientation_error:.2f}°)")
            break
            
        # Update point cloud with new robot configuration
        samples = sampler(qt).type_as(point_cloud)
        point_cloud[:, : samples.shape[1], :3] = samples

    return np.asarray([t.squeeze().detach().cpu().numpy() for t in trajectory])


num_problems = min(NUM_DEMO, len(dataset))
problems_to_visualize = range(num_problems)

for problem_idx in problems_to_visualize:
    print(f"\n======= Visualizing problem {problem_idx} =======")
    
    data = dataset[problem_idx]
    expert_trajectory = get_expert_trajectory(dataset, problem_idx)
        
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
        for c, d, q in zip(list(cuboid_centers), list(cuboid_dims), list(cuboid_quats))
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
    
    obstacles = cuboids + cylinders
    
    # Get target pose from target configuration
    target_pose = FrankaRobot.fk(data["target_configuration"].cpu().numpy(), eff_frame="right_gripper")
    
    # Generate rollout using improved function
    trajectory = rollout_until_success(
        model,
        data["configuration"].cpu().numpy(),
        target_pose,
        obstacles,
        gpu_fk_sampler
    )
    
    # Load obstacles in simulation
    sim.load_primitives(obstacles, color=[0.6, 0.6, 0.6, 1], visual_only=True)
    
    # Visualize target
    target_franka.marionette(target_pose)
    
    # Visualize point clouds from primitives
    obstacle_points = construct_mixed_point_cloud(obstacles, NUM_OBSTACLE_POINTS)
    target_points = gpu_fk_sampler.sample_end_effector(
        torch.as_tensor(target_pose.matrix).float().cuda(),
        num_points=NUM_TARGET_POINTS
    ).squeeze(0).cpu().numpy()
    
    point_cloud_colors = np.zeros((3, NUM_OBSTACLE_POINTS + NUM_TARGET_POINTS))
    point_cloud_colors[1, :NUM_OBSTACLE_POINTS] = 1  # Green for obstacles
    point_cloud_colors[0, NUM_OBSTACLE_POINTS:] = 1  # Red for target
    
    combined_points = np.vstack([obstacle_points[:, :3], target_points])
    
    viz["point_cloud"].set_object(
        meshcat.geometry.PointCloud(
            position=combined_points.T,
            color=point_cloud_colors,
            size=0.005,
        )
    )
    
    # Initialize robot and execute trajectory
    franka.marionette(trajectory[0])
    time.sleep(0.5)
    
    for q in tqdm(trajectory):
        franka.control_position(q)
        sim.step()
        sim_config, _ = franka.get_joint_states()
        
        # Update meshcat visualization
        for idx, (k, v) in enumerate(urdf.visual_trimesh_fk(sim_config[:8]).items()):
            viz[f"robot/{idx}"].set_transform(v)
        time.sleep(0.05)
    
    # Collision checking
    traj_tensor = torch.tensor(np.array(trajectory), dtype=torch.float32, device="cuda:0").unsqueeze(0)
    traj_flat = traj_tensor.reshape(-1, 7)

    cuboids_torch = TorchCuboids(
        torch.tensor(cuboid_centers).unsqueeze(0).cuda(),
        torch.tensor(cuboid_dims).unsqueeze(0).cuda(),
        torch.tensor(cuboid_quats).unsqueeze(0).cuda()
    )
    cylinders_torch = TorchCylinders(
        torch.tensor(cylinder_centers).unsqueeze(0).cuda(),
        torch.tensor(cylinder_radii).unsqueeze(0).cuda(),
        torch.tensor(cylinder_heights).unsqueeze(0).cuda(),
        torch.tensor(cylinder_quats).unsqueeze(0).cuda()
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

    print(f"Trajectory collision: {'YES' if has_collision else 'NO'}")
    
    # Extra time to view final pose
    for _ in range(20):
        sim.step()
        time.sleep(0.05)
    
    sim.clear_all_obstacles()