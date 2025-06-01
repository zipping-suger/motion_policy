import torch
from pathlib import Path
from tqdm import tqdm
import argparse
from models.policynet import PolicyNet
from data_loader import PointCloudTrajectoryDataset, DatasetType
from robofin.pointcloud.torch import FrankaSampler, FrankaCollisionSampler
from geometry import TorchCuboids, TorchCylinders

# Constants
NUM_ROBOT_POINTS = 2048
NUM_OBSTACLE_POINTS = 4096
MAX_STEPS = 100  
GOAL_THRESHOLD = 0.05  # 5cm threshold

def run_eval(model_path: str, val_data_path: str, num_val: int = 100) -> None:
    """Evaluate the model on the validation dataset."""
    # Load model
    model = PolicyNet.load_from_checkpoint(model_path).cuda()
    model.eval()

    # Load dataset
    dataset = PointCloudTrajectoryDataset(
        Path(val_data_path),
        "global_solutions",
        NUM_ROBOT_POINTS,
        NUM_OBSTACLE_POINTS,
        DatasetType.VAL
    )

    # Initialize samplers
    fk_sampler = FrankaSampler("cuda:0", use_cache=True)
    collision_sampler = FrankaCollisionSampler("cuda:0", with_base_link=False)

    # Metrics
    total_samples = len(dataset) if num_val is None else min(num_val, len(dataset))
    total_goals_reached = 0
    total_collisions = 0
    total_steps = 0
    total_ee_error = 0.0
    num_failed = 0

    for problem_idx in tqdm(range(total_samples), desc="Evaluating"):
        data = dataset[problem_idx]
        # Move data to GPU
        for key in data:
            if isinstance(data[key], torch.Tensor):
                data[key] = data[key].cuda()

        # Prepare inputs with batch dimension
        xyz = data["xyz"].unsqueeze(0)
        q = data["configuration"].unsqueeze(0)
        target = data["target_configuration"].unsqueeze(0)
        # target = data["target_pose"].unsqueeze(0)  # Use target pose instead of configuration
        target_position = data["target_position"].unsqueeze(0)

        # Perform rollout
        trajectory = []
        current_q = q.clone()
        goal_reached = False
        steps_taken = 0

        for step in range(MAX_STEPS):
            with torch.no_grad():
                delta_q = model(xyz, current_q, target)
            current_q += delta_q
            trajectory.append(current_q.squeeze(0))

            # Update point cloud with new robot points
            robot_points = fk_sampler.sample(current_q, NUM_ROBOT_POINTS)
            xyz[:, :NUM_ROBOT_POINTS, :3] = robot_points

            # Check end-effector distance
            ee_pose = fk_sampler.end_effector_pose(current_q)
            current_pos = ee_pose[:, :3, -1]
            distance = torch.norm(current_pos - target_position, dim=1).item()

            if distance < GOAL_THRESHOLD:
                goal_reached = True
                steps_taken = step + 1  # Steps are 0-indexed
                break

        # If goal not reached, use all steps
        if not goal_reached:
            steps_taken = MAX_STEPS
            # Compute final end effector error
            ee_pose = fk_sampler.end_effector_pose(current_q)
            current_pos = ee_pose[:, :3, -1]
            ee_error = torch.norm(current_pos - target_position, dim=1).item()
            total_ee_error += ee_error
            num_failed += 1

        # Collision checking
        traj_tensor = torch.stack(trajectory).unsqueeze(0)  # (1, T, 7)
        traj_flat = traj_tensor.reshape(-1, 7)  # (T, 7)

        # Extract obstacle data with batch dimension
        cuboids = TorchCuboids(
            data["cuboid_centers"].unsqueeze(0),
            data["cuboid_dims"].unsqueeze(0),
            data["cuboid_quats"].unsqueeze(0)
        )
        cylinders = TorchCylinders(
            data["cylinder_centers"].unsqueeze(0),
            data["cylinder_radii"].unsqueeze(0),
            data["cylinder_heights"].unsqueeze(0),
            data["cylinder_quats"].unsqueeze(0)
        )

        # Compute collision spheres for all configurations
        collision_spheres = collision_sampler.compute_spheres(traj_flat)
        has_collision = False

        for radius, spheres in collision_spheres:
            num_spheres = spheres.size(-2)
            spheres_reshaped = spheres.view(1, -1, num_spheres, 3)  # (1, T, S, 3)
            
            # Compute SDF values
            sdf_cuboids = cuboids.sdf_sequence(spheres_reshaped)
            sdf_cylinders = cylinders.sdf_sequence(spheres_reshaped)
            sdf_values = torch.minimum(sdf_cuboids, sdf_cylinders)
            
            # Check collisions
            collision_mask = sdf_values <= radius
            if torch.any(collision_mask):
                has_collision = True
                break

        # Update metrics
        total_goals_reached += int(goal_reached)
        total_collisions += int(has_collision)
        if goal_reached:
            total_steps += steps_taken

    # Calculate final metrics
    goal_reach_rate = total_goals_reached / total_samples
    collision_rate = total_collisions / total_samples
    avg_steps = total_steps / total_goals_reached if total_goals_reached > 0 else 0
    avg_ee_error = total_ee_error / num_failed if num_failed > 0 else 0

    print("\n=== Evaluation Results ===")
    print(f"Goal Reach Rate: {goal_reach_rate * 100:.2f}%")
    print(f"Collision Rate: {collision_rate * 100:.2f}%")
    print(f"Avg Steps (Successful): {avg_steps:.1f}/{MAX_STEPS}")
    print(f"Avg End Effector Error (Failed): {avg_ee_error:.4f} m")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate PolicyNet on validation data.")
    parser.add_argument("--model_path", type=str, default="./checkpoints/table_6k/last.ckpt", help="Path to the trained model checkpoint.")
    parser.add_argument("--val_data_path", type=str, default="./pretrain_data/ompl_table_30k", help="Path to the validation dataset directory.")
    args = parser.parse_args()
    run_eval(args.model_path, args.val_data_path)