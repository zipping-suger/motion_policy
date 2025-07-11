import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from models.encoderent import TrainingEncoderNet
from data_loader import DataModule  # Using your existing DataModule
import utils
from tqdm import tqdm
import numpy as np
from typing import Dict, List
import matplotlib.pyplot as plt
import os
from pathlib import Path

NUM_ROBOT_POINTS = 2048  # Number of points to sample per robot link
NUM_OBSTACLE_POINTS = 4096
NUM_TARGET_POINTS = 128
RANDOM_SCALE = 0

def run_eval_encoder(
    checkpoint_path: str,
    test_data_path: str,
    pc_latent_dim: int = 2048,
    num_robot_points: int = 2048,
    batch_size: int = 1,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    save_plots: bool = True,
    output_dir: str = "./eval_results"
):
    """
    Evaluate a trained encoder model on test data with enhanced metrics and visualization
    
    Args:
        checkpoint_path: Path to trained model checkpoint
        test_data_path: Path to test dataset
        pc_latent_dim: Latent dimension of point cloud encoder
        num_robot_points: Number of points to sample per robot link
        batch_size: Evaluation batch size
        device: Device to run evaluation on
        save_plots: Whether to save error distribution plots
        output_dir: Directory to save evaluation results
    """
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 1. Load model from checkpoint
    model = TrainingEncoderNet.load_from_checkpoint(
        checkpoint_path,
        pc_latent_dim=pc_latent_dim,
        num_robot_points=num_robot_points
    )
    model.to(device)
    model.eval()
    
    # 2. Prepare dataset using existing DataModule
    dm = DataModule(
        batch_size=batch_size,
        train_mode='finetune_tasks',
        data_dir=test_data_path,
        trajectory_key="global_solutions",
        num_robot_points=NUM_ROBOT_POINTS,
        num_obstacle_points=NUM_OBSTACLE_POINTS,
        num_target_points=NUM_TARGET_POINTS,
        random_scale=RANDOM_SCALE
    )
    dm.setup(stage='fit')
    test_loader = dm.val_dataloader()
    
    fk_sampler = utils.FrankaSampler(device, use_cache=True)
    
    # 3. Initialize metrics
    metrics = {
        'total_mse': 0.0,
        'link_mae': [0.0] * 7,
        'link_mse': [0.0] * 7,
        'link_max_error': [0.0] * 7,
        'all_preds': [],
        'all_targets': [],
        'num_batches': 0
    }
    
    # 4. Run evaluation
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Evaluating")):
            # Transfer data to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Forward pass
            pred_dists = model(batch["xyz"])

            # Compute ground truth distances
            links_pc = fk_sampler.sample_per_link(batch["configuration"])
            gt_dists = utils.links_obs_dist(
                links_pc,
                batch["cuboid_centers"],
                batch["cuboid_dims"],
                batch["cuboid_quats"],
                batch["cylinder_centers"],
                batch["cylinder_radii"],
                batch["cylinder_heights"],
                batch["cylinder_quats"]
            )
            
            # Store predictions and targets for overall analysis
            metrics['all_preds'].append(pred_dists.cpu().numpy())
            metrics['all_targets'].append(gt_dists.cpu().numpy())
            
            # Compute batch metrics
            batch_mse = torch.nn.functional.mse_loss(pred_dists, gt_dists)
            metrics['total_mse'] += batch_mse.item()
            
            # Compute per-link metrics
            errors = pred_dists - gt_dists
            abs_errors = torch.abs(errors)
            
            for i in range(7):
                metrics['link_mae'][i] += abs_errors[:, i].mean().item()
                metrics['link_mse'][i] += torch.mean(errors[:, i]**2).item()
                metrics['link_max_error'][i] = max(
                    metrics['link_max_error'][i], 
                    abs_errors[:, i].max().item()
                )
            
            metrics['num_batches'] += 1
    
    # 5. Calculate final metrics
    metrics['avg_mse'] = metrics['total_mse'] / metrics['num_batches']
    metrics['avg_mae'] = sum(metrics['link_mae']) / (7 * metrics['num_batches'])
    
    # Concatenate all predictions and targets
    all_preds = np.concatenate(metrics['all_preds'])
    all_targets = np.concatenate(metrics['all_targets'])
    
    # 6. Generate and save plots if requested
    if save_plots:
        # Error distribution plot
        plt.figure(figsize=(12, 6))
        for i in range(7):
            errors = all_preds[:, i] - all_targets[:, i]
            plt.hist(errors, bins=50, alpha=0.5, label=f'Link {i}')
        plt.xlabel('Prediction Error')
        plt.ylabel('Frequency')
        plt.title('Error Distribution by Link')
        plt.legend()
        plt.savefig(os.path.join(output_dir, 'error_distribution.png'))
        plt.close()
        
        # Scatter plot of predictions vs targets
        plt.figure(figsize=(10, 10))
        plt.scatter(all_targets.flatten(), all_preds.flatten(), alpha=0.3)
        plt.plot([0, all_targets.max()], [0, all_targets.max()], 'r--')
        plt.xlabel('Ground Truth Distance')
        plt.ylabel('Predicted Distance')
        plt.title('Predictions vs Ground Truth')
        plt.savefig(os.path.join(output_dir, 'predictions_vs_truth.png'))
        plt.close()
    
    # 7. Print comprehensive results
    print(f"\n{'='*40}")
    print(f"{'Encoder Evaluation Results':^40}")
    print(f"{'='*40}")
    print(f"\nOverall Metrics:")
    print(f"  Average MSE: {metrics['avg_mse']:.6f}")
    print(f"  Average MAE: {metrics['avg_mae']:.6f}")
    
    print("\nPer-link Metrics:")
    print(f"{'Link':<8}{'MAE':<12}{'MSE':<12}{'Max Error':<12}")
    for i in range(7):
        print(f"{i:<8}"
              f"{metrics['link_mae'][i]/metrics['num_batches']:.6f}\t"
              f"{metrics['link_mse'][i]/metrics['num_batches']:.6f}\t"
              f"{metrics['link_max_error'][i]:.6f}")
    
    # Save metrics to file
    metrics_file = os.path.join(output_dir, 'metrics.txt')
    with open(metrics_file, 'w') as f:
        f.write(f"Average MSE: {metrics['avg_mse']:.6f}\n")
        f.write(f"Average MAE: {metrics['avg_mae']:.6f}\n\n")
        f.write("Per-link Metrics:\n")
        f.write(f"{'Link':<8}{'MAE':<12}{'MSE':<12}{'Max Error':<12}\n")
        for i in range(7):
            f.write(f"{i:<8}"
                   f"{metrics['link_mae'][i]/metrics['num_batches']:.6f}\t"
                   f"{metrics['link_mse'][i]/metrics['num_batches']:.6f}\t"
                   f"{metrics['link_max_error'][i]:.6f}\n")
    
    print(f"\nResults saved to: {output_dir}")
    
    return metrics

if __name__ == "__main__":
    # Example usage with your checkpoint path
    results = run_eval_encoder(
        checkpoint_path="./checkpoints/zvsf9h69/last.ckpt",
        test_data_path="./pretrain_data/single_cubby_tasks_135k",
        batch_size=1,
        save_plots=True,
        output_dir="./encoder_eval_results"
    )