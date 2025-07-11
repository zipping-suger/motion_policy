import utils
import torch
from torch import nn
import pytorch_lightning as pl
from models.pcn import PCNEncoder  # Importing the Point Cloud Encoder
from models.pn import PCEncoder
from typing import List, Tuple, Sequence, Dict, Callable
from utils import collision_loss, links_obs_dist, minimal_collision_distance
from geometry import TorchCuboids, TorchCylinders
from robofin.robots import FrankaRealRobot


class EncoderNet(pl.LightningModule):

    def __init__(self, pc_latent_dim=1024): 
        """
        Constructs the model
        """
        super().__init__()
        self.point_cloud_encoder = PCNEncoder(pc_latent_dim)  # Point Cloud Network
        
        self.links_dist_decoder = nn.Sequential(
            nn.Linear(pc_latent_dim, 256),
            nn.LeakyReLU(inplace=True),
            nn.Linear(256, 1),  # Output for delta q
        )

    def configure_optimizers(self):
        """
        A standard method in PyTorch lightning to set the optimizer
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        pc_encoding = self.point_cloud_encoder(xyz)
        return self.links_dist_decoder(pc_encoding)   


class TrainingEncoderNet(EncoderNet):
    def __init__(
        self,
        pc_latent_dim: int,
        num_robot_points: int,
    ):
        super().__init__(pc_latent_dim)
        self.num_robot_points = num_robot_points
        self.fk_sampler = None
        self.collision_sampler = None
        self.loss_fun = nn.MSELoss()
        self.validation_step_outputs = []

    def training_step(  # type: ignore[override]
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        xyz, q = (
            batch["xyz"],
            batch["configuration"],
        )
        
        pred_links_dist = self(xyz)
        (
            cuboid_centers,
            cuboid_dims,
            cuboid_quats,
            cylinder_centers,
            cylinder_radii,
            cylinder_heights,
            cylinder_quats,
        ) = (
            batch["cuboid_centers"],
            batch["cuboid_dims"],
            batch["cuboid_quats"],
            batch["cylinder_centers"],
            batch["cylinder_radii"],
            batch["cylinder_heights"],
            batch["cylinder_quats"],
        )
        
        if self.fk_sampler is None:
            self.fk_sampler = utils.FrankaSampler(self.device, use_cache=True)
        
        # links_pc = self.fk_sampler.sample_per_link(q)
        # ground_truth_links_dist = links_obs_dist(
        #     links_pc,
        #     cuboid_centers,
        #     cuboid_dims,
        #     cuboid_quats,
        #     cylinder_centers,
        #     cylinder_radii,
        #     cylinder_heights,
        #     cylinder_quats,
        # )
        
        input_pc = self.fk_sampler.sample(q, self.num_robot_points)
        ground_truth_min_dist = minimal_collision_distance(
            input_pc,
            cuboid_centers,
            cuboid_dims,
            cuboid_quats,
            cylinder_centers,
            cylinder_radii,
            cylinder_heights,
            cylinder_quats,
        )
        
        # train_loss = self.loss_fun(pred_links_dist, ground_truth_min_dist)
        train_loss = self.loss_fun(pred_links_dist.squeeze(-1), ground_truth_min_dist)
        self.log("train_loss", train_loss)
        return train_loss

    def sample(self, q: torch.Tensor) -> torch.Tensor:
        """
        Samples a point cloud from the surface of all the robot's links

        :param q torch.Tensor: Batched configuration in joint space
        :rtype torch.Tensor: Batched point cloud of size [B, self.num_robot_points, 3]
        """
        assert self.fk_sampler is not None
        return self.fk_sampler.sample(q, self.num_robot_points)

    def validation_step(  # type: ignore[override]
        self, batch: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        with torch.no_grad():
            # Unpack batch data
            xyz, q = (
                batch["xyz"],
                batch["configuration"],
            )
            
            # Get model predictions
            pred_links_dist = self(xyz)
            
            # Extract obstacle parameters
            (
                cuboid_centers,
                cuboid_dims,
                cuboid_quats,
                cylinder_centers,
                cylinder_radii,
                cylinder_heights,
                cylinder_quats,
            ) = (
                batch["cuboid_centers"],
                batch["cuboid_dims"],
                batch["cuboid_quats"],
                batch["cylinder_centers"],
                batch["cylinder_radii"],
                batch["cylinder_heights"],
                batch["cylinder_quats"],
            )
            
            # Initialize FK sampler if needed
            if self.fk_sampler is None:
                self.fk_sampler = utils.FrankaSampler(self.device, use_cache=True)
            
            # # Calculate ground truth distances
            # links_pc = self.fk_sampler.sample_per_link(q)
            # ground_truth_links_dist = links_obs_dist(
            #     links_pc,
            #     cuboid_centers,
            #     cuboid_dims,
            #     cuboid_quats,
            #     cylinder_centers,
            #     cylinder_radii,
            #     cylinder_heights,
            #     cylinder_quats,
            # )
            
            input_pc = self.fk_sampler.sample(q, self.num_robot_points)
            ground_truth_min_dist = minimal_collision_distance(
                input_pc,
                cuboid_centers,
                cuboid_dims,
                cuboid_quats,
                cylinder_centers,
                cylinder_radii,
                cylinder_heights,
                cylinder_quats,
            )
            
            # Calculate validation loss
            # val_loss = self.loss_fun(pred_links_dist, ground_truth_min_dist)
            val_loss = self.loss_fun(pred_links_dist.squeeze(-1), ground_truth_min_dist)
            self.validation_step_outputs.append(val_loss)
            return val_loss

    def on_validation_epoch_end(self):        
        # Calculate average validation loss
        val_losses = torch.stack(self.validation_step_outputs)
        avg_val_loss = val_losses.mean()
        
        # Log both training and validation losses
        self.log("val_loss", avg_val_loss)
        
        self.validation_step_outputs.clear()
