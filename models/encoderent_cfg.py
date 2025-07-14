import utils
import torch
from torch import nn
import pytorch_lightning as pl
from models.pcn import PCNEncoder  
from models.pn import PCEncoder
from models.pointnet2 import PointNet2
from typing import List, Tuple, Sequence, Dict, Callable
from utils import collision_loss, links_obs_dist, minimal_collision_distance
from geometry import TorchCuboids, TorchCylinders
from robofin.robots import FrankaRealRobot


class EncoderNet(pl.LightningModule):

    def __init__(self, pc_latent_dim=2048): 
        """
        Constructs the model
        """
        super().__init__()
        self.point_cloud_encoder = PCNEncoder(pc_latent_dim)  # Point Completion Network Encoder
        # self.point_cloud_encoder = PCEncoder(pc_latent_dim)  # Point Cloud Encoder
        # self.point_cloud_encoder = PointNet2()  # PointNet2 Encoder
        
        self.links_dist_decoder = nn.Sequential(
            nn.Linear(pc_latent_dim, 256),
            nn.LeakyReLU(inplace=True),
            nn.Linear(256, 7),  # Output for delta q
        )

    def configure_optimizers(self):
        """
        A standard method in PyTorch lightning to set the optimizer
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        pc_encoding = self.point_cloud_encoder(xyz.float())  # xyz: (B, N, 4), last dimension is feature 0 for robot, 1 for obstacle
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
        
        pred_q = self(xyz)
                
        train_loss = self.loss_fun(pred_q, q)
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
            pred_q = self(xyz)
                        
            # Calculate validation loss
            val_loss = self.loss_fun(pred_q, q)
            self.validation_step_outputs.append(val_loss)
            return val_loss

    def on_validation_epoch_end(self):        
        # Calculate average validation loss
        val_losses = torch.stack(self.validation_step_outputs)
        avg_val_loss = val_losses.mean()
        
        # Log both training and validation losses
        self.log("val_loss", avg_val_loss)
        
        self.validation_step_outputs.clear()
