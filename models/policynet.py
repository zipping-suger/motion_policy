import torch
torch.set_float32_matmul_precision('medium')
from torch import nn
import pytorch_lightning as pl
from models.pcn import PCNEncoder
from models.ptv3 import PointTransformerNet
from typing import List, Tuple, Sequence, Dict, Callable
import utils
from utils import collision_loss
from geometry import TorchCuboids, TorchCylinders
from robofin.pointcloud.torch import FrankaSampler, FrankaCollisionSampler

class PolicyNet(pl.LightningModule):
    """
    The architecture laid out here is the default architecture laid out in the
    Motion Policy Networks paper (Fishman, et. al, 2022).
    """
    def __init__(self, pc_latent_dim=2048): 
        """
        Constructs the model
        """
        super().__init__()
        self.point_cloud_encoder = PCNEncoder(pc_latent_dim)  # Point Cloud Network
        # self.point_cloud_encoder = PointTransformerNet(feature_dim=pc_latent_dim) # Point Transformer V3
        
        self.config_encoder = nn.Sequential(
            nn.Linear(7, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 64),
        )
        
        self.target_encoder = nn.Sequential(
            nn.Linear(7, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 64),
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(pc_latent_dim + 64 + 64, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Linear(128,7)
        )

    def configure_optimizers(self):
        """
        A standard method in PyTorch lightning to set the optimizer
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer

    def forward(self, xyz: torch.Tensor, q: torch.Tensor, target:torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        pc_encoding = self.point_cloud_encoder(xyz)
        config_encoding = self.config_encoder(q)
        target_encoding = self.target_encoder(target)
        x = torch.cat((pc_encoding, config_encoding, target_encoding), dim=1)
        return self.decoder(x)
    
    
class TrainingPolicyNet(PolicyNet):
    def __init__(
        self,
        pc_latent_dim: int,
        num_robot_points: int,
        bc_loss_weight: float,
        collision_loss_weight: float,
    ):
        """
        Creates the network and assigns additional parameters for training


        :param num_robot_points int: The number of robot points used when resampling
                                     the robot points during rollouts (used in validation)
        :rtype Self: An instance of the network
        """
        super().__init__(pc_latent_dim)
        self.num_robot_points = num_robot_points
        self.fk_sampler = None
        self.collision_sampler = None
        self.bc_loss_weight = bc_loss_weight
        self.collision_loss_weight = collision_loss_weight
        self.loss_fun = nn.MSELoss()
        self.validation_step_outputs = []

    def rollout(
        self,
        batch: Dict[str, torch.Tensor],
        rollout_length: int,
        sampler: Callable[[torch.Tensor], torch.Tensor],
    ) -> List[torch.Tensor]:
        xyz, q, target = (
            batch["xyz"],
            batch["configuration"],
            batch["target_configuration"],
            # batch["target_pose"],
        )
        
        # This block is to adapt for the case where we only want to roll out a
        # single trajectory
        if q.ndim == 1:
            xyz = xyz.unsqueeze(0)
            q = q.unsqueeze(0)

        trajectory = [q]

        for i in range(rollout_length):
            q = q + self(xyz, q, target)

            trajectory.append(q)

            robot_samples = sampler(q).type_as(xyz)
            # replace the fist num_robot_points points in the point cloud
            # with the robot samples
            xyz[:, : self.num_robot_points, :3] = robot_samples
            

        return trajectory

    def training_step(  # type: ignore[override]
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """
        A function called automatically by Pytorch Lightning during training.
        This function handles the forward pass, the loss calculation, and what to log

        :param batch Dict[str, torch.Tensor]: A data batch coming from the
                                                   data loader--should already be
                                                   on the correct device
        :param batch_idx int: The index of the batch (not used by this function)
        :rtype torch.Tensor: The overall weighted loss (used for backprop)
        """
        xyz, q, target = (
            batch["xyz"],
            batch["configuration"],
            batch["target_configuration"],
            # batch["target_pose"],
        )
        
        delta_q = self(xyz, q, target)
        (
            cuboid_centers,
            cuboid_dims,
            cuboid_quats,
            cylinder_centers,
            cylinder_radii,
            cylinder_heights,
            cylinder_quats,
            supervision,
        ) = (
            batch["cuboid_centers"],
            batch["cuboid_dims"],
            batch["cuboid_quats"],
            batch["cylinder_centers"],
            batch["cylinder_radii"],
            batch["cylinder_heights"],
            batch["cylinder_quats"],
            batch["supervision"],
        )
        
        
        bc_loss = self.loss_fun(delta_q, supervision)
        # if self.fk_sampler is None:
        #     self.fk_sampler = FrankaSampler(self.device, use_cache=True)
        # input_pc = self.fk_sampler.sample(
        #     q + delta_q, self.num_robot_points
        # )
        # colli_los = collision_loss(
        #     input_pc,
        #     cuboid_centers,
        #     cuboid_dims,
        #     cuboid_quats,
        #     cylinder_centers,
        #     cylinder_radii,
        #     cylinder_heights,
        #     cylinder_quats,
        # )
        self.log("bc_loss", bc_loss)
        # self.log("collision_loss", colli_los)
        val_loss = (
            self.bc_loss_weight * bc_loss
            # + self.collision_loss_weight * colli_los
        )
        self.log("val_loss", val_loss)
        return val_loss

    def sample(self, q: torch.Tensor) -> torch.Tensor:
        """
        Samples a point cloud from the surface of all the robot's links

        :param q torch.Tensor: Batched configuration in joint space
        :rtype torch.Tensor: Batched point cloud of size [B, self.num_robot_points, 3]
        """
        assert self.fk_sampler is not None
        return self.fk_sampler.sample(q, self.num_robot_points)

    def validation_step(  # type: ignore[override]
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """
        This is a Pytorch Lightning function run automatically across devices
        during the validation loop

        :param batch Dict[str, torch.Tensor]: The batch coming from the dataloader
        :param batch_idx int: The index of the batch (not used by this function)
        :rtype torch.Tensor: The loss values which are to be collected into summary stats
        """

        # These are defined here because they need to be set on the correct devices.
        # The easiest way to do this is to do it at call-time
        if self.fk_sampler is None:
            self.fk_sampler = FrankaSampler(self.device, use_cache=True)
        if self.collision_sampler is None:
            self.collision_sampler = FrankaCollisionSampler(
                self.device, with_base_link=False
            )
        rollout = self.rollout(batch, 54, self.sample)

        assert self.fk_sampler is not None  # Necessary for mypy to type properly
        
        eff = self.fk_sampler.end_effector_pose(rollout[-1])
        position_error = torch.linalg.vector_norm(
            eff[:, :3, -1] - batch["target_position"], dim=1
        )
        avg_target_error = torch.mean(position_error)

        cuboids = TorchCuboids(
            batch["cuboid_centers"],
            batch["cuboid_dims"],
            batch["cuboid_quats"],
        )
        cylinders = TorchCylinders(
            batch["cylinder_centers"],
            batch["cylinder_radii"],
            batch["cylinder_heights"],
            batch["cylinder_quats"],
        )

        B = batch["cuboid_centers"].size(0)
        rollout = torch.stack(rollout, dim=1)
        # Here is some Pytorch broadcasting voodoo to calculate whether each
        # rollout has a collision or not (looking to calculate the collision rate)
        assert rollout.shape == (B, 55, 7)
        rollout = rollout.reshape(-1, 7)
        has_collision = torch.zeros(B, dtype=torch.bool, device=self.device)
        collision_spheres = self.collision_sampler.compute_spheres(rollout)
        for radius, spheres in collision_spheres:
            num_spheres = spheres.shape[-2]
            sphere_sequence = spheres.reshape((B, -1, num_spheres, 3))
            sdf_values = torch.minimum(
                cuboids.sdf_sequence(sphere_sequence),
                cylinders.sdf_sequence(sphere_sequence),
            )
            assert sdf_values.shape == (B, 55, num_spheres)
            radius_collisions = torch.any(
                sdf_values.reshape((sdf_values.size(0), -1)) <= radius, dim=-1
            )
            has_collision = torch.logical_or(radius_collisions, has_collision)

        avg_collision_rate = torch.count_nonzero(has_collision) / B
        result = {
            "avg_target_error": avg_target_error,
            "avg_collision_rate": avg_collision_rate,
        }
        self.validation_step_outputs.append(result)
        return result

    def on_validation_epoch_end(self):
        avg_target_error = torch.mean(
            torch.stack([x["avg_target_error"] for x in self.validation_step_outputs])
        )
        self.log("avg_target_error", avg_target_error)

        avg_collision_rate = torch.mean(
            torch.stack([x["avg_collision_rate"] for x in self.validation_step_outputs])
        )
        self.log("avg_collision_rate", avg_collision_rate)
        self.validation_step_outputs.clear()
