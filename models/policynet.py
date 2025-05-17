import torch
import torch.distributions as D
torch.set_float32_matmul_precision('medium')
from torch import nn
import pytorch_lightning as pl
from models.pcn import PCNEncoder
from typing import List, Tuple, Sequence, Dict, Callable
import utils
from utils import collision_loss
from geometry import TorchCuboids, TorchCylinders
from robofin.pointcloud.torch import FrankaSampler, FrankaCollisionSampler

class PolicyNet(pl.LightningModule):
    def __init__(self, pc_latent_dim=2048, num_gaussians=5):
        super().__init__()
        self.point_cloud_encoder = PCNEncoder(pc_latent_dim)
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
        
        # MDN Head: Predicts means (mu), variances (sigma), and mixture coefficients (pi)
        self.mdn_head = nn.Sequential(
            nn.Linear(pc_latent_dim + 64 + 64, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
        )
        
        # Output layers for MDN parameters
        self.mu_layer = nn.Linear(128, 7 * num_gaussians)  # Mean for each Gaussian
        self.sigma_layer = nn.Linear(128, 7 * num_gaussians)  # Diagonal covariance
        self.pi_layer = nn.Linear(128, num_gaussians)  # Mixture weights

        self.num_gaussians = num_gaussians

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)

    def forward(self, xyz: torch.Tensor, q: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pc_encoding = self.point_cloud_encoder(xyz)
        config_encoding = self.config_encoder(q)
        target_encoding = self.target_encoder(target)
        x = torch.cat((pc_encoding, config_encoding, target_encoding), dim=1)
        x = self.mdn_head(x)
        
        # Predict MDN parameters
        mu = self.mu_layer(x).view(-1, self.num_gaussians, 7)  # Shape: [B, K, 7]
        sigma = torch.exp(self.sigma_layer(x)).view(-1, self.num_gaussians, 7)  # Positive variance
        pi = torch.softmax(self.pi_layer(x), dim=-1)  # Shape: [B, K]
        
        return mu, sigma, pi

    def sample_action(self, mu: torch.Tensor, sigma: torch.Tensor, pi: torch.Tensor) -> torch.Tensor:
        """Sample an action from the mixture distribution."""
        # Choose a component
        comp = D.Categorical(pi).sample()
        # Sample from the selected Gaussian
        batch_size = mu.shape[0]
        mu_selected = mu[torch.arange(batch_size), comp]
        sigma_selected = sigma[torch.arange(batch_size), comp]
        return mu_selected + torch.randn_like(mu_selected) * sigma_selected


class TrainingPolicyNet(PolicyNet):
    def __init__(
        self,
        pc_latent_dim: int,
        num_robot_points: int,
        bc_loss_weight: float,
        collision_loss_weight: float,
        num_gaussians: int = 5,
    ):
        super().__init__(pc_latent_dim, num_gaussians)
        self.num_robot_points = num_robot_points
        self.fk_sampler = None
        self.collision_sampler = None
        self.bc_loss_weight = bc_loss_weight
        self.collision_loss_weight = collision_loss_weight
        self.validation_step_outputs = []

    def mdn_loss(self, mu: torch.Tensor, sigma: torch.Tensor, pi: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Negative log-likelihood loss for MDN."""
        # Create a mixture distribution
        mix = D.Categorical(pi)
        comp = D.Independent(D.Normal(mu, sigma), 1)
        mixture = D.MixtureSameFamily(mix, comp)
        # Calculate NLL
        return -mixture.log_prob(target).mean()

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        xyz, q, target = batch["xyz"], batch["configuration"], batch["target_pose"]
        supervision = batch["supervision"]
        
        mu, sigma, pi = self(xyz, q, target)
        bc_loss = self.mdn_loss(mu, sigma, pi, supervision)
        
        self.log("bc_loss", bc_loss)
        val_loss = self.bc_loss_weight * bc_loss
        self.log("val_loss", val_loss)
        return val_loss

    def rollout(self, batch: Dict[str, torch.Tensor], rollout_length: int, sampler: Callable[[torch.Tensor], torch.Tensor]) -> List[torch.Tensor]:
        xyz, q, target = batch["xyz"], batch["configuration"], batch["target_pose"]
        if q.ndim == 1:
            xyz = xyz.unsqueeze(0)
            q = q.unsqueeze(0)

        trajectory = [q]
        for _ in range(rollout_length):
            mu, sigma, pi = self(xyz, q, target)
            delta_q = self.sample_action(mu, sigma, pi)
            q = q + delta_q
            trajectory.append(q)
            robot_samples = sampler(q).type_as(xyz)
            xyz[:, : self.num_robot_points, :3] = robot_samples

        return trajectory