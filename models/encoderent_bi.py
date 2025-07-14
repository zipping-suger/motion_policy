import utils
import torch
from torch import nn
import pytorch_lightning as pl
from models.pn import PCEncoder  # Point Cloud Encoder
from models.pcn import PCNEncoder  # Point Cloud Encoder for the encoder model
from typing import Dict


def dice_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Computes the Dice loss for binary multi-label predictions.

    :param pred: Tensor of predicted probabilities (after sigmoid), shape [B, C]
    :param target: Tensor of ground truth binary labels, shape [B, C]
    :param eps: Small constant for numerical stability
    :return: Dice loss (scalar)
    """
    pred_flat = pred.view(pred.size(0), -1)
    target_flat = target.view(target.size(0), -1)
    intersection = (pred_flat * target_flat).sum(dim=1)
    sums = pred_flat.sum(dim=1) + target_flat.sum(dim=1)
    dice_score = (2 * intersection + eps) / (sums + eps)
    return 1 - dice_score.mean()


class EncoderNet(pl.LightningModule):

    def __init__(self, pc_latent_dim: int = 1024):
        """
        Constructs the model. Outputs per-link binary predictions.
        """
        super().__init__()
        self.point_cloud_encoder = PCNEncoder(pc_latent_dim)
        self.links_dist_decoder = nn.Sequential(
            nn.Linear(pc_latent_dim, 256),
            nn.LeakyReLU(inplace=True),
            nn.Linear(256, 7)
        )
        self.sigmoid = nn.Sigmoid()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        pc_encoding = self.point_cloud_encoder(xyz)
        logits = self.links_dist_decoder(pc_encoding)
        return self.sigmoid(logits)


class TrainingEncoderNet(EncoderNet):
    def __init__(
        self,
        pc_latent_dim: int,
        num_robot_points: int,
    ):
        super().__init__(pc_latent_dim)
        self.num_robot_points = num_robot_points
        self.fk_sampler = None
        self.validation_step_outputs = []

    def _compute_metrics(self, pred_probs: torch.Tensor, gt_labels: torch.Tensor, prefix: str):
        """
        Computes and logs per-link metrics: accuracy, precision, recall
        """
        # Binarize predictions at 0.5
        preds = (pred_probs >= 0.5).float()
        # True positives, false positives, false negatives
        tp = (preds * gt_labels).sum(dim=0)
        fp = (preds * (1 - gt_labels)).sum(dim=0)
        fn = ((1 - preds) * gt_labels).sum(dim=0)
        # Accuracy
        acc = (preds == gt_labels).float().mean(dim=0)
        # Precision, Recall with eps for stability
        eps = 1e-6
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        # Log per-link metrics
        for i in range(7):
            self.log(f"{prefix}_accuracy_link_{i}", acc[i])
            self.log(f"{prefix}_precision_link_{i}", precision[i])
            self.log(f"{prefix}_recall_link_{i}", recall[i])

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        xyz, q = batch["xyz"], batch["configuration"]
        pred_probs = self(xyz)

        # Unpack obstacles
        cuboid_centers = batch["cuboid_centers"]
        cuboid_dims    = batch["cuboid_dims"]
        cuboid_quats   = batch["cuboid_quats"]
        cylinder_centers = batch["cylinder_centers"]
        cylinder_radii   = batch["cylinder_radii"]
        cylinder_heights = batch["cylinder_heights"]
        cylinder_quats   = batch["cylinder_quats"]

        # Initialize FK sampler if needed
        if self.fk_sampler is None:
            self.fk_sampler = utils.FrankaSampler(self.device, use_cache=True)
        links_pc = self.fk_sampler.sample_per_link(q)
        gt_labels = utils.check_links_in_range(
            links_pc,
            cuboid_centers,
            cuboid_dims,
            cuboid_quats,
            cylinder_centers,
            cylinder_radii,
            cylinder_heights,
            cylinder_quats,
        )

        loss = dice_loss(pred_probs, gt_labels)
        self.log("train_loss", loss)
        # Compute and log metrics
        self._compute_metrics(pred_probs, gt_labels, prefix="train")
        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        xyz, q = batch["xyz"], batch["configuration"]
        pred_probs = self(xyz)

        # Unpack obstacles
        cuboid_centers = batch["cuboid_centers"]
        cuboid_dims    = batch["cuboid_dims"]
        cuboid_quats   = batch["cuboid_quats"]
        cylinder_centers = batch["cylinder_centers"]
        cylinder_radii   = batch["cylinder_radii"]
        cylinder_heights = batch["cylinder_heights"]
        cylinder_quats   = batch["cylinder_quats"]

        # Initialize FK sampler if needed
        if self.fk_sampler is None:
            self.fk_sampler = utils.FrankaSampler(self.device, use_cache=True)
        links_pc = self.fk_sampler.sample_per_link(q)
        gt_labels = utils.check_links_in_range(
            links_pc,
            cuboid_centers,
            cuboid_dims,
            cuboid_quats,
            cylinder_centers,
            cylinder_radii,
            cylinder_heights,
            cylinder_quats,
        )

        val_loss = dice_loss(pred_probs, gt_labels)
        self.validation_step_outputs.append(val_loss)
        # Log metrics
        self._compute_metrics(pred_probs, gt_labels, prefix="val")
        return val_loss

    def on_validation_epoch_end(self):
        if not self.validation_step_outputs:
            return
        avg_val_loss = torch.stack(self.validation_step_outputs).mean()
        self.log("val_loss", avg_val_loss)
        self.validation_step_outputs.clear()

    def sample(self, q: torch.Tensor) -> torch.Tensor:
        assert self.fk_sampler is not None, "FK sampler not initialized"
        return self.fk_sampler.sample(q, self.num_robot_points)
