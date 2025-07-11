import torch
from torch import nn
import pytorch_lightning as pl


class PCEncoder(pl.LightningModule):
    """Encoder for Pointcloud using PyTorch Lightning."""

    def __init__(
        self,
        pc_latent_dim: int = 2048,
        use_layernorm: bool = False,
        final_norm: str = 'layernorm',
        use_projection: bool = True,
        **kwargs
    ):
        super().__init__()

        in_channels = 4  # xyzf: x, y, z, feature
        block_channel = [64, 128, 256]

        self.mlp = nn.Sequential(
            nn.Linear(in_channels, block_channel[0]),
            nn.LayerNorm(block_channel[0]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[0], block_channel[1]),
            nn.LayerNorm(block_channel[1]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[1], block_channel[2]),
            nn.LayerNorm(block_channel[2]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
        )

        if final_norm == 'layernorm':
            self.final_projection = nn.Sequential(
                nn.Linear(block_channel[-1], pc_latent_dim),
                nn.LayerNorm(pc_latent_dim)
            )
        elif final_norm == 'none':
            self.final_projection = nn.Linear(block_channel[-1], pc_latent_dim)
        else:
            raise NotImplementedError(f"final_norm: {final_norm}")

        if not use_projection:
            self.final_projection = nn.Identity()

    def forward(self, x):
        """
        Args:
            x: (B, N, 4) input point cloud
        Returns:
            (B, latent_dim) global feature
        """
        x = self.mlp(x)
        x = torch.max(x, 1)[0]
        x = self.final_projection(x)
        return x
    

if __name__ == "__main__":
    # Example usage
    B, N = 8, 2048
    xyzf = torch.rand(B, N, 4)
    encoder = PCEncoder(pc_latent_dim=2048)
    
    # Show model architecture and summary and size
    print(encoder)
    total_params = sum(p.numel() for p in encoder.parameters())
    print(f"Total parameters: {total_params}")
    
    out = encoder(xyzf)
    print(f"Output shape: {out.shape}")  # Should be (B, latent_dim)