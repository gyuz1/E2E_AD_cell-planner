"""Full-resolution BEV latent world model for perception-based LAW.

This follows the perception-based LAW description:
  1. flatten the full BEV feature map to K = H_bev * W_bev latent vectors;
  2. concatenate the single predicted ego waypoint sequence to every latent;
  3. map the concatenated vectors back to D channels with an MLP;
  4. predict the future full-resolution BEV latent with stacked deformable
     self-attention + FFN blocks.

No spatial pooling is applied.
"""


import torch
import torch.nn as nn

from projects.mmdet3d_plugin.VAD.modules.decoder import (
    CustomMSDeformableAttention,
)


class DeformableSelfAttentionBlock(nn.Module):
    """One deformable self-attention and feed-forward block."""

    def __init__(
        self,
        embed_dims: int = 256,
        num_heads: int = 8,
        num_points: int = 4,
        ffn_dims: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.attn = CustomMSDeformableAttention(
            embed_dims=embed_dims,
            num_heads=num_heads,
            num_levels=1,
            num_points=num_points,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(embed_dims)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dims, ffn_dims),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(ffn_dims, embed_dims),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(embed_dims)

    def forward(
        self,
        x: torch.Tensor,
        reference_points: torch.Tensor,
        spatial_shapes: torch.Tensor,
        level_start_index: torch.Tensor,
    ) -> torch.Tensor:
        # CustomMSDeformableAttention already adds the identity residual.
        x = self.attn(
            query=x,
            key=x,
            value=x,
            identity=x,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
        )
        x = self.norm1(x)
        x = self.norm2(x + self.ffn(x))
        return x


class BEVLatentWorldModel(nn.Module):
    """Predict the full future BEV latent from current BEV and ego waypoints.

    Args:
        embed_dims: BEV feature dimension D.
        num_waypoints: Number M of ego future waypoints.
        bev_h: BEV grid height.
        bev_w: BEV grid width.
        num_layers: Number of deformable Transformer blocks.
        num_heads: Number of attention heads.
        num_points: Sampling points per head in deformable attention.
        ffn_dims: Feed-forward hidden dimension.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        embed_dims: int = 256,
        num_waypoints: int = 6,
        bev_h: int = 100,
        bev_w: int = 100,
        num_layers: int = 2,
        num_heads: int = 8,
        num_points: int = 4,
        ffn_dims: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if min(embed_dims, num_waypoints, bev_h, bev_w, num_layers) <= 0:
            raise ValueError("All dimensions and num_layers must be positive.")

        self.embed_dims = embed_dims
        self.num_waypoints = num_waypoints
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.num_tokens = bev_h * bev_w

        self.action_encoder = nn.Sequential(
            nn.Linear(embed_dims + 2 * num_waypoints, embed_dims),
            nn.LayerNorm(embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dims, embed_dims),
        )
        self.blocks = nn.ModuleList([
            DeformableSelfAttentionBlock(
                embed_dims=embed_dims,
                num_heads=num_heads,
                num_points=num_points,
                ffn_dims=ffn_dims,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        reference_points = self._make_reference_points(bev_h, bev_w)
        self.register_buffer(
            "reference_points",
            reference_points,
            persistent=False,
        )
        self.register_buffer(
            "spatial_shapes",
            torch.tensor([[bev_h, bev_w]], dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "level_start_index",
            torch.tensor([0], dtype=torch.long),
            persistent=False,
        )

    @staticmethod
    def _make_reference_points(bev_h: int, bev_w: int) -> torch.Tensor:
        """Create normalized BEV cell-center coordinates [1, K, 1, 2]."""
        try:
            grid_y, grid_x = torch.meshgrid(
                torch.arange(bev_h, dtype=torch.float32),
                torch.arange(bev_w, dtype=torch.float32),
                indexing="ij",
            )
        except TypeError:  # torch 1.9 compatibility
            grid_y, grid_x = torch.meshgrid(
                torch.arange(bev_h, dtype=torch.float32),
                torch.arange(bev_w, dtype=torch.float32),
            )
        ref_x = (grid_x.reshape(-1) + 0.5) / float(bev_w)
        ref_y = (grid_y.reshape(-1) + 0.5) / float(bev_h)
        return torch.stack([ref_x, ref_y], dim=-1)[None, :, None, :]

    def to_batch_first(self, bev_latent: torch.Tensor) -> torch.Tensor:
        """Convert VAD BEV [K,B,D] or [B,K,D] to [B,K,D]."""
        if bev_latent.ndim != 3:
            raise ValueError(
                f"bev_latent must be 3D, got {tuple(bev_latent.shape)}."
            )
        if bev_latent.shape[1] == self.num_tokens:
            bev = bev_latent
        elif bev_latent.shape[0] == self.num_tokens:
            bev = bev_latent.permute(1, 0, 2).contiguous()
        else:
            raise ValueError(
                "Expected one BEV dimension to equal "
                f"{self.num_tokens}, got {tuple(bev_latent.shape)}."
            )
        if bev.shape[-1] != self.embed_dims:
            raise ValueError(
                f"Expected {self.embed_dims} channels, got {bev.shape[-1]}."
            )
        return bev

    def forward(
        self,
        bev_latent: torch.Tensor,
        predicted_waypoints: torch.Tensor,
    ) -> torch.Tensor:
        """Return predicted future BEV latent [B, K, D]."""
        bev_latent = self.to_batch_first(bev_latent)
        if predicted_waypoints.ndim != 3 or predicted_waypoints.shape[-1] != 2:
            raise ValueError(
                "predicted_waypoints must be [B,M,2], got "
                f"{tuple(predicted_waypoints.shape)}."
            )
        if predicted_waypoints.shape[1] != self.num_waypoints:
            raise ValueError(
                f"Expected {self.num_waypoints} waypoints, got "
                f"{predicted_waypoints.shape[1]}."
            )
        if predicted_waypoints.shape[0] != bev_latent.shape[0]:
            raise ValueError("BEV and trajectory batch sizes must match.")

        batch_size = bev_latent.shape[0]
        trajectory = predicted_waypoints.reshape(batch_size, -1)
        trajectory = trajectory[:, None, :].expand(-1, self.num_tokens, -1)
        x = self.action_encoder(torch.cat([bev_latent, trajectory], dim=-1))

        reference_points = self.reference_points.to(
            device=x.device, dtype=x.dtype
        ).expand(batch_size, -1, -1, -1)
        spatial_shapes = self.spatial_shapes.to(device=x.device)
        level_start_index = self.level_start_index.to(device=x.device)

        for block in self.blocks:
            x = block(
                x,
                reference_points=reference_points,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
            )
        return x