# -*- coding: utf-8 -*-
"""
UNet architecture for conditional diffusion model.
Supports attribute conditioning through cross-attention and classifier-free guidance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
from einops import rearrange


class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal position embeddings for timesteps."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class Block(nn.Module):
    """Basic residual block with group normalization."""

    def __init__(self, in_ch: int, out_ch: int, time_emb_dim: int, groups: int = 8):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(groups, out_ch)
        self.norm2 = nn.GroupNorm(groups, out_ch)
        self.activation = nn.SiLU()

        # Residual connection
        if in_ch != out_ch:
            self.residual_conv = nn.Conv2d(in_ch, out_ch, 1)
        else:
            self.residual_conv = nn.Identity()

    def forward(self, x, time_emb):
        h = self.conv1(x)
        h = self.norm1(h)
        h = self.activation(h)

        # Add time embedding
        time_emb = self.activation(self.time_mlp(time_emb))
        h = h + time_emb[:, :, None, None]

        h = self.conv2(h)
        h = self.norm2(h)
        h = self.activation(h)

        return h + self.residual_conv(x)


class CrossAttention(nn.Module):
    """Cross-attention for conditioning on attributes."""

    def __init__(self, dim: int, context_dim: int, heads: int = 8):
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5

        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(context_dim, dim, bias=False)
        self.to_v = nn.Linear(context_dim, dim, bias=False)
        self.to_out = nn.Linear(dim, dim)

    def forward(self, x, context):
        h, w = x.shape[-2:]
        x = rearrange(x, 'b c h w -> b (h w) c')

        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), (q, k, v))

        sim = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        attn = sim.softmax(dim=-1)

        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)

        return rearrange(out, 'b (h w) c -> b c h w', h=h, w=w)


class AttentionBlock(nn.Module):
    """Attention block with optional cross-attention for conditioning."""

    def __init__(self, dim: int, context_dim: Optional[int] = None, heads: int = 8):
        super().__init__()
        self.norm = nn.GroupNorm(8, dim)
        self.self_attn = SelfAttention(channels=dim)

        if context_dim is not None:
            self.cross_attn = CrossAttention(dim, context_dim, heads)
            self.norm_cross = nn.GroupNorm(8, dim)
        else:
            self.cross_attn = None

    def forward(self, x, context=None):
        # Self-attention
        residual = x
        x = self.norm(x)
        x = self.self_attn(x, x) + residual

        # Cross-attention
        if self.cross_attn is not None and context is not None:
            residual = x
            x = self.norm_cross(x)
            x = self.cross_attn(x, context) + residual

        return x


class SelfAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels

        # Use proper channel dimensions
        self.to_q = nn.Linear(channels, channels)
        self.to_k = nn.Linear(channels, channels)
        self.to_v = nn.Linear(channels, channels)

        self.scale = channels ** -0.5
        self.gamma = nn.Parameter(torch.tensor(1.0))  # Learnable scale parameter

    def forward(self, x, context=None):
        context = x if context is None else context

        # Preserve original shape
        batch, c, h, w = x.shape

        # Reshape with proper channel dimension
        x_flat = x.view(batch, c, h * w).permute(0, 2, 1)  # [batch, h*w, c]

        # Compute query/key/value projections
        q = self.to_q(x_flat)
        k = self.to_k(context.view(batch, c, -1).permute(0, 2, 1))
        v = self.to_v(context.view(batch, c, -1).permute(0, 2, 1))

        # Attention computation
        attn = torch.bmm(q, k.permute(0, 2, 1)) * self.scale
        attn = F.softmax(attn, dim=-1)

        out = torch.bmm(attn, v)
        out = out.permute(0, 2, 1).view(batch, c, h, w)
        return self.gamma * out + x  # Residual connection


class UNet(nn.Module):
    """UNet architecture for conditional diffusion model."""

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        time_emb_dim: int = 128,
        attr_emb_dim: int = 128,
        num_attributes: int = 40,  # CelebA has 40 attributes
        channels: Tuple[int, ...] = (64, 128, 256, 512),
        num_blocks: int = 2,
        num_heads: int = 8,
        use_attention: bool = True,
    ):
        super().__init__()

        self.time_emb_dim = time_emb_dim
        self.attr_emb_dim = attr_emb_dim

        # Time embedding
        self.time_pos_emb = SinusoidalPositionEmbeddings(time_emb_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim),
        )

        # Attribute embedding for conditioning
        self.attr_embedding = nn.Embedding(num_attributes * 2, attr_emb_dim)  # *2 for binary attributes
        self.attr_mlp = nn.Sequential(
            nn.Linear(num_attributes * attr_emb_dim, attr_emb_dim),
            nn.SiLU(),
            nn.Linear(attr_emb_dim, attr_emb_dim),
        )

        # Initial convolution
        self.init_conv = nn.Conv2d(in_channels, channels[0], 7, padding=3)

        # Encoder (downsampling)
        self.encoder_blocks = nn.ModuleList()
        self.down_samples = nn.ModuleList()

        in_ch = channels[0]
        for i, out_ch in enumerate(channels):
            blocks = nn.ModuleList()
            for _ in range(num_blocks):
                blocks.append(Block(in_ch, out_ch, time_emb_dim))
                if use_attention and i >= len(channels) // 2:
                    blocks.append(AttentionBlock(out_ch, attr_emb_dim, num_heads))
                in_ch = out_ch

            self.encoder_blocks.append(blocks)

            if i < len(channels) - 1:
                self.down_samples.append(nn.Conv2d(out_ch, out_ch, 4, stride=2, padding=1))
            else:
                self.down_samples.append(nn.Identity())

        # Middle block
        self.middle_block = nn.ModuleList([
            Block(channels[-1], channels[-1], time_emb_dim),
            AttentionBlock(channels[-1], attr_emb_dim, num_heads) if use_attention else nn.Identity(),
            Block(channels[-1], channels[-1], time_emb_dim),
        ])

        # Decoder (upsampling)
        self.decoder_blocks = nn.ModuleList()
        self.up_samples = nn.ModuleList()

        for i, out_ch in enumerate(reversed(channels)):
            blocks = nn.ModuleList()
            in_ch = channels[-1] if i == 0 else channels[-i]
            skip_ch = channels[-i-1] if i < len(channels) - 1 else channels[0]

            for j in range(num_blocks + 1):
                if j == 0:
                    # First block receives skip connection
                    blocks.append(Block(in_ch + skip_ch, out_ch, time_emb_dim))
                else:
                    blocks.append(Block(out_ch, out_ch, time_emb_dim))

                if use_attention and (len(channels) - i - 1) >= len(channels) // 2:
                    blocks.append(AttentionBlock(out_ch, attr_emb_dim, num_heads))

            self.decoder_blocks.append(blocks)

            if i < len(channels) - 1:
                self.up_samples.append(nn.ConvTranspose2d(out_ch, out_ch, 4, stride=2, padding=1))
            else:
                self.up_samples.append(nn.Identity())

        # Final convolution
        self.final_conv = nn.Sequential(
            nn.GroupNorm(8, channels[0]),
            nn.SiLU(),
            nn.Conv2d(channels[0], out_channels, 3, padding=1),
        )

    def forward(self, x, timesteps, attributes=None, cfg_scale=1.0):
        """
        Forward pass of the UNet.
        
        Args:
            x: Input tensor [B, C, H, W]
            timesteps: Timestep tensor [B]
            attributes: Attribute tensor [B, num_attributes] (binary)
            cfg_scale: Classifier-free guidance scale
        """
        batch_size = x.shape[0]

        # Time embedding
        time_emb = self.time_pos_emb(timesteps)
        time_emb = self.time_mlp(time_emb)

        # Attribute embedding (for conditioning)
        attr_context = None
        if attributes is not None:
            # Convert binary attributes to embeddings
            attr_indices = attributes.long()  # [B, num_attributes]
            attr_embs = []

            for i in range(attributes.shape[1]):
                # For each attribute, get embedding for 0 or 1
                attr_idx = attr_indices[:, i] + i * 2  # Offset for each attribute
                attr_embs.append(self.attr_embedding(attr_idx))

            attr_emb = torch.cat(attr_embs, dim=-1)  # [B, num_attributes * attr_emb_dim]
            attr_context = self.attr_mlp(attr_emb).unsqueeze(1)  # [B, 1, attr_emb_dim]

        # Classifier-free guidance
        if cfg_scale > 1.0 and attr_context is not None:
            # Unconditional forward pass
            x_uncond = self._forward_impl(x, time_emb, None)
            # Conditional forward pass
            x_cond = self._forward_impl(x, time_emb, attr_context)
            # Apply CFG
            return x_uncond + cfg_scale * (x_cond - x_uncond)
        else:
            return self._forward_impl(x, time_emb, attr_context)

    def _forward_impl(self, x, time_emb, attr_context):
        """Implementation of forward pass."""
        # Initial convolution
        x = self.init_conv(x)

        # Store skip connections
        skip_connections = [x]

        # Encoder
        for blocks, downsample in zip(self.encoder_blocks, self.down_samples):
            for block in blocks:
                if isinstance(block, Block):
                    x = block(x, time_emb)
                elif isinstance(block, AttentionBlock):
                    x = block(x, attr_context)

            skip_connections.append(x)
            x = downsample(x)

        # Middle
        for block in self.middle_block:
            if isinstance(block, Block):
                x = block(x, time_emb)
            elif isinstance(block, AttentionBlock):
                x = block(x, attr_context)
            elif not isinstance(block, nn.Identity):
                x = block(x)

        # Decoder
        for blocks, upsample in zip(self.decoder_blocks, self.up_samples):
            # Skip connection
            skip = skip_connections.pop()
            x = torch.cat([x, skip], dim=1)

            for block in blocks:
                if isinstance(block, Block):
                    x = block(x, time_emb)
                elif isinstance(block, AttentionBlock):
                    x = block(x, attr_context)

            x = upsample(x)

        # Final convolution
        return self.final_conv(x)


if __name__ == "__main__":
    # Test the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UNet(
        in_channels=3,
        out_channels=3,
        time_emb_dim=128,
        attr_emb_dim=128,
        num_attributes=40,
        channels=(64, 128, 256, 512),
        num_blocks=2,
        use_attention=True,
    ).to(device)

    # Test input
    x = torch.randn(2, 3, 64, 64).to(device)
    t = torch.randint(0, 1000, (2,)).to(device)
    attrs = torch.randint(0, 2, (2, 40)).float().to(device)

    # Forward pass
    with torch.no_grad():
        output = model(x, t, attrs)
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {output.shape}")
        print("Model test successful!")

        # Test classifier-free guidance
        output_cfg = model(x, t, attrs, cfg_scale=2.0)
        print(f"CFG output shape: {output_cfg.shape}")