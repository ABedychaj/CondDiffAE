# -*- coding: utf-8 -*-
"""
Diffusion Model implementation with conditional generation and attribute manipulation.
Combines UNet and Autoencoder for attribute-conditioned image generation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict, List
from tqdm import tqdm


class DiffusionScheduler:
    """Noise scheduler for diffusion process."""
    
    def __init__(
        self,
        num_timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        schedule_type: str = "linear"
    ):
        self.num_timesteps = num_timesteps
        
        if schedule_type == "linear":
            self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        elif schedule_type == "cosine":
            self.betas = self._cosine_beta_schedule(num_timesteps, beta_start, beta_end)
        else:
            raise ValueError(f"Unknown schedule type: {schedule_type}")
        
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # Calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )

    def _cosine_beta_schedule(self, timesteps, beta_start, beta_end):
        """Cosine noise schedule."""
        s = 0.008
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, beta_start, beta_end)

    def add_noise(self, x_start, noise, timesteps):
        """Add noise to the original images."""
        sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(x_start.device)
        sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(x_start.device)
        
        sqrt_alphas_cumprod_t = sqrt_alphas_cumprod[timesteps].view(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod[timesteps].view(-1, 1, 1, 1)
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def sample_timesteps(self, batch_size, device):
        """Sample random timesteps."""
        return torch.randint(0, self.num_timesteps, (batch_size,), device=device)


class ConditionalDiffusionModel(nn.Module):
    """
    Main diffusion model that combines UNet and Autoencoder for conditional generation.
    """
    
    def __init__(
        self,
        unet_model,
        autoencoder_model,
        num_timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        schedule_type: str = "linear",
    ):
        super().__init__()
        
        self.unet = unet_model
        self.autoencoder = autoencoder_model
        self.scheduler = DiffusionScheduler(
            num_timesteps=num_timesteps,
            beta_start=beta_start,
            beta_end=beta_end,
            schedule_type=schedule_type,
        )

    def forward(self, x, attributes=None, cfg_scale=1.0):
        """
        Training forward pass.
        
        Args:
            x: Input images [B, C, H, W]
            attributes: Attribute conditioning [B, num_attributes]
            cfg_scale: Classifier-free guidance scale
            
        Returns:
            Loss dictionary
        """
        batch_size = x.size(0)
        device = x.device
        
        # Sample timesteps
        timesteps = self.scheduler.sample_timesteps(batch_size, device)
        
        # Sample noise
        noise = torch.randn_like(x)
        
        # Add noise to images
        x_noisy = self.scheduler.add_noise(x, noise, timesteps)
        
        # Predict noise with UNet
        predicted_noise = self.unet(x_noisy, timesteps, attributes, cfg_scale)
        
        # Compute diffusion loss
        diffusion_loss = F.mse_loss(predicted_noise, noise)
        
        # Get autoencoder loss if training jointly
        if self.training and attributes is not None:
            ae_output = self.autoencoder(x, return_latent=True)
            ae_losses = self._compute_autoencoder_loss(ae_output, x, attributes)
            
            return {
                'diffusion_loss': diffusion_loss,
                'total_loss': diffusion_loss + ae_losses['total_loss'] * 0.1,  # Weight AE loss
                **ae_losses
            }
        
        return {'diffusion_loss': diffusion_loss, 'total_loss': diffusion_loss}

    def _compute_autoencoder_loss(self, output, target_images, target_attributes):
        """Compute autoencoder losses."""
        recon_loss = F.mse_loss(output['reconstruction'], target_images)
        kl_loss = -0.5 * torch.sum(1 + output['logvar'] - output['mu'].pow(2) - output['logvar'].exp())
        kl_loss = kl_loss / target_images.size(0)
        attr_loss = F.binary_cross_entropy_with_logits(output['attributes'], target_attributes.float())
        
        total_loss = recon_loss + 0.1 * kl_loss + attr_loss
        
        return {
            'total_loss': total_loss,
            'reconstruction_loss': recon_loss,
            'kl_loss': kl_loss,
            'attribute_loss': attr_loss,
        }

    @torch.no_grad()
    def sample(
        self,
        batch_size: int = 1,
        image_size: Tuple[int, int] = (64, 64),
        attributes: Optional[torch.Tensor] = None,
        cfg_scale: float = 1.0,
        num_inference_steps: Optional[int] = None,
        device: str = "cuda",
    ) -> torch.Tensor:
        """
        Sample images using DDPM.
        
        Args:
            batch_size: Number of images to generate
            image_size: Size of generated images (H, W)
            attributes: Conditioning attributes [batch_size, num_attributes]
            cfg_scale: Classifier-free guidance scale
            num_inference_steps: Number of denoising steps (defaults to full schedule)
            device: Device to run on
            
        Returns:
            Generated images [batch_size, 3, H, W]
        """
        if num_inference_steps is None:
            num_inference_steps = self.scheduler.num_timesteps
        
        # Start with random noise
        x = torch.randn(batch_size, 3, *image_size, device=device)
        
        # Create timestep schedule
        timesteps = torch.linspace(
            self.scheduler.num_timesteps - 1, 0, num_inference_steps, dtype=torch.long, device=device
        )
        
        self.eval()
        for i, t in enumerate(tqdm(timesteps, desc="Sampling")):
            t_batch = t.repeat(batch_size)
            
            # Predict noise
            predicted_noise = self.unet(x, t_batch, attributes, cfg_scale)
            
            # Compute denoising step
            x = self._ddpm_step(x, predicted_noise, t)
        
        return torch.clamp(x, -1, 1)

    @torch.no_grad()
    def ddim_sample(
        self,
        batch_size: int = 1,
        image_size: Tuple[int, int] = (64, 64),
        attributes: Optional[torch.Tensor] = None,
        cfg_scale: float = 1.0,
        num_inference_steps: int = 50,
        eta: float = 0.0,
        device: str = "cuda",
    ) -> torch.Tensor:
        """
        Sample images using DDIM (faster sampling).
        
        Args:
            batch_size: Number of images to generate
            image_size: Size of generated images (H, W)
            attributes: Conditioning attributes
            cfg_scale: Classifier-free guidance scale
            num_inference_steps: Number of denoising steps
            eta: DDIM parameter (0.0 for deterministic)
            device: Device to run on
            
        Returns:
            Generated images
        """
        # Create timestep schedule for DDIM
        skip = self.scheduler.num_timesteps // num_inference_steps
        timesteps = torch.arange(0, self.scheduler.num_timesteps, skip, device=device).flip(0)
        
        # Start with random noise
        x = torch.randn(batch_size, 3, *image_size, device=device)
        
        self.eval()
        for i, t in enumerate(tqdm(timesteps, desc="DDIM Sampling")):
            t_batch = t.repeat(batch_size)
            
            # Predict noise
            predicted_noise = self.unet(x, t_batch, attributes, cfg_scale)
            
            # DDIM step
            prev_t = timesteps[i + 1] if i < len(timesteps) - 1 else torch.tensor(-1, device=device)
            x = self._ddim_step(x, predicted_noise, t, prev_t, eta)
        
        return torch.clamp(x, -1, 1)

    def _ddpm_step(self, x_t, predicted_noise, t):
        """Single DDPM denoising step."""
        device = x_t.device
        
        betas = self.scheduler.betas.to(device)
        alphas = self.scheduler.alphas.to(device)
        alphas_cumprod = self.scheduler.alphas_cumprod.to(device)
        posterior_variance = self.scheduler.posterior_variance.to(device)
        
        # Extract values for current timestep
        alpha_t = alphas[t]
        alpha_cumprod_t = alphas_cumprod[t]
        beta_t = betas[t]
        
        # Predict x_0
        x_0_pred = (x_t - torch.sqrt(1 - alpha_cumprod_t) * predicted_noise) / torch.sqrt(alpha_cumprod_t)
        
        # Compute mean of posterior
        x_0_pred = torch.clamp(x_0_pred, -1, 1)
        
        if t > 0:
            alpha_cumprod_prev = alphas_cumprod[t - 1]
            posterior_mean = (
                torch.sqrt(alpha_cumprod_prev) * beta_t / (1 - alpha_cumprod_t) * x_0_pred +
                torch.sqrt(alpha_t) * (1 - alpha_cumprod_prev) / (1 - alpha_cumprod_t) * x_t
            )
            
            # Add noise
            noise = torch.randn_like(x_t)
            x_prev = posterior_mean + torch.sqrt(posterior_variance[t]) * noise
        else:
            x_prev = x_0_pred
        
        return x_prev

    def _ddim_step(self, x_t, predicted_noise, t, prev_t, eta):
        """Single DDIM denoising step."""
        device = x_t.device
        alphas_cumprod = self.scheduler.alphas_cumprod.to(device)
        
        alpha_cumprod_t = alphas_cumprod[t]
        alpha_cumprod_prev = alphas_cumprod[prev_t] if prev_t >= 0 else torch.tensor(1.0, device=device)
        
        # Predict x_0
        x_0_pred = (x_t - torch.sqrt(1 - alpha_cumprod_t) * predicted_noise) / torch.sqrt(alpha_cumprod_t)
        x_0_pred = torch.clamp(x_0_pred, -1, 1)
        
        # Direction pointing to x_t
        dir_xt = torch.sqrt(1 - alpha_cumprod_prev - eta**2 * (1 - alpha_cumprod_t)) * predicted_noise
        
        # Random noise
        noise = torch.randn_like(x_t) if eta > 0 else torch.zeros_like(x_t)
        
        x_prev = torch.sqrt(alpha_cumprod_prev) * x_0_pred + dir_xt + eta * torch.sqrt(1 - alpha_cumprod_t) * noise
        
        return x_prev

    @torch.no_grad()
    def manipulate_attributes(
        self,
        source_image: torch.Tensor,
        target_attributes: torch.Tensor,
        num_steps: int = 100,
        cfg_scale: float = 2.0,
        strength: float = 0.8,
    ) -> torch.Tensor:
        """
        Manipulate attributes of a source image.
        
        Args:
            source_image: Source image [1, 3, H, W]
            target_attributes: Target attributes [1, num_attributes]
            num_steps: Number of manipulation steps
            cfg_scale: Classifier-free guidance scale
            strength: Manipulation strength (0.0 to 1.0)
            
        Returns:
            Manipulated image
        """
        self.eval()
        device = source_image.device
        
        # Add noise to source image
        timestep = int(strength * self.scheduler.num_timesteps)
        t = torch.tensor([timestep], device=device)
        noise = torch.randn_like(source_image)
        x_t = self.scheduler.add_noise(source_image, noise, t)
        
        # Denoise with target attributes
        timesteps = torch.arange(timestep, 0, -max(1, timestep // num_steps), device=device)
        
        for t in timesteps:
            t_batch = t.repeat(source_image.size(0))
            predicted_noise = self.unet(x_t, t_batch, target_attributes, cfg_scale)
            x_t = self._ddpm_step(x_t, predicted_noise, t)
        
        return torch.clamp(x_t, -1, 1)

    @torch.no_grad()
    def interpolate_between_attributes(
        self,
        attributes_1: torch.Tensor,
        attributes_2: torch.Tensor,
        num_steps: int = 10,
        image_size: Tuple[int, int] = (64, 64),
        cfg_scale: float = 2.0,
    ) -> torch.Tensor:
        """
        Generate interpolation between two sets of attributes.
        
        Args:
            attributes_1: First set of attributes [1, num_attributes]
            attributes_2: Second set of attributes [1, num_attributes]
            num_steps: Number of interpolation steps
            image_size: Size of generated images
            cfg_scale: Classifier-free guidance scale
            
        Returns:
            Interpolated images [num_steps, 3, H, W]
        """
        device = attributes_1.device
        
        # Create interpolation weights
        alphas = torch.linspace(0, 1, num_steps, device=device)
        
        # Generate images for each interpolation step
        interpolated_images = []
        
        for alpha in alphas:
            # Interpolate attributes
            attrs_interp = (1 - alpha) * attributes_1 + alpha * attributes_2
            
            # Generate image with interpolated attributes
            img = self.ddim_sample(
                batch_size=1,
                image_size=image_size,
                attributes=attrs_interp,
                cfg_scale=cfg_scale,
                num_inference_steps=50,
                device=device,
            )
            interpolated_images.append(img)
        
        return torch.cat(interpolated_images, dim=0)


if __name__ == "__main__":
    # Test the complete diffusion model
    from unet_model import UNet
    from autoencoder_model import ConditionalAutoEncoder
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create models
    unet = UNet(
        in_channels=3,
        out_channels=3,
        time_emb_dim=128,
        attr_emb_dim=128,
        num_attributes=40,
        channels=(64, 128, 256, 512),
        num_blocks=2,
        use_attention=True,
    )
    
    autoencoder = ConditionalAutoEncoder(
        in_channels=3,
        latent_dim=512,
        num_attributes=40,
    )
    
    # Create diffusion model
    diffusion_model = ConditionalDiffusionModel(
        unet_model=unet,
        autoencoder_model=autoencoder,
        num_timesteps=1000,
    ).to(device)
    
    # Test training step
    x = torch.randn(2, 3, 64, 64).to(device)
    attrs = torch.randint(0, 2, (2, 40)).float().to(device)
    
    with torch.no_grad():
        losses = diffusion_model(x, attrs)
        print(f"Diffusion loss: {losses['diffusion_loss'].item():.4f}")
        print(f"Total loss: {losses['total_loss'].item():.4f}")
        
        # Test sampling
        sampled = diffusion_model.ddim_sample(
            batch_size=1,
            image_size=(64, 64),
            attributes=attrs[:1],
            num_inference_steps=10,
            device=device,
        )
        print(f"Sampled image shape: {sampled.shape}")
        
        print("Diffusion model test successful!")