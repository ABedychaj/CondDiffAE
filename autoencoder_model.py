# -*- coding: utf-8 -*-
"""
Autoencoder for attribute conditioning in the diffusion model.
Includes encoder, decoder, and attribute classifier components.
"""

from typing import Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttributeEncoder(nn.Module):
    """Encoder that maps images to a latent space suitable for attribute manipulation."""

    def __init__(
            self,
            in_channels: int = 3,
            latent_dim: int = 512,
            channels: Tuple[int, ...] = (64, 128, 256, 512),
    ):
        super().__init__()
        self.latent_dim = latent_dim

        # Encoder layers
        layers = []
        prev_ch = in_channels

        for ch in channels:
            layers.extend([
                nn.Conv2d(prev_ch, ch, 4, stride=2, padding=1),
                nn.BatchNorm2d(ch),
                nn.LeakyReLU(0.2, inplace=True),
            ])
            prev_ch = ch

        self.encoder = nn.Sequential(*layers)

        # Calculate the size after convolutions (for channels[0] x channels[0] input)
        with torch.no_grad():
            dummy_input = torch.zeros(1, in_channels, channels[0], channels[0])
            dummy_output = self.encoder(dummy_input)
            self.flatten_size = dummy_output.numel()

        # Final layers to latent space
        self.fc_mu = nn.Linear(self.flatten_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_size, latent_dim)

    def forward(self, x):
        """Encode input to latent distribution parameters."""
        h = self.encoder(x)
        h = h.view(h.size(0), -1)

        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        return mu, logvar

    def reparameterize(self, mu, logvar):
        """Reparameterization trick for VAE."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std


class AttributeDecoder(nn.Module):
    """Decoder that reconstructs images from latent representations."""

    def __init__(
            self,
            latent_dim: int = 512,
            out_channels: int = 3,
            channels: Tuple[int, ...] = (512, 256, 128, 64),
            initial_size: int = 2,
    ):
        super().__init__()
        self.initial_size = initial_size
        self.initial_channels = channels[0]

        # Project latent to initial feature map
        self.fc = nn.Linear(latent_dim, channels[0] * initial_size * initial_size)

        # Decoder layers
        layers = []
        prev_ch = channels[0]

        for i, ch in enumerate(channels[1:], 1):
            layers.extend([
                nn.ConvTranspose2d(prev_ch, ch, 4, stride=2, padding=1),
                nn.BatchNorm2d(ch),
                nn.ReLU(inplace=True),
            ])
            prev_ch = ch

        # Final layer to output
        layers.extend([
            nn.ConvTranspose2d(prev_ch, out_channels, 4, stride=2, padding=1),
            nn.Tanh(),
        ])

        self.decoder = nn.Sequential(*layers)

    def forward(self, z):
        """Decode latent representation to image."""
        h = self.fc(z)
        h = h.view(h.size(0), self.initial_channels, self.initial_size, self.initial_size)
        return self.decoder(h)


class AttributeClassifier(nn.Module):
    """Classifier for predicting facial attributes from latent representations."""

    def __init__(
            self,
            latent_dim: int = 512,
            num_attributes: int = 40,
            hidden_dims: Tuple[int, ...] = (256, 128),
    ):
        super().__init__()

        layers = []
        prev_dim = latent_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2),
            ])
            prev_dim = hidden_dim

        # Final classification layer
        layers.append(nn.Linear(prev_dim, num_attributes))

        self.classifier = nn.Sequential(*layers)

    def forward(self, z):
        """Predict attributes from latent representation."""
        return self.classifier(z)


class ConditionalAutoEncoder(nn.Module):
    """
    Complete autoencoder system for conditional diffusion model.
    Combines encoder, decoder, and attribute classifier.
    """

    def __init__(
            self,
            in_channels: int = 3,
            latent_dim: int = 512,
            num_attributes: int = 40,
            encoder_channels: Tuple[int, ...] = (64, 128, 256, 512),
            decoder_channels: Tuple[int, ...] = (512, 256, 128, 64),
            classifier_hidden_dims: Tuple[int, ...] = (256, 128),
            initial_size: int = 4,
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.num_attributes = num_attributes

        # Components
        self.encoder = AttributeEncoder(
            in_channels=in_channels,
            latent_dim=latent_dim,
            channels=encoder_channels,
        )

        self.decoder = AttributeDecoder(
            latent_dim=latent_dim,
            out_channels=in_channels,
            channels=decoder_channels,
            initial_size=initial_size,
        )

        self.classifier = AttributeClassifier(
            latent_dim=latent_dim,
            num_attributes=num_attributes,
            hidden_dims=classifier_hidden_dims,
        )

    def encode(self, x):
        """Encode images to latent distribution."""
        return self.encoder(x)

    def decode(self, z):
        """Decode latent representations to images."""
        return self.decoder(z)

    def classify(self, z):
        """Classify attributes from latent representations."""
        return self.classifier(z)

    def forward(self, x, return_latent=False):
        """
        Full forward pass: encode -> decode -> classify
        
        Args:
            x: Input images [B, C, H, W]
            return_latent: Whether to return latent representations
            
        Returns:
            Dict containing reconstructed images, predicted attributes, and optionally latents
        """
        # Encode
        mu, logvar = self.encode(x)
        z = self.encoder.reparameterize(mu, logvar)

        # Decode
        x_recon = self.decode(z)

        # Classify
        attr_pred = self.classify(z)

        result = {
            'reconstruction': x_recon,
            'attributes': attr_pred,
            'mu': mu,
            'logvar': logvar,
        }

        if return_latent:
            result['latent'] = z

        return result

    def interpolate_attributes(
            self,
            x1: torch.Tensor,
            x2: torch.Tensor,
            steps: int = 10
    ) -> torch.Tensor:
        """
        Interpolate between two images in latent space.
        
        Args:
            x1: First image [1, C, H, W]
            x2: Second image [1, C, H, W]
            steps: Number of interpolation steps
            
        Returns:
            Interpolated images [steps, C, H, W]
        """
        self.eval()
        with torch.no_grad():
            # Encode both images
            mu1, _ = self.encode(x1)
            mu2, _ = self.encode(x2)

            # Create interpolation weights
            alphas = torch.linspace(0, 1, steps, device=x1.device)

            # Interpolate in latent space
            interpolated = []
            for alpha in alphas:
                z_interp = (1 - alpha) * mu1 + alpha * mu2
                x_interp = self.decode(z_interp)
                interpolated.append(x_interp)

            return torch.cat(interpolated, dim=0)

    def manipulate_attribute(
            self,
            x: torch.Tensor,
            target_attributes: torch.Tensor,
            manipulation_strength: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Manipulate specific attributes of an image.
        
        Args:
            x: Input image [1, C, H, W]
            target_attributes: Target attribute values [1, num_attributes]
            manipulation_strength: Strength of manipulation (0.0 to 1.0)
            
        Returns:
            Tuple of (manipulated_image, predicted_attributes)
        """
        self.eval()
        with torch.no_grad():
            # Encode original image
            mu, _ = self.encode(x)

            # Get current attributes
            current_attrs = torch.sigmoid(self.classify(mu))

            # Compute attribute direction
            attr_diff = target_attributes - current_attrs

            # Apply manipulation (this is a simplified approach)
            # In practice, you might want to learn attribute directions
            z_manipulated = mu + manipulation_strength * attr_diff.mean() * mu

            # Decode manipulated representation
            x_manipulated = self.decode(z_manipulated)
            attr_pred = torch.sigmoid(self.classify(z_manipulated))

            return x_manipulated, attr_pred

    def get_latent_representation(self, x: torch.Tensor) -> torch.Tensor:
        """Get latent representation for diffusion conditioning."""
        with torch.no_grad():
            mu, _ = self.encode(x)
            return mu


def compute_autoencoder_loss(
        output: Dict[str, torch.Tensor],
        target_images: torch.Tensor,
        target_attributes: torch.Tensor,
        reconstruction_weight: float = 1.0,
        kl_weight: float = 0.1,
        classification_weight: float = 1.0,
) -> Dict[str, torch.Tensor]:
    """
    Compute combined loss for autoencoder training.
    
    Args:
        output: Output from ConditionalAutoEncoder.forward()
        target_images: Ground truth images
        target_attributes: Ground truth attributes
        reconstruction_weight: Weight for reconstruction loss
        kl_weight: Weight for KL divergence loss
        classification_weight: Weight for classification loss
        
    Returns:
        Dictionary of losses
    """
    # Reconstruction loss (MSE)
    recon_loss = F.mse_loss(output['reconstruction'], target_images)

    # KL divergence loss
    kl_loss = -0.5 * torch.sum(1 + output['logvar'] - output['mu'].pow(2) - output['logvar'].exp())
    kl_loss = kl_loss / target_images.size(0)  # Normalize by batch size

    # Attribute classification loss (BCE)
    attr_loss = F.binary_cross_entropy_with_logits(
        output['attributes'],
        target_attributes.float()
    )

    # Total loss
    total_loss = (
            reconstruction_weight * recon_loss +
            kl_weight * kl_loss +
            classification_weight * attr_loss
    )

    return {
        'total_loss': total_loss,
        'reconstruction_loss': recon_loss,
        'kl_loss': kl_loss,
        'attribute_loss': attr_loss,
    }


if __name__ == "__main__":
    # Test the autoencoder
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ConditionalAutoEncoder(
        in_channels=3,
        latent_dim=64,
        num_attributes=6,
        encoder_channels=(32, 64),
        decoder_channels=(64, 32),
        classifier_hidden_dims=(128, 64),
        initial_size=8
    ).to(device)

    # Test input
    x = torch.randn(2, 3, 32, 32).to(device)
    attrs = torch.randint(0, 2, (2, 6)).float().to(device)

    # Forward pass
    with torch.no_grad():
        output = model(x, return_latent=True)
        print(f"Input shape: {x.shape}")
        print(f"Reconstruction shape: {output['reconstruction'].shape}")
        print(f"Attributes shape: {output['attributes'].shape}")
        print(f"Latent shape: {output['latent'].shape}")

        # Test loss computation
        losses = compute_autoencoder_loss(output, x, attrs)
        print(f"Total loss: {losses['total_loss'].item():.4f}")

        # Test interpolation
        x1 = x[:1]
        x2 = x[1:2]
        interpolated = model.interpolate_attributes(x1, x2, steps=5)
        print(f"Interpolated shape: {interpolated.shape}")

        print("Autoencoder test successful!")
