# -*- coding: utf-8 -*-
"""
Training script for conditional diffusion model with autoencoder.
"""

import os
import argparse
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from datetime import datetime

# Import custom modules
from unet_model import UNet
from autoencoder_model import ConditionalAutoEncoder
from diffusion_model import ConditionalDiffusionModel
from shapes_dataset import create_shapes3d_dataloaders

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train conditional diffusion model on CelebA")
    
    # Data arguments
    parser.add_argument("--data_root", type=str, required=True,
                        help="Root directory of CelebA dataset")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for training")
    parser.add_argument("--image_size", type=int, default=32,
                        help="Image resolution")
    parser.add_argument("--num_workers", type=int, default=1,
                        help="Number of data loading workers")
    
    # Model arguments
    parser.add_argument("--num_timesteps", type=int, default=1000,
                        help="Number of diffusion timesteps")
    parser.add_argument("--channels", nargs="+", type=int, default=[32, 64],
                        help="UNet channel dimensions")
    parser.add_argument("--latent_dim", type=int, default=64,
                        help="Autoencoder latent dimension")
    parser.add_argument("--time_emb_dim", type=int, default=64,
                        help="Time embedding dimension")
    parser.add_argument("--attr_emb_dim", type=int, default=64,
                        help="Attribute embedding dimension")
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=2e-4,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-6,
                        help="Weight decay")
    parser.add_argument("--ae_weight", type=float, default=0.5,
                        help="Weight for autoencoder loss")
    parser.add_argument("--cfg_scale", type=float, default=1.5,
                        help="Classifier-free guidance scale")
    parser.add_argument("--cfg_dropout", type=float, default=0.1,
                        help="Probability of dropping conditioning for CFG")
    
    # Checkpointing and logging
    parser.add_argument("--save_dir", type=str, default="./checkpoints",
                        help="Directory to save checkpoints")
    parser.add_argument("--log_dir", type=str, default="./logs",
                        help="Directory for tensorboard logs")
    parser.add_argument("--save_interval", type=int, default=10,
                        help="Save checkpoint every N epochs")
    parser.add_argument("--sample_interval", type=int, default=5,
                        help="Generate samples every N epochs")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    
    # Device
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use for training")
    
    return parser.parse_args()


def train_epoch(model, dataloader, optimizer, device, cfg_dropout=0.1):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_diffusion_loss = 0
    total_ae_loss = 0
    
    progress_bar = tqdm(dataloader, desc="Training")

    for batch_idx, batch in enumerate(progress_bar):
        images = batch["image"].to(device)
        attributes = batch["factors"].to(device)
        
        # Randomly drop conditioning for classifier-free guidance
        if torch.rand(1).item() < cfg_dropout:
            attributes = None
        
        # Forward pass
        optimizer.zero_grad()

        losses = model(images, attributes)
        
        # Backward pass
        loss = losses["total_loss"]
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        
        # Update metrics
        total_loss += loss.item()
        total_diffusion_loss += losses["diffusion_loss"].item()
        if "reconstruction_loss" in losses:
            total_ae_loss += losses["reconstruction_loss"].item()
        
        # Update progress bar
        progress_bar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "diff": f"{losses['diffusion_loss'].item():.4f}",
        })
    
    return {
        "total_loss": total_loss / len(dataloader),
        "diffusion_loss": total_diffusion_loss / len(dataloader),
        "ae_loss": total_ae_loss / len(dataloader) if total_ae_loss > 0 else 0,
    }


def validate_epoch(model, dataloader, device):
    """Validate for one epoch."""
    model.eval()
    total_loss = 0
    total_diffusion_loss = 0
    total_ae_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            images = batch["image"].to(device)
            attributes = batch["factors"].to(device)
            
            losses = model(images, attributes)
            
            total_loss += losses["total_loss"].item()
            total_diffusion_loss += losses["diffusion_loss"].item()
            if "reconstruction_loss" in losses:
                total_ae_loss += losses["reconstruction_loss"].item()
    
    return {
        "total_loss": total_loss / len(dataloader),
        "diffusion_loss": total_diffusion_loss / len(dataloader),
        "ae_loss": total_ae_loss / len(dataloader) if total_ae_loss > 0 else 0,
    }


def generate_samples(model, attr_dict, device, num_samples=8, image_size=(64, 64)):
    """Generate sample images for different attributes."""
    model.eval()
    
    samples = {}
    
    with torch.no_grad():
        # Unconditional generation
        unconditional = model.ddim_sample(
            batch_size=num_samples,
            image_size=image_size,
            attributes=None,
            num_inference_steps=50,
            device=device,
        )
        samples["unconditional"] = unconditional
        
        # Conditional generation for specific attributes
        attribute_sets = {
            "object_wall": {"wall_hue": 0, "object_hue": 0},
            "floor_wall": {"floor_hue": 0.2, "wall_hue": 0.8},
            "scale_1": {"scale": 1.25, "wall_hue": 0},
            "orientation_0": {"orientation": 0, "scale": 0.75},
        }
        
        for name, attrs in attribute_sets.items():
            # Create attribute vector
            attr_vector = torch.zeros(len(attr_dict))
            for attr_name, value in attrs.items():
                if attr_name in attr_dict:
                    attr_vector[attr_dict[attr_name]] = float(value)
            
            attr_batch = attr_vector.unsqueeze(0).repeat(num_samples, 1).to(device)
            
            conditional = model.ddim_sample(
                batch_size=num_samples,
                image_size=image_size,
                attributes=attr_batch,
                cfg_scale=2.0,
                num_inference_steps=50,
                device=device,
            )
            samples[name] = conditional
    
    return samples


def save_checkpoint(model, optimizer, epoch, loss, filepath):
    """Save training checkpoint."""
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
    }
    torch.save(checkpoint, filepath)


def load_checkpoint(filepath, model, optimizer=None):
    """Load training checkpoint."""
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    return checkpoint["epoch"], checkpoint["loss"]


def main():
    args = parse_args()
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create directories
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Create timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"conditional_diffusion_{timestamp}"
    
    # Setup tensorboard
    writer = SummaryWriter(os.path.join(args.log_dir, run_name))
    
    # Load dataset
    print("Loading 3dshapes dataset...")
    dataloaders = create_shapes3d_dataloaders(
        root_dir=args.data_root,
        batch_size=args.batch_size,
        image_size=args.image_size,
        num_workers=args.num_workers,
    )
    
    train_loader = dataloaders["train"]
    val_loader = dataloaders["val"]
    attr_dict = dataloaders["factor_dict"]
    num_attributes = dataloaders["num_factors"]
    
    print(f"Dataset loaded: {len(train_loader)} training batches, {len(val_loader)} validation batches")
    print(f"Number of attributes: {num_attributes}")
    
    # Create models
    print("Creating models...")
    
    # UNet
    unet = UNet(
        in_channels=3,
        out_channels=3,
        time_emb_dim=args.time_emb_dim,
        attr_emb_dim=args.attr_emb_dim,
        num_attributes=num_attributes,
        channels=tuple(args.channels),
        num_blocks=2,
        use_attention=True,
    )
    
    # Autoencoder
    autoencoder = ConditionalAutoEncoder(
        in_channels=3,
        latent_dim=args.latent_dim,
        num_attributes=num_attributes,
        encoder_channels = tuple(args.channels),
        decoder_channels = tuple(reversed(args.channels)),
        classifier_hidden_dims = tuple(args.channels),
        initial_size=8,
    )
    
    # Combined diffusion model
    model = ConditionalDiffusionModel(
        unet_model=unet,
        autoencoder_model=autoencoder,
        num_timesteps=args.num_timesteps,
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Setup optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999)
    )
    
    # Setup learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        print(f"Resuming from {args.resume}")
        start_epoch, _ = load_checkpoint(args.resume, model, optimizer)
        start_epoch += 1
    
    # Training loop
    print("Starting training...")
    best_val_loss = float('inf')
    
    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, device, args.cfg_dropout
        )
        
        # Validate
        val_metrics = validate_epoch(model, val_loader, device)
        
        # Update learning rate
        scheduler.step()
        
        # Log metrics
        for key in ["total_loss", "diffusion_loss", "ae_loss"]:
            writer.add_scalar(f"train/{key}", train_metrics[key], epoch)
            writer.add_scalar(f"val/{key}", val_metrics[key], epoch)
        
        writer.add_scalar("lr", scheduler.get_last_lr()[0], epoch)
        
        print(f"Train Loss: {train_metrics['total_loss']:.4f}, "
              f"Val Loss: {val_metrics['total_loss']:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % args.save_interval == 0:
            checkpoint_path = os.path.join(args.save_dir, f"checkpoint_epoch_{epoch + 1}.pth")
            save_checkpoint(model, optimizer, epoch, val_metrics["total_loss"], checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")
        
        # Save best model
        if val_metrics["total_loss"] < best_val_loss:
            best_val_loss = val_metrics["total_loss"]
            best_path = os.path.join(args.save_dir, "best_model.pth")
            save_checkpoint(model, optimizer, epoch, val_metrics["total_loss"], best_path)
        
        # Generate samples
        if (epoch + 1) % args.sample_interval == 0:
            print("Generating samples...")
            samples = generate_samples(
                model, attr_dict, device, 
                num_samples=4, image_size=(args.image_size, args.image_size)
            )
            
            # Log samples to tensorboard
            for name, sample in samples.items():
                # Convert from [-1, 1] to [0, 1]
                sample = (sample + 1) / 2
                sample = torch.clamp(sample, 0, 1)
                writer.add_images(f"samples/{name}", sample, epoch, dataformats="NCHW")
    
    # Save final model
    final_path = os.path.join(args.save_dir, "final_model.pth")
    save_checkpoint(model, optimizer, args.epochs - 1, val_metrics["total_loss"], final_path)
    
    writer.close()
    print("Training completed!")


if __name__ == "__main__":
    main()