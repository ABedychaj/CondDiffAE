# -*- coding: utf-8 -*-
"""
Attribute manipulation and traversal script for conditional diffusion model.
"""

import os
import argparse
import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tqdm import tqdm

# Import custom modules
from unet_model import UNet
from autoencoder_model import ConditionalAutoEncoder
from diffusion_model import ConditionalDiffusionModel
from celeba_dataset import create_celeba_dataloaders, create_attribute_vector


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Manipulate attributes in conditional diffusion model")
    
    # Model arguments
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained model checkpoint")
    parser.add_argument("--data_root", type=str, required=True,
                        help="Root directory of CelebA dataset")
    parser.add_argument("--output_dir", type=str, default="./outputs",
                        help="Directory to save generated images")
    
    # Generation arguments
    parser.add_argument("--image_size", type=int, default=64,
                        help="Image resolution")
    parser.add_argument("--num_samples", type=int, default=8,
                        help="Number of samples to generate")
    parser.add_argument("--cfg_scale", type=float, default=2.0,
                        help="Classifier-free guidance scale")
    parser.add_argument("--num_inference_steps", type=int, default=50,
                        help="Number of DDIM steps")
    
    # Manipulation arguments
    parser.add_argument("--source_image", type=str, default=None,
                        help="Path to source image for manipulation")
    parser.add_argument("--manipulation_strength", type=float, default=0.8,
                        help="Strength of attribute manipulation")
    parser.add_argument("--interpolation_steps", type=int, default=10,
                        help="Number of interpolation steps")
    
    # Device
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use for inference")
    
    return parser.parse_args()


def load_model(checkpoint_path, device):
    """Load trained model from checkpoint."""
    print(f"Loading model from {checkpoint_path}")
    
    # Create model architecture (should match training)
    unet = UNet(
        in_channels=3,
        out_channels=3,
        time_emb_dim=128,
        attr_emb_dim=128,
        num_attributes=39,
        channels=(64, 128, 256, 512),
        num_blocks=2,
        use_attention=True,
    )
    
    autoencoder = ConditionalAutoEncoder(
        in_channels=3,
        latent_dim=512,
        num_attributes=39,
    )
    
    model = ConditionalDiffusionModel(
        unet_model=unet,
        autoencoder_model=autoencoder,
        num_timesteps=1000,
    ).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    print("Model loaded successfully!")
    return model


def create_attribute_combinations():
    """Create interesting attribute combinations for generation."""
    return {
        "smiling_woman": {"Smiling": 1, "Male": 0, "Young": 1},
        "serious_man": {"Smiling": 0, "Male": 1, "Young": 1},
        "blonde_woman": {"Blond_Hair": 1, "Male": 0, "Attractive": 1},
        "bearded_man": {"Male": 1, "No_Beard": 0, "Mustache": 1},
        "glasses_person": {"Eyeglasses": 1, "Young": 1, "Attractive": 1},
        "elderly_person": {"Young": 0, "Gray_Hair": 1, "Smiling": 1},
        "makeup_woman": {"Heavy_Makeup": 1, "Male": 0, "Wearing_Lipstick": 1},
        "curly_hair": {"Wavy_Hair": 1, "Black_Hair": 1, "Young": 1},
    }


def generate_conditional_samples(model, attr_dict, device, args):
    """Generate samples for different attribute combinations."""
    print("Generating conditional samples...")
    
    attribute_combinations = create_attribute_combinations()
    all_samples = {}
    
    with torch.no_grad():
        for name, attrs in tqdm(attribute_combinations.items(), desc="Generating samples"):
            # Create attribute vector
            attr_vector = create_attribute_vector(attr_dict, **attrs)
            attr_batch = attr_vector.unsqueeze(0).repeat(args.num_samples, 1).to(device)
            
            # Generate images
            samples = model.ddim_sample(
                batch_size=args.num_samples,
                image_size=(args.image_size, args.image_size),
                attributes=attr_batch,
                cfg_scale=args.cfg_scale,
                num_inference_steps=args.num_inference_steps,
                device=device,
            )
            
            # Convert to [0, 1] range
            samples = (samples + 1) / 2
            samples = torch.clamp(samples, 0, 1)
            
            all_samples[name] = samples
    
    return all_samples


def create_attribute_interpolation(model, attr_dict, device, args):
    """Create interpolation between different attributes."""
    print("Creating attribute interpolations...")
    
    interpolations = {}
    
    # Define interpolation pairs
    interpolation_pairs = [
        ({"Smiling": 0, "Male": 1}, {"Smiling": 1, "Male": 0}),  # Serious man to smiling woman
        ({"Young": 1, "Blond_Hair": 0}, {"Young": 0, "Gray_Hair": 1}),  # Young to elderly
        ({"Heavy_Makeup": 0, "Male": 1}, {"Heavy_Makeup": 1, "Male": 0}),  # No makeup man to makeup woman
        ({"Eyeglasses": 0, "Smiling": 0}, {"Eyeglasses": 1, "Smiling": 1}),  # No glasses serious to glasses smiling
    ]
    
    with torch.no_grad():
        for i, (attrs1, attrs2) in enumerate(tqdm(interpolation_pairs, desc="Creating interpolations")):
            # Create attribute vectors
            attr_vector1 = create_attribute_vector(attr_dict, **attrs1).to(device)
            attr_vector2 = create_attribute_vector(attr_dict, **attrs2).to(device)
            
            # Generate interpolation
            interpolated = model.interpolate_between_attributes(
                attr_vector1.unsqueeze(0),
                attr_vector2.unsqueeze(0),
                num_steps=args.interpolation_steps,
                image_size=(args.image_size, args.image_size),
                cfg_scale=args.cfg_scale,
            )
            
            # Convert to [0, 1] range
            interpolated = (interpolated + 1) / 2
            interpolated = torch.clamp(interpolated, 0, 1)
            
            interpolations[f"interpolation_{i+1}"] = interpolated
    
    return interpolations


def manipulate_source_image(model, source_image_path, attr_dict, device, args):
    """Manipulate attributes of a source image."""
    if not source_image_path or not os.path.exists(source_image_path):
        print("Source image not provided or doesn't exist. Skipping manipulation.")
        return {}
    
    print(f"Manipulating source image: {source_image_path}")
    
    # Load and preprocess source image
    source_image = Image.open(source_image_path).convert("RGB")
    source_image = source_image.resize((args.image_size, args.image_size))
    
    # Convert to tensor and normalize
    source_tensor = torch.from_numpy(np.array(source_image)).permute(2, 0, 1).float() / 255.0
    source_tensor = (source_tensor - 0.5) / 0.5  # Normalize to [-1, 1]
    source_tensor = source_tensor.unsqueeze(0).to(device)
    
    manipulations = {}
    
    # Define target attributes for manipulation
    manipulation_targets = {
        "add_smile": {"Smiling": 1},
        "remove_smile": {"Smiling": 0},
        "add_glasses": {"Eyeglasses": 1},
        "make_blonde": {"Blond_Hair": 1, "Black_Hair": 0},
        "add_makeup": {"Heavy_Makeup": 1, "Wearing_Lipstick": 1},
        "make_young": {"Young": 1},
        "make_male": {"Male": 1},
        "make_female": {"Male": 0},
    }
    
    with torch.no_grad():
        for name, target_attrs in tqdm(manipulation_targets.items(), desc="Manipulating attributes"):
            # Create target attribute vector
            target_vector = create_attribute_vector(attr_dict, **target_attrs)
            target_batch = target_vector.unsqueeze(0).to(device)
            
            # Manipulate image
            manipulated = model.manipulate_attributes(
                source_tensor,
                target_batch,
                num_steps=50,
                cfg_scale=args.cfg_scale,
                strength=args.manipulation_strength,
            )
            
            # Convert to [0, 1] range
            manipulated = (manipulated + 1) / 2
            manipulated = torch.clamp(manipulated, 0, 1)
            
            manipulations[name] = manipulated
    
    # Add original image
    source_display = (source_tensor + 1) / 2
    source_display = torch.clamp(source_display, 0, 1)
    manipulations["original"] = source_display
    
    return manipulations


def save_image_grid(images, filepath, nrow=8, titles=None):
    """Save a grid of images."""
    import torchvision.utils as vutils
    
    if isinstance(images, dict):
        # If images is a dictionary, create a grid for each category
        for name, imgs in images.items():
            if len(imgs.shape) == 4:  # Batch of images
                grid = vutils.make_grid(imgs, nrow=nrow, normalize=False, padding=2)
                grid_img = grid.permute(1, 2, 0).cpu().numpy()
                
                plt.figure(figsize=(15, 10))
                plt.imshow(grid_img)
                plt.title(f"{name.replace('_', ' ').title()}")
                plt.axis('off')
                
                save_path = filepath.replace('.png', f'_{name}.png')
                plt.savefig(save_path, bbox_inches='tight', dpi=150)
                plt.close()
    else:
        # Single grid
        grid = vutils.make_grid(images, nrow=nrow, normalize=False, padding=2)
        grid_img = grid.permute(1, 2, 0).cpu().numpy()
        
        plt.figure(figsize=(15, 10))
        plt.imshow(grid_img)
        if titles:
            plt.title(titles)
        plt.axis('off')
        plt.savefig(filepath, bbox_inches='tight', dpi=150)
        plt.close()


def save_interpolation_grid(interpolations, output_dir):
    """Save interpolation sequences as grids."""
    for name, sequence in interpolations.items():
        # Create a grid showing the interpolation sequence
        plt.figure(figsize=(20, 4))
        
        for i, img in enumerate(sequence):
            plt.subplot(1, len(sequence), i + 1)
            plt.imshow(img.permute(1, 2, 0).cpu().numpy())
            plt.axis('off')
            if i == 0:
                plt.title("Start")
            elif i == len(sequence) - 1:
                plt.title("End")
            else:
                plt.title(f"Step {i}")
        
        plt.suptitle(f"Attribute Interpolation: {name.replace('_', ' ').title()}")
        plt.tight_layout()
        
        save_path = os.path.join(output_dir, f"{name}_sequence.png")
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()
        
        print(f"Saved interpolation: {save_path}")


def main():
    args = parse_args()
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load dataset info to get attribute dictionary
    print("Loading dataset info...")
    dataloaders = create_celeba_dataloaders(
        root_dir=args.data_root,
        batch_size=1,
        image_size=args.image_size,
        num_workers=0,
    )
    attr_dict = dataloaders["attr_dict"]
    
    # Load model
    model = load_model(args.checkpoint, device)
    
    # Generate conditional samples
    conditional_samples = generate_conditional_samples(model, attr_dict, device, args)
    
    # Save conditional samples
    for name, samples in conditional_samples.items():
        save_path = os.path.join(args.output_dir, f"conditional_{name}.png")
        save_image_grid(samples, save_path, nrow=4)
        print(f"Saved conditional samples: {save_path}")
    
    # Create attribute interpolations
    interpolations = create_attribute_interpolation(model, attr_dict, device, args)
    save_interpolation_grid(interpolations, args.output_dir)
    
    # Manipulate source image if provided
    if args.source_image:
        manipulations = manipulate_source_image(model, args.source_image, attr_dict, device, args)
        
        if manipulations:
            # Save original and all manipulations in a grid
            all_manipulated = torch.cat(list(manipulations.values()), dim=0)
            save_path = os.path.join(args.output_dir, "attribute_manipulations.png")
            save_image_grid(all_manipulated, save_path, nrow=len(manipulations))
            print(f"Saved attribute manipulations: {save_path}")
    
    # Generate unconditional samples for comparison
    print("Generating unconditional samples...")
    with torch.no_grad():
        unconditional = model.ddim_sample(
            batch_size=args.num_samples,
            image_size=(args.image_size, args.image_size),
            attributes=None,
            cfg_scale=1.0,  # No guidance for unconditional
            num_inference_steps=args.num_inference_steps,
            device=device,
        )
        
        unconditional = (unconditional + 1) / 2
        unconditional = torch.clamp(unconditional, 0, 1)
        
        save_path = os.path.join(args.output_dir, "unconditional_samples.png")
        save_image_grid(unconditional, save_path, nrow=4, titles="Unconditional Generation")
        print(f"Saved unconditional samples: {save_path}")
    
    print(f"All outputs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()