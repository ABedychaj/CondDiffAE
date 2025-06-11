# Conditional Diffusion Model Demo

This notebook demonstrates how to use the conditional diffusion model for facial attribute manipulation and generation.

## Setup

```python
import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os

# Import our custom modules
from unet_model import UNet
from autoencoder_model import ConditionalAutoEncoder
from diffusion_model import ConditionalDiffusionModel
from celeba_dataset import create_celeba_dataloaders, create_attribute_vector

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
```

## Load Pre-trained Model

```python
def load_model(checkpoint_path, device):
    """Load trained model from checkpoint."""
    
    # Create model architecture (should match training configuration)
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
    
    model = ConditionalDiffusionModel(
        unet_model=unet,
        autoencoder_model=autoencoder,
        num_timesteps=1000,
    ).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    return model

# Load your trained model
checkpoint_path = "/path/to/your/checkpoint.pth"  # Update this path
model = load_model(checkpoint_path, device)
```

## Load Dataset Information

```python
# Load dataset to get attribute information
data_root = "/path/to/celeba"  # Update this path
dataloaders = create_celeba_dataloaders(
    root_dir=data_root,
    batch_size=1,
    image_size=64,
    num_workers=0,
)

attr_dict = dataloaders["attr_dict"]
attr_names = dataloaders["attr_names"]

print(f"Available attributes: {len(attr_names)}")
print(f"First 10 attributes: {attr_names[:10]}")
```

## Utility Functions

```python
def tensor_to_image(tensor):
    """Convert tensor to displayable image."""
    if tensor.dim() == 4:
        tensor = tensor[0]  # Take first image if batch
    
    # Convert from [-1, 1] to [0, 1]
    tensor = (tensor + 1) / 2
    tensor = torch.clamp(tensor, 0, 1)
    
    # Convert to numpy and transpose
    image = tensor.permute(1, 2, 0).cpu().numpy()
    return image

def display_images(images, titles=None, cols=4):
    """Display a grid of images."""
    if not isinstance(images, list):
        images = [images]
    
    rows = (len(images) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    
    if rows == 1:
        axes = [axes] if cols == 1 else axes
    else:
        axes = axes.flatten()
    
    for i, img in enumerate(images):
        if isinstance(img, torch.Tensor):
            img = tensor_to_image(img)
        
        axes[i].imshow(img)
        axes[i].axis('off')
        
        if titles and i < len(titles):
            axes[i].set_title(titles[i])
    
    # Hide unused subplots
    for i in range(len(images), len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()
```

## 1. Unconditional Generation

```python
print("Generating unconditional samples...")

with torch.no_grad():
    unconditional_samples = model.ddim_sample(
        batch_size=8,
        image_size=(64, 64),
        attributes=None,  # No conditioning
        cfg_scale=1.0,    # No guidance
        num_inference_steps=50,
        device=device,
    )

display_images(unconditional_samples, titles=[f"Sample {i+1}" for i in range(8)])
```

## 2. Conditional Generation with Specific Attributes

```python
# Define interesting attribute combinations
attribute_sets = {
    "Smiling Woman": {"Smiling": 1, "Male": 0, "Young": 1},
    "Serious Man": {"Smiling": 0, "Male": 1, "Young": 1},
    "Blonde Woman": {"Blond_Hair": 1, "Male": 0, "Attractive": 1},
    "Bearded Man": {"Male": 1, "No_Beard": 0, "Mustache": 1},
    "Person with Glasses": {"Eyeglasses": 1, "Young": 1, "Smiling": 1},
    "Elderly Person": {"Young": 0, "Gray_Hair": 1, "Smiling": 1},
}

conditional_samples = {}

print("Generating conditional samples...")

for name, attrs in attribute_sets.items():
    print(f"Generating: {name}")
    
    # Create attribute vector
    attr_vector = create_attribute_vector(attr_dict, **attrs)
    attr_batch = attr_vector.unsqueeze(0).repeat(4, 1).to(device)
    
    with torch.no_grad():
        samples = model.ddim_sample(
            batch_size=4,
            image_size=(64, 64),
            attributes=attr_batch,
            cfg_scale=2.0,  # Strong guidance
            num_inference_steps=50,
            device=device,
        )
    
    conditional_samples[name] = samples
    display_images(samples, titles=[f"{name} {i+1}" for i in range(4)])
```

## 3. Attribute Interpolation

```python
def create_interpolation(attr_dict, attrs1, attrs2, num_steps=8):
    """Create interpolation between two attribute sets."""
    
    attr_vector1 = create_attribute_vector(attr_dict, **attrs1)
    attr_vector2 = create_attribute_vector(attr_dict, **attrs2)
    
    with torch.no_grad():
        interpolated = model.interpolate_between_attributes(
            attr_vector1.unsqueeze(0),
            attr_vector2.unsqueeze(0),
            num_steps=num_steps,
            image_size=(64, 64),
            cfg_scale=2.0,
        )
    
    return interpolated

# Example 1: Male to Female transition
print("Interpolating: Serious Man → Smiling Woman")
interp1 = create_interpolation(
    attr_dict,
    {"Male": 1, "Smiling": 0},      # Serious man
    {"Male": 0, "Smiling": 1},      # Smiling woman
    num_steps=8
)
display_images(interp1, titles=[f"Step {i+1}" for i in range(8)])

# Example 2: Young to Old transition
print("Interpolating: Young → Elderly")
interp2 = create_interpolation(
    attr_dict,
    {"Young": 1, "Gray_Hair": 0},   # Young person
    {"Young": 0, "Gray_Hair": 1},   # Elderly person
    num_steps=8
)
display_images(interp2, titles=[f"Step {i+1}" for i in range(8)])
```

## 4. Single Attribute Manipulation

```python
def generate_attribute_variants(base_attrs, vary_attr, attr_dict, num_samples=6):
    """Generate variants by changing a single attribute."""
    
    variants = []
    titles = []
    
    # Generate base image
    base_vector = create_attribute_vector(attr_dict, **base_attrs)
    
    with torch.no_grad():
        # Base image
        base_sample = model.ddim_sample(
            batch_size=1,
            image_size=(64, 64),
            attributes=base_vector.unsqueeze(0).to(device),
            cfg_scale=2.0,
            num_inference_steps=50,
            device=device,
        )
        variants.append(base_sample[0])
        titles.append(f"Base ({vary_attr}=0)")
        
        # Variant with attribute enabled
        variant_attrs = base_attrs.copy()
        variant_attrs[vary_attr] = 1
        variant_vector = create_attribute_vector(attr_dict, **variant_attrs)
        
        variant_sample = model.ddim_sample(
            batch_size=1,
            image_size=(64, 64),
            attributes=variant_vector.unsqueeze(0).to(device),
            cfg_scale=2.0,
            num_inference_steps=50,
            device=device,
        )
        variants.append(variant_sample[0])
        titles.append(f"Variant ({vary_attr}=1)")
    
    return variants, titles

# Test different attributes
base_attributes = {"Male": 0, "Young": 1, "Attractive": 1}

test_attributes = ["Smiling", "Eyeglasses", "Blond_Hair", "Heavy_Makeup"]

for attr in test_attributes:
    print(f"Testing attribute: {attr}")
    variants, titles = generate_attribute_variants(base_attributes, attr, attr_dict)
    display_images(variants, titles=titles, cols=2)
```

## 5. Classifier-Free Guidance Comparison

```python
def compare_guidance_scales(attr_dict, attributes, scales=[1.0, 1.5, 2.0, 3.0]):
    """Compare different classifier-free guidance scales."""
    
    attr_vector = create_attribute_vector(attr_dict, **attributes)
    attr_batch = attr_vector.unsqueeze(0).to(device)
    
    samples = []
    titles = []
    
    with torch.no_grad():
        for scale in scales:
            sample = model.ddim_sample(
                batch_size=1,
                image_size=(64, 64),
                attributes=attr_batch,
                cfg_scale=scale,
                num_inference_steps=50,
                device=device,
            )
            samples.append(sample[0])
            titles.append(f"CFG Scale: {scale}")
    
    return samples, titles

# Compare guidance scales for a smiling woman
print("Comparing Classifier-Free Guidance scales:")
cfg_samples, cfg_titles = compare_guidance_scales(
    attr_dict, 
    {"Smiling": 1, "Male": 0, "Young": 1}
)
display_images(cfg_samples, titles=cfg_titles)
```

## 6. Custom Attribute Combinations

```python
def interactive_generation():
    """Interactive function to generate images with custom attributes."""
    
    print("Available attributes:")
    for i, attr in enumerate(attr_names):
        print(f"{i:2d}: {attr}")
    
    # You can customize these attributes
    custom_attributes = {
        "Smiling": 1,
        "Male": 0,
        "Young": 1,
        "Blond_Hair": 1,
        "Attractive": 1,
        "Heavy_Makeup": 1,
    }
    
    print(f"\\nGenerating with attributes: {custom_attributes}")
    
    attr_vector = create_attribute_vector(attr_dict, **custom_attributes)
    attr_batch = attr_vector.unsqueeze(0).repeat(4, 1).to(device)
    
    with torch.no_grad():
        samples = model.ddim_sample(
            batch_size=4,
            image_size=(64, 64),
            attributes=attr_batch,
            cfg_scale=2.0,
            num_inference_steps=50,
            device=device,
        )
    
    display_images(samples, titles=[f"Custom {i+1}" for i in range(4)])

interactive_generation()
```

## 7. Batch Generation with Different Attributes

```python
def batch_diverse_generation(num_samples=12):
    """Generate a diverse batch with different random attributes."""
    
    samples = []
    descriptions = []
    
    # Define some diverse attribute combinations
    diverse_combinations = [
        {"Male": 1, "Smiling": 1, "Young": 1},
        {"Male": 0, "Smiling": 1, "Blond_Hair": 1},
        {"Male": 1, "Eyeglasses": 1, "No_Beard": 0},
        {"Male": 0, "Heavy_Makeup": 1, "Attractive": 1},
        {"Male": 1, "Gray_Hair": 1, "Smiling": 0},
        {"Male": 0, "Young": 0, "Smiling": 1},
        {"Male": 0, "Black_Hair": 1, "Eyeglasses": 1},
        {"Male": 1, "Mustache": 1, "Serious": 1},
        {"Male": 0, "Wavy_Hair": 1, "Young": 1},
        {"Male": 1, "Bald": 1, "Smiling": 1},
        {"Male": 0, "Bangs": 1, "Attractive": 1},
        {"Male": 1, "Goatee": 1, "Young": 1},
    ]
    
    with torch.no_grad():
        for i, attrs in enumerate(diverse_combinations[:num_samples]):
            attr_vector = create_attribute_vector(attr_dict, **attrs)
            
            sample = model.ddim_sample(
                batch_size=1,
                image_size=(64, 64),
                attributes=attr_vector.unsqueeze(0).to(device),
                cfg_scale=2.0,
                num_inference_steps=50,
                device=device,
            )
            
            samples.append(sample[0])
            
            # Create description
            desc_parts = [f"{k}={v}" for k, v in attrs.items() if v == 1]
            descriptions.append(" + ".join(desc_parts))
    
    display_images(samples, titles=descriptions, cols=4)

print("Generating diverse batch with different attributes:")
batch_diverse_generation(12)
```

## Tips for Best Results

1. **CFG Scale**: Use 1.5-3.0 for good conditioning. Higher values give stronger conditioning but may reduce diversity.

2. **Inference Steps**: 50 steps usually give good quality. You can use fewer (20-30) for faster generation or more (100+) for higher quality.

3. **Attribute Combinations**: Some attributes work better together. Experiment with different combinations.

4. **Model Loading**: Make sure your checkpoint path is correct and the model architecture matches your training configuration.

5. **GPU Memory**: If you run out of memory, reduce batch size or use CPU (though it will be much slower).

## Save Generated Images

```python
def save_samples(samples, prefix="sample", output_dir="./outputs"):
    """Save generated samples to disk."""
    os.makedirs(output_dir, exist_ok=True)
    
    for i, sample in enumerate(samples):
        img = tensor_to_image(sample)
        img_pil = Image.fromarray((img * 255).astype(np.uint8))
        img_pil.save(os.path.join(output_dir, f"{prefix}_{i+1:03d}.png"))
    
    print(f"Saved {len(samples)} images to {output_dir}")

# Example: Save some conditional samples
if 'conditional_samples' in locals():
    for name, samples in conditional_samples.items():
        save_samples(samples, prefix=name.replace(" ", "_").lower())
```

This notebook provides a comprehensive demonstration of the conditional diffusion model capabilities. You can modify the attribute combinations, generation parameters, and experiment with different aspects of the model.