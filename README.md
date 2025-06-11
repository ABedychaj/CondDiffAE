# Conditional Diffusion Model with Autoencoder for CelebA

A PyTorch implementation of a conditional diffusion model trained on the CelebA dataset, featuring an autoencoder for attribute conditioning and continuous traversal between different facial attributes. This project allows you to generate faces with specific attributes and smoothly transition between different characteristics like gender, age, hair color, and facial expressions.

## Features

- **Conditional Diffusion Model**: DDPM/DDIM implementation with UNet architecture
- **Attribute Conditioning**: Cross-attention mechanism for 40 CelebA facial attributes
- **Autoencoder Integration**: VAE for latent space manipulation and attribute prediction
- **Classifier-Free Guidance**: Improved conditional generation quality
- **Attribute Manipulation**: Edit specific attributes of existing images
- **Smooth Interpolation**: Continuous traversal between different attribute combinations
- **Comprehensive Training**: Joint training of diffusion model and autoencoder

## Architecture Overview

The system consists of three main components:

1. **UNet Diffusion Model**: Predicts noise for the reverse diffusion process, conditioned on facial attributes through cross-attention layers
2. **Conditional Autoencoder**: Encodes images to latent space and predicts attributes, enabling latent space manipulation
3. **Diffusion Scheduler**: Handles the forward and reverse diffusion processes with configurable noise schedules

## Requirements

Install the required dependencies:

```bash
pip install -r requirements.txt
```

### Dependencies
- PyTorch >= 1.10.0
- torchvision >= 0.11.0
- numpy >= 1.19.0
- matplotlib >= 3.4.0
- tqdm >= 4.60.0
- pillow >= 8.0.0
- scipy >= 1.6.0
- scikit-learn >= 0.24.0
- tensorboard >= 2.5.0
- einops >= 0.4.0
- accelerate >= 0.12.0
- datasets >= 2.0.0

## Dataset Setup

1. Download the CelebA dataset from [the official website](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)

2. Extract the dataset with the following structure:
```
celeba/
├── img_align_celeba/
│   ├── 000001.jpg
│   ├── 000002.jpg
│   └── ...
├── list_attr_celeba.txt
├── list_eval_partition.txt
└── list_landmarks_align_celeba.txt
```

3. The dataset contains:
   - **img_align_celeba/**: 202,599 aligned and cropped face images
   - **list_attr_celeba.txt**: 40 binary attribute annotations per image
   - **list_eval_partition.txt**: Train/validation/test split information

## Usage

### Training

Train the conditional diffusion model with autoencoder:

```bash
python train.py --data_root /path/to/celeba \
                --batch_size 16 \
                --image_size 64 \
                --epochs 100 \
                --lr 2e-4 \
                --save_dir ./checkpoints \
                --log_dir ./logs
```

#### Training Arguments:
- `--data_root`: Path to CelebA dataset directory
- `--batch_size`: Training batch size (default: 16)
- `--image_size`: Image resolution for training (default: 64)
- `--epochs`: Number of training epochs (default: 100)
- `--lr`: Learning rate (default: 2e-4)
- `--channels`: UNet channel dimensions (default: [64, 128, 256, 512])
- `--latent_dim`: Autoencoder latent dimension (default: 512)
- `--cfg_dropout`: Classifier-free guidance dropout rate (default: 0.1)
- `--ae_weight`: Weight for autoencoder loss (default: 0.1)
- `--save_interval`: Save checkpoint every N epochs (default: 10)
- `--sample_interval`: Generate samples every N epochs (default: 5)

### Attribute Manipulation and Generation

Generate images with specific attributes and perform manipulations:

```bash
python manipulate.py --checkpoint ./checkpoints/best_model.pth \
                     --data_root /path/to/celeba \
                     --output_dir ./outputs \
                     --num_samples 8 \
                     --cfg_scale 2.0 \
                     --source_image /path/to/source/image.jpg
```

#### Manipulation Arguments:
- `--checkpoint`: Path to trained model checkpoint
- `--data_root`: Path to CelebA dataset (needed for attribute names)
- `--output_dir`: Directory to save generated images
- `--num_samples`: Number of samples to generate for each condition
- `--cfg_scale`: Classifier-free guidance scale (higher = more conditioning)
- `--source_image`: Source image for attribute manipulation (optional)
- `--manipulation_strength`: Strength of attribute changes (0.0-1.0)
- `--interpolation_steps`: Number of steps for attribute interpolation

## Model Architecture Details

### UNet with Cross-Attention

The UNet model includes:
- **Sinusoidal position embeddings** for timestep encoding
- **Residual blocks** with group normalization and SiLU activation
- **Cross-attention layers** for attribute conditioning
- **Self-attention blocks** in higher resolution layers
- **Skip connections** between encoder and decoder

### Conditional Autoencoder

The autoencoder consists of:
- **Encoder**: Maps images to latent distribution (μ, σ)
- **Decoder**: Reconstructs images from latent codes
- **Attribute Classifier**: Predicts facial attributes from latent codes
- **VAE formulation** with KL divergence regularization

### Diffusion Process

- **Forward process**: Gradually adds Gaussian noise to images
- **Reverse process**: Iteratively denoises using predicted noise
- **DDIM sampling**: Faster deterministic sampling with fewer steps
- **Classifier-free guidance**: Improves conditional generation quality

## Examples

### Conditional Generation

Generate images with specific attributes:

```python
# Create attribute vector for "smiling young woman"
attr_vector = create_attribute_vector(attr_dict, 
    Smiling=1, 
    Male=0, 
    Young=1
)

# Generate images
samples = model.ddim_sample(
    batch_size=4,
    image_size=(64, 64),
    attributes=attr_vector.unsqueeze(0).repeat(4, 1),
    cfg_scale=2.0,
    num_inference_steps=50
)
```

### Attribute Interpolation

Smoothly transition between attributes:

```python
# Define start and end attributes
attrs_start = {"Male": 1, "Smiling": 0}  # Serious man
attrs_end = {"Male": 0, "Smiling": 1}    # Smiling woman

# Create interpolation
interpolated = model.interpolate_between_attributes(
    create_attribute_vector(attr_dict, **attrs_start).unsqueeze(0),
    create_attribute_vector(attr_dict, **attrs_end).unsqueeze(0),
    num_steps=10
)
```

### Image Manipulation

Edit attributes of an existing image:

```python
# Load source image
source_image = load_and_preprocess_image("path/to/image.jpg")

# Define target attributes
target_attrs = create_attribute_vector(attr_dict, Smiling=1, Eyeglasses=1)

# Manipulate image
manipulated = model.manipulate_attributes(
    source_image,
    target_attrs.unsqueeze(0),
    strength=0.8
)
```

## Available Attributes

The model supports all 40 CelebA attributes:

**Appearance**: `Attractive`, `High_Cheekbones`, `Oval_Face`, `Pale_Skin`, `Young`

**Hair**: `Bald`, `Bangs`, `Black_Hair`, `Blond_Hair`, `Brown_Hair`, `Gray_Hair`, `Receding_Hairline`, `Straight_Hair`, `Wavy_Hair`

**Facial Features**: `Arched_Eyebrows`, `Bags_Under_Eyes`, `Big_Lips`, `Big_Nose`, `Bushy_Eyebrows`, `Chubby`, `Double_Chin`, `Narrow_Eyes`, `Pointy_Nose`, `Rosy_Cheeks`

**Facial Hair**: `5_o_Clock_Shadow`, `Goatee`, `Mustache`, `No_Beard`, `Sideburns`

**Accessories**: `Eyeglasses`, `Wearing_Earrings`, `Wearing_Hat`, `Wearing_Necklace`, `Wearing_Necktie`

**Makeup**: `Heavy_Makeup`, `Wearing_Lipstick`

**Expression**: `Mouth_Slightly_Open`, `Smiling`

**Gender**: `Male`

**Image Quality**: `Blurry`

## Training Tips

1. **Start with smaller images**: Train on 64x64 images first, then fine-tune on higher resolutions
2. **Monitor CFG**: Use classifier-free guidance dropout (10-20%) during training
3. **Balance losses**: Adjust autoencoder loss weight (0.05-0.2) based on reconstruction quality
4. **Learning rate**: Use cosine annealing with warmup for stable training
5. **Batch size**: Larger batches (16-32) generally produce better results
6. **Validation**: Monitor both diffusion and reconstruction losses

## Results

The trained model can:
- Generate diverse, high-quality facial images
- Conditionally generate faces with specific attributes
- Smoothly interpolate between different attribute combinations
- Edit specific attributes of real images while preserving identity
- Support classifier-free guidance for improved quality

## File Structure

```
conditional_diffusion_celeba/
├── unet_model.py           # UNet architecture with cross-attention
├── autoencoder_model.py    # Conditional autoencoder implementation
├── diffusion_model.py      # Main diffusion model combining UNet and AE
├── celeba_dataset.py       # CelebA dataset loader and preprocessing
├── train.py                # Training script
├── manipulate.py           # Attribute manipulation and generation
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Technical Details

### Loss Functions

The model optimizes a combination of losses:

1. **Diffusion Loss**: MSE between predicted and actual noise
2. **Reconstruction Loss**: MSE between original and reconstructed images
3. **KL Divergence**: Regularization for VAE latent space
4. **Attribute Classification**: Binary cross-entropy for attribute prediction

### Sampling Methods

- **DDPM**: Full denoising process (1000 steps)
- **DDIM**: Deterministic sampling with fewer steps (50-100)
- **Classifier-Free Guidance**: Improved conditional generation

### Memory Requirements

- Training: ~8-12GB GPU memory for batch size 16 on 64x64 images
- Inference: ~2-4GB GPU memory for generating 8 images

## Citation

If you use this code in your research, please cite:

```bibtex
@article{ho2020denoising,
  title={Denoising diffusion probabilistic models},
  author={Ho, Jonathan and Jain, Ajay and Abbeel, Pieter},
  journal={Advances in Neural Information Processing Systems},
  volume={33},
  pages={6840--6851},
  year={2020}
}

@inproceedings{liu2015faceattributes,
  title={Deep learning face attributes in the wild},
  author={Liu, Ziwei and Luo, Ping and Wang, Xiaogang and Tang, Xiaoou},
  booktitle={Proceedings of the IEEE international conference on computer vision},
  pages={3730--3738},
  year={2015}
}
```

## License

This project is released under the Apache License. See LICENSE file for details.

## Acknowledgments

- Based on the DDPM paper by Ho et al.
- CelebA dataset by Liu et al.
- Inspired by various diffusion model implementations and research
- Thanks to the PyTorch and open-source community
