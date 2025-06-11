# -*- coding: utf-8 -*-
"""
3D Shapes dataset loader with factor handling.
Adapted from CelebA dataset loader for extracted 3D Shapes JPEGs and TSV factors.
"""

import os
from typing import Optional, Tuple, Callable, Dict, List, Union

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class Shapes3DDataset(Dataset):
    """
    3D Shapes dataset with factor loading and preprocessing.
    Works with extracted JPEG images and TSV factor file.
    """

    # Factor definitions
    FACTOR_NAMES = ['floor_hue', 'wall_hue', 'object_hue', 'scale', 'shape', 'orientation']
    FACTOR_RANGES = {
        'floor_hue': (0, 1),      # 10 discrete values
        'wall_hue': (0, 1),       # 10 discrete values
        'object_hue': (0, 1),     # 10 discrete values
        'scale': (0.75, 1.25),          # 8 discrete values
        'shape': (0, 3),          # 4 discrete values (cube, sphere, cylinder, capsule)
        'orientation': (-30, 30)    # continous values
    }
    
    # Shape mappings for interpretability
    SHAPE_NAMES = {0: 'cube', 1: 'sphere', 2: 'cylinder', 3: 'capsule'}

    def __init__(
            self,
            root_dir: str,
            factors_file: str = "factors.tsv",
            images_dir: str = "images",
            split: str = "train",
            transform: Optional[Callable] = None,
            target_size: Tuple[int, int] = (64, 64),
            normalize_image: bool = True,
            normalize_factors: bool = True,
            split_ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
            random_seed: int = 42,
    ):
        """
        Initialize 3D Shapes dataset.
        
        Args:
            root_dir: Root directory containing images/ and factors.tsv
            factors_file: Filename of factor annotations (TSV format)
            images_dir: Directory name containing JPEG images
            split: Dataset split ('train', 'val', or 'test')
            transform: Custom transforms to apply
            target_size: Target image size
            normalize_image: Whether to normalize images to [-1, 1]
            normalize_factors: Whether to normalize factors to [0, 1]
            split_ratios: (train, val, test) split ratios
            random_seed: Random seed for reproducible splits
        """
        self.root_dir = root_dir
        self.images_dir = os.path.join(root_dir, images_dir)
        self.split = split
        self.normalize_factors = normalize_factors
        self.random_seed = random_seed

        # Validate directories
        if not os.path.exists(self.images_dir):
            raise FileNotFoundError(f"Images directory not found: {self.images_dir}")

        # Load factors
        factors_path = os.path.join(root_dir, factors_file)
        if not os.path.exists(factors_path):
            raise FileNotFoundError(f"Factors file not found: {factors_path}")
            
        self.factors_df = pd.read_csv(factors_path, sep='\t')
        
        # Validate factor columns
        expected_cols = ['image_id'] + self.FACTOR_NAMES
        if not all(col in self.factors_df.columns for col in expected_cols):
            raise ValueError(f"Expected columns {expected_cols}, got {list(self.factors_df.columns)}")

        # Create splits
        self._create_splits(split_ratios)
        
        # Filter dataframe for current split
        self.df = self.factors_df.iloc[self.split_indices].reset_index(drop=True)
        
        # Set up transforms
        if transform is not None:
            self.transform = transform
        else:
            transform_list = [
                transforms.Resize(target_size),
                transforms.ToTensor(),
            ]
            
            if normalize_image:
                transform_list.append(
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1]
                )
            
            self.transform = transforms.Compose(transform_list)

        self.num_factors = len(self.FACTOR_NAMES)

    def _create_splits(self, split_ratios: Tuple[float, float, float]):
        """Create reproducible train/val/test splits."""
        np.random.seed(self.random_seed)
        
        total_samples = len(self.factors_df)
        indices = np.random.permutation(total_samples)
        
        train_size = int(split_ratios[0] * total_samples)
        val_size = int(split_ratios[1] * total_samples)
        
        if self.split == "train":
            self.split_indices = indices[:train_size]
        elif self.split == "val":
            self.split_indices = indices[train_size:train_size + val_size]
        elif self.split == "test":
            self.split_indices = indices[train_size + val_size:]
        else:
            raise ValueError(f"Invalid split: {self.split}. Must be 'train', 'val', or 'test'")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        """
        Get an item from the dataset.
        
        Returns:
            Dict with 'image', 'factors', 'factors_dict', and 'image_id'
        """
        row = self.df.iloc[idx]
        image_id = row["image_id"]

        # Load image
        img_path = os.path.join(self.images_dir, image_id)
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")
            
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        # Get factors
        factors = torch.tensor(row[self.FACTOR_NAMES].values.astype(np.float32))
        
        # Normalize factors if requested
        if self.normalize_factors:
            factors = self._normalize_factors(factors)
        
        # Create factors dictionary for easy access
        factors_dict = {name: factors[i].item() for i, name in enumerate(self.FACTOR_NAMES)}

        return {
            "image": image,
            "factors": factors,
            "factors_dict": factors_dict,
            "image_id": image_id
        }

    def _normalize_factors(self, factors: torch.Tensor) -> torch.Tensor:
        """
        Normalize factors to [0, 1] range.
        
        Args:
            factors: Raw factor values
            
        Returns:
            Normalized factor values
        """
        normalized = factors.clone()
        for i, factor_name in enumerate(self.FACTOR_NAMES):
            min_val, max_val = self.FACTOR_RANGES[factor_name]
            normalized[i] = (factors[i] - min_val) / (max_val - min_val)
        return normalized

    def _denormalize_factors(self, factors: torch.Tensor) -> torch.Tensor:
        """
        Denormalize factors from [0, 1] back to original ranges.
        
        Args:
            factors: Normalized factor values
            
        Returns:
            Denormalized factor values
        """
        denormalized = factors.clone()
        for i, factor_name in enumerate(self.FACTOR_NAMES):
            min_val, max_val = self.FACTOR_RANGES[factor_name]
            denormalized[i] = factors[i] * (max_val - min_val) + min_val
        return denormalized

    def get_factor_names(self) -> List[str]:
        """Get list of factor names."""
        return self.FACTOR_NAMES.copy()

    def get_factor_dict(self) -> Dict[str, int]:
        """Get dictionary mapping factor names to indices."""
        return {name: i for i, name in enumerate(self.FACTOR_NAMES)}

    def get_factor_ranges(self) -> Dict[str, Tuple[int, int]]:
        """Get dictionary of factor ranges."""
        return self.FACTOR_RANGES.copy()
    
    def get_shape_name(self, shape_value: Union[int, float]) -> str:
        """Get shape name from shape value."""
        shape_idx = int(round(shape_value))
        return self.SHAPE_NAMES.get(shape_idx, f"unknown_{shape_idx}")

    def get_samples_by_factors(self, **factor_conditions) -> List[int]:
        """
        Get sample indices that match specified factor conditions.
        
        Args:
            **factor_conditions: Factor name/value pairs (e.g., shape=0, scale=3)
            
        Returns:
            List of sample indices matching the conditions
        """
        mask = pd.Series([True] * len(self.df))
        
        for factor_name, value in factor_conditions.items():
            if factor_name not in self.FACTOR_NAMES:
                raise ValueError(f"Unknown factor: {factor_name}")
            mask &= (self.df[factor_name] == value)
        
        return mask[mask].index.tolist()

    def get_factor_statistics(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for each factor in the current split."""
        stats = {}
        for factor_name in self.FACTOR_NAMES:
            factor_values = self.df[factor_name].values
            stats[factor_name] = {
                'min': float(factor_values.min()),
                'max': float(factor_values.max()),
                'mean': float(factor_values.mean()),
                'std': float(factor_values.std()),
                'unique_count': len(np.unique(factor_values))
            }
        return stats


def create_shapes3d_dataloaders(
        root_dir: str,
        batch_size: int = 32,
        image_size: int = 64,
        num_workers: int = 4,
        normalize_factors: bool = True,
        split_ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
        random_seed: int = 42,
):
    """
    Create dataloaders for 3D Shapes dataset.
    
    Args:
        root_dir: Root directory containing images/ and factors.tsv
        batch_size: Batch size
        image_size: Target image size
        num_workers: Number of workers for data loading
        normalize_factors: Whether to normalize factors to [0, 1]
        split_ratios: (train, val, test) split ratios
        random_seed: Random seed for reproducible splits
        
    Returns:
        Dict with 'train', 'val', 'test' dataloaders and dataset info
    """
    # Common transforms
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # Normalize to [-1, 1]
    ])

    # Create datasets
    train_dataset = Shapes3DDataset(
        root_dir=root_dir,
        split="train",
        transform=transform,
        normalize_factors=normalize_factors,
        split_ratios=split_ratios,
        random_seed=random_seed,
    )

    val_dataset = Shapes3DDataset(
        root_dir=root_dir,
        split="val",
        transform=transform,
        normalize_factors=normalize_factors,
        split_ratios=split_ratios,
        random_seed=random_seed,
    )

    test_dataset = Shapes3DDataset(
        root_dir=root_dir,
        split="test",
        transform=transform,
        normalize_factors=normalize_factors,
        split_ratios=split_ratios,
        random_seed=random_seed,
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
        "factor_names": train_dataset.get_factor_names(),
        "factor_dict": train_dataset.get_factor_dict(),
        "factor_ranges": train_dataset.get_factor_ranges(),
        "num_factors": train_dataset.num_factors,
        "train_stats": train_dataset.get_factor_statistics(),
    }


# Utility functions for factor manipulation

def get_factor_index(factor_dict: Dict[str, int], factor_name: str) -> int:
    """
    Get index of a factor by name.
    
    Args:
        factor_dict: Factor dictionary from Shapes3DDataset
        factor_name: Name of the factor
        
    Returns:
        Index of the factor
    """
    if factor_name not in factor_dict:
        raise ValueError(f"Factor '{factor_name}' not found in 3D Shapes. "
                         f"Available factors: {list(factor_dict.keys())}")
    return factor_dict[factor_name]


def create_factor_vector(factor_dict: Dict[str, int], normalize: bool = True, **kwargs) -> torch.Tensor:
    """
    Create a factor vector based on specified factors.
    
    Args:
        factor_dict: Factor dictionary from Shapes3DDataset
        normalize: Whether to normalize factors to [0, 1]
        **kwargs: Factor name/value pairs (e.g., floor_hue=5, shape=1)
        
    Returns:
        Factor vector for conditioning
    """
    factor_vector = torch.zeros(len(factor_dict))

    for factor_name, value in kwargs.items():
        idx = get_factor_index(factor_dict, factor_name)
        factor_vector[idx] = float(value)
        
        # Normalize if requested
        if normalize and factor_name in Shapes3DDataset.FACTOR_RANGES:
            min_val, max_val = Shapes3DDataset.FACTOR_RANGES[factor_name]
            factor_vector[idx] = (factor_vector[idx] - min_val) / (max_val - min_val)

    return factor_vector


def interpolate_factors(factor_vector1: torch.Tensor, factor_vector2: torch.Tensor, 
                       num_steps: int = 10) -> torch.Tensor:
    """
    Create interpolation between two factor vectors.
    
    Args:
        factor_vector1: Starting factor vector
        factor_vector2: Ending factor vector
        num_steps: Number of interpolation steps
        
    Returns:
        Tensor of shape (num_steps, num_factors) with interpolated factors
    """
    alphas = torch.linspace(0, 1, num_steps).unsqueeze(1)
    factor_vector1 = factor_vector1.unsqueeze(0)
    factor_vector2 = factor_vector2.unsqueeze(0)
    
    interpolated = (1 - alphas) * factor_vector1 + alphas * factor_vector2
    return interpolated


def get_factor_combinations(factor_dict: Dict[str, int], **fixed_factors) -> List[torch.Tensor]:
    """
    Get all possible combinations for non-fixed factors.
    
    Args:
        factor_dict: Factor dictionary from Shapes3DDataset  
        **fixed_factors: Fixed factor values
        
    Returns:
        List of factor vectors with all combinations
    """
    combinations = []
    
    # Get variable factors
    variable_factors = [name for name in Shapes3DDataset.FACTOR_NAMES if name not in fixed_factors]
    
    # Generate all combinations (simplified for demonstration)
    # In practice, you might want to limit this or use sampling
    for shape in range(4):  # Example: iterate through all shapes
        factor_vector = create_factor_vector(factor_dict, normalize=True, shape=shape, **fixed_factors)
        combinations.append(factor_vector)
    
    return combinations


if __name__ == "__main__":
    # Test the dataset loader
    # Replace with your extracted 3D Shapes dataset path
    shapes3d_root = "./data/3d-shapes"

    if os.path.exists(shapes3d_root):
        # Create dataloaders
        dataloaders = create_shapes3d_dataloaders(
            root_dir=shapes3d_root,
            batch_size=4,
            image_size=64,
            num_workers=0,
        )

        # Test train loader
        train_loader = dataloaders["train"]
        factor_names = dataloaders["factor_names"]
        factor_dict = dataloaders["factor_dict"]
        factor_ranges = dataloaders["factor_ranges"]

        print(f"Dataset loaded successfully!")
        print(f"Factors: {factor_names}")
        print(f"Factor ranges: {factor_ranges}")
        print(f"Train samples: {len(train_loader.dataset)}")
        print(f"Val samples: {len(dataloaders['val'].dataset)}")
        print(f"Test samples: {len(dataloaders['test'].dataset)}")

        # Test batch loading
        batch = next(iter(train_loader))
        print(f"\nBatch info:")
        print(f"Image shape: {batch['image'].shape}")
        print(f"Factors shape: {batch['factors'].shape}")
        print(f"Sample factors: {batch['factors'][0]}")
        print(f"Sample factors dict: {batch['factors_dict']}")

        # Test factor vector creation
        factor_vector = create_factor_vector(
            factor_dict,
            normalize=True,
            floor_hue=5,
            wall_hue=3,
            shape=1,  # sphere
            scale=4
        )
        print(f"\nCreated factor vector: {factor_vector}")

        # Test interpolation
        factor_vector2 = create_factor_vector(
            factor_dict,
            normalize=True,
            floor_hue=8,
            wall_hue=7,
            shape=2,  # cylinder
            scale=6
        )
        
        interpolated = interpolate_factors(factor_vector, factor_vector2, num_steps=5)
        print(f"\nInterpolation shape: {interpolated.shape}")
        print(f"First interpolation step: {interpolated[0]}")
        print(f"Last interpolation step: {interpolated[-1]}")

        print("\n3D Shapes dataset test successful!")
    else:
        print(f"3D Shapes dataset not found at {shapes3d_root}.")
        print("Please run the extraction script first to create the JPEG dataset.")