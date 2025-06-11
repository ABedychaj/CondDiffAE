# -*- coding: utf-8 -*-
"""
CelebA dataset loader with attribute handling.
"""

import os
from typing import Optional, Tuple, Callable

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class CelebADataset(Dataset):
    """
    CelebA dataset with attribute loading and preprocessing.
    """

    def __init__(
            self,
            root_dir: str,
            attr_file: str = "list_attr_celeba.txt",
            partition_file: str = "list_eval_partition.txt",
            split: str = "train",
            transform: Optional[Callable] = None,
            target_size: Tuple[int, int] = (64, 64),
            normalize: bool = True,
    ):
        """
        Initialize CelebA dataset.
        
        Args:
            root_dir: Root directory of the CelebA dataset
            attr_file: Filename of attribute annotations
            partition_file: Filename of train/val/test partitions
            split: Dataset split ('train', 'val', or 'test')
            transform: Custom transforms to apply
            target_size: Target image size
            normalize: Whether to normalize images to [-1, 1]
        """
        self.root_dir = root_dir
        self.split = split

        # Load attributes
        attr_path = os.path.join(root_dir, attr_file)
        self.attr_df = pd.read_csv(attr_path, sep=r'\s+', skiprows=1)

        # Convert -1/1 to 0/1 for attributes
        self.attr_df.replace(-1, 0, inplace=True)

        # Load partitions
        partition_path = os.path.join(root_dir, partition_file)
        self.partition_df = pd.read_csv(partition_path, sep=r'\s+', header=None, names=["image_id", "partition"])

        # Get images for the specified split
        split_map = {"train": 0, "val": 1, "test": 2}
        self.partition_df = self.partition_df[self.partition_df["partition"] == split_map[split]]

        # Merge partition and attribute dataframes
        self.df = pd.merge(self.partition_df, self.attr_df, left_on="image_id", right_index=True, how="inner")

        # Get attribute names
        self.attr_names = list(self.attr_df.columns)[1:]
        self.num_attributes = len(self.attr_names)

        # Image transforms
        if transform is not None:
            self.transform = transform
        else:
            self.transform = transforms.Compose([
                transforms.Resize(target_size),
                transforms.ToTensor(),
            ])

            if normalize:
                self.transform = transforms.Compose([
                    self.transform,
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1]
                ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        """
        Get an item from the dataset.
        
        Returns:
            Dict with 'image', 'attributes', and 'image_id'
        """
        row = self.df.iloc[idx]
        image_id = row["image_id"]

        # Load image
        img_path = os.path.join(self.root_dir, "img_align_celeba", image_id)
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        # Get attributes
        attributes = torch.tensor(row[self.attr_names].values.astype(np.float32))

        return {
            "image": image,
            "attributes": attributes,
            "image_id": image_id
        }

    def get_attribute_names(self):
        """Get list of attribute names."""
        return self.attr_names

    def get_attribute_dict(self):
        """Get dictionary mapping attribute names to indices."""
        return {name: i for i, name in enumerate(self.attr_names)}


def create_celeba_dataloaders(
        root_dir: str,
        batch_size: int = 32,
        image_size: int = 64,
        num_workers: int = 4,
):
    """
    Create dataloaders for CelebA dataset.
    
    Args:
        root_dir: Root directory of the CelebA dataset
        batch_size: Batch size
        image_size: Target image size
        num_workers: Number of workers for data loading
        
    Returns:
        Dict with 'train', 'val', and 'test' dataloaders
    """
    # Common transforms
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # Normalize to [-1, 1]
    ])

    # Create datasets
    train_dataset = CelebADataset(
        root_dir=root_dir,
        split="train",
        transform=transform
    )

    val_dataset = CelebADataset(
        root_dir=root_dir,
        split="val",
        transform=transform
    )

    test_dataset = CelebADataset(
        root_dir=root_dir,
        split="test",
        transform=transform
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
        "attr_names": train_dataset.get_attribute_names(),
        "attr_dict": train_dataset.get_attribute_dict(),
        "num_attributes": train_dataset.num_attributes,
    }


def get_attribute_index(attr_dict, attribute_name):
    """
    Get index of an attribute by name.
    
    Args:
        attr_dict: Attribute dictionary from CelebADataset
        attribute_name: Name of the attribute
        
    Returns:
        Index of the attribute
    """
    if attribute_name not in attr_dict:
        raise ValueError(f"Attribute '{attribute_name}' not found in CelebA. "
                         f"Available attributes: {list(attr_dict.keys())}")

    return attr_dict[attribute_name]


def create_attribute_vector(attr_dict, **kwargs):
    """
    Create an attribute vector based on specified attributes.
    
    Args:
        attr_dict: Attribute dictionary from CelebADataset
        **kwargs: Attribute name/value pairs (e.g., Smiling=1, Male=0)
        
    Returns:
        Attribute vector for conditioning
    """
    attr_vector = torch.zeros(len(attr_dict))

    for attr_name, value in kwargs.items():
        idx = get_attribute_index(attr_dict, attr_name)
        attr_vector[idx] = float(value)

    return attr_vector


if __name__ == "__main__":
    # Test the dataset loader
    # Replace with your CelebA dataset path
    celeba_root = "./data/celeba"

    if os.path.exists(celeba_root):
        # Create dataloaders
        dataloaders = create_celeba_dataloaders(
            root_dir=celeba_root,
            batch_size=4,
            image_size=64,
            num_workers=0,
        )

        # Test train loader
        train_loader = dataloaders["train"]
        attr_names = dataloaders["attr_names"]
        attr_dict = dataloaders["attr_dict"]

        batch = next(iter(train_loader))

        print(f"Batch size: {batch['image'].size(0)}")
        print(f"Image shape: {batch['image'].shape}")
        print(f"Attributes shape: {batch['attributes'].shape}")
        print(f"Number of attributes: {len(attr_names)}")

        # Test attribute vector creation
        attr_vector = create_attribute_vector(
            attr_dict,
            Smiling=1,
            Male=0,
            Young=1
        )

        print(f"Attribute vector shape: {attr_vector.shape}")
        print("CelebA dataset test successful!")
    else:
        print(f"CelebA dataset not found at {celeba_root}. Please update the path.")
