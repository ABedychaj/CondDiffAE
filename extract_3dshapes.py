import os
import h5py
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import argparse


def extract_3dshapes_dataset(hdf5_path, output_dir='./3dshapes_extracted',
                             image_size=(64, 64), batch_size=1000):
    """
    Extract full 3D Shapes dataset to JPEGs and TSV file
    """
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    images_dir = os.path.join(output_dir, 'images')
    os.makedirs(images_dir, exist_ok=True)

    # Open HDF5 file
    with h5py.File(hdf5_path, 'r') as f:
        images = f['images']
        labels = f['labels']
        total_samples = images.shape[0]

        # Create TSV file
        tsv_path = os.path.join(output_dir, 'factors.tsv')
        factor_names = ['floor_hue', 'wall_hue', 'object_hue', 'scale', 'shape', 'orientation']

        with open(tsv_path, 'w') as tsvfile:
            # Write header
            tsvfile.write('\t'.join(['image_id'] + factor_names) + '\n')

            # Process in batches
            for start_idx in tqdm(range(0, total_samples, batch_size),
                                  desc="Extracting dataset"):
                end_idx = min(start_idx + batch_size, total_samples)

                # Get batch data
                image_batch = images[start_idx:end_idx]
                label_batch = labels[start_idx:end_idx]

                # Process each sample
                for batch_idx, (image, factors) in enumerate(zip(image_batch, label_batch)):
                    global_idx = start_idx + batch_idx

                    # Save image
                    img = Image.fromarray(image)
                    img_path = os.path.join(images_dir, f'image_{global_idx:06d}.jpg')
                    img.save(img_path, 'JPEG', quality=95)

                    # Write to TSV
                    tsv_row = [f'image_{global_idx:06d}.jpg'] + [str(f) for f in factors]
                    tsvfile.write('\t'.join(tsv_row) + '\n')

        # Create dataset summary
        summary = f"""3D Shapes Dataset Extraction Summary
Total images: {total_samples:,}
Image size: {image_size[0]}x{image_size[1]}
Factors: {', '.join(factor_names)}
"""
        with open(os.path.join(output_dir, 'dataset_summary.txt'), 'w') as f:
            f.write(summary)

    print(f"Extraction complete! Results saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract 3D Shapes dataset')
    parser.add_argument('--hdf5_path', required=True, help='Path to 3dshapes.h5')
    parser.add_argument('--output_dir', default='./3dshapes_extracted',
                        help='Output directory')
    args = parser.parse_args()

    extract_3dshapes_dataset(args.hdf5_path, args.output_dir)
