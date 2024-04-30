import os
import random
import shutil
from tqdm import tqdm

import pathlib
import yaml
import pandas as pd
import collections
from sklearn.model_selection import KFold

def _copy_files(images,
                labels,
                subset,
                dataset_path):
        img_subset_dir = os.path.join(dataset_path, 'images', subset)
        lbl_subset_dir = os.path.join(dataset_path, 'labels', subset)
        
        print(f'Copying {subset} images into {img_subset_dir}...')
        if os.path.exists(img_subset_dir):
            shutil.rmtree(img_subset_dir)
        os.makedirs(img_subset_dir, exist_ok=True)
        for image in tqdm(images, desc=f'Copying {subset} images'):
            shutil.copy(image, img_subset_dir)

        print(f'Copying {subset} labels into {lbl_subset_dir}...')
        if os.path.exists(lbl_subset_dir):
            shutil.rmtree(lbl_subset_dir)
        os.makedirs(lbl_subset_dir, exist_ok=True)
        for label in tqdm(labels, desc=f'Copying {subset} labels'):
            shutil.copy(label, lbl_subset_dir)

def partition_dataset(train_ratio: float = 0.8):
    """Divide a dataset into training, validation, and test subsets.

    Args:
        dataset (str): Which dataset to partition
        train_ratio (float, optional): _description_. Defaults to 0.8, val_ratio is the complement of train_ratio.
    """
    DATASET = 'NAPLab-LiDAR'
    
    print(f'Partitioning dataset in {DATASET} into train and val subsets...\n')
    try:
        dataset_path = pathlib.Path(f'datasets/{DATASET}')
    except FileNotFoundError:
        raise Exception(f"Dataset {DATASET} not found")

    assert train_ratio < 1, 'Train ratio must be less than 1'
    assert train_ratio > 0, 'Train ratio must be greater than 0'
    
    labels = sorted(dataset_path.glob("labels/*.txt"))

    random.shuffle(labels)

    images = [label.parent.parent / 'images' / (label.stem + '.PNG') for label in labels]

    train_labels = labels[:int(len(labels) * train_ratio)]
    val_labels = labels[int(len(labels) * train_ratio):]
    train_images = images[:int(len(images) * train_ratio)]
    val_images = images[int(len(images) * train_ratio):]

    _copy_files(train_images, train_labels, 'train', dataset_path)
    _copy_files(val_images, val_labels, 'val', dataset_path)

def partition_video_dataset(num_clips: int,
                            num_val: int = 1,
                            seed: int = None):
    """
    Function to partition a video dataset into training and validation subsets.
    
    
    Args:
    dataset (str): Name of the dataset
    num_clips (int): Number of clips in the dataset
    num_val (int): Number of validation clips
    seed (int): Random seed for reproducibility
    """
    DATASET = 'NAPLab-LiDAR'
    
    assert num_val < num_clips, 'Number of validation clips must be less than total number of clips'
    
    try:
        dataset_path = pathlib.Path(f'datasets/{DATASET}')
    except FileNotFoundError:
        raise Exception(f"Dataset {DATASET} not found")

    # Extract labels and names from the dataset
    labels = sorted(dataset_path.glob("labels/*.txt"))

    if DATASET == 'NAPLab-LiDAR':
        assert num_clips == 18, 'NAPLab-LiDAR dataset must be split into 18 folds'
        # test frames are 201 - 301, assumes test frames removed from dataset
        video_labels = []
        start = 0
        for i in range(num_clips):
            # Some segments have 101 frames instead of 100
            end = start + 101 if i in {1, 2, 3, 4} else start + 100
            video_labels.append(labels[start:end])
            start = end

        if seed is not None:
            random.seed(seed)
        random.shuffle(video_labels)
        
        train_labels = video_labels[num_val:]
        val_labels = video_labels[:num_val]
    else:
        raise NotImplementedError('Function not yet implemented for this dataset')

    # Flatten the list of lists
    train_labels = [label for sublist in train_labels for label in sublist]
    val_labels = [label for sublist in val_labels for label in sublist]
    train_imgs = [label.parent.parent / 'images' / (label.stem + '.PNG') for label in train_labels]
    val_imgs = [label.parent.parent / 'images' / (label.stem + '.PNG') for label in val_labels]

    _copy_files(train_imgs, train_labels, 'train', dataset_path)
    _copy_files(val_imgs, val_labels, 'val', dataset_path)

if __name__ == "__main__":
    partition_dataset('NAPLab-LiDAR')