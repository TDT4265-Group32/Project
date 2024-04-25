import os
import random
import shutil
from tqdm import tqdm

import pathlib
import yaml
import pandas as pd
import collections
from sklearn.model_selection import KFold

def partition_dataset(dataset, train_ratio=0.8, val_ratio=0.2, force_repartition=False):
    print(f'Partitioning dataset in {dataset} into train, val, and test subsets...\n')
    dataset_path = f'datasets/{dataset}'
    image_dir = os.path.join(dataset_path, 'images')
    label_dir = os.path.join(dataset_path, 'labels')

    assert train_ratio + val_ratio == 1

    all_images = [img for img in os.listdir(image_dir) if img.endswith('.PNG')]
    all_labels = [label for label in os.listdir(label_dir) if label.endswith('.txt')]
    
    assert len(all_images) == len(all_labels), 'Number of images and labels do not match'

    random.shuffle(all_images)
    random.shuffle(all_labels)

    train_size = int(len(all_images) * train_ratio)
    val_size = int(len(all_images) * val_ratio)

    train_images = all_images[:train_size]
    val_images = all_images[train_size:train_size + val_size]
    
    train_labels = [img.replace('.PNG', '.txt') for img in train_images]
    val_labels = [img.replace('.PNG', '.txt') for img in val_images]

    def copy_files(images, labels, subset):
        img_subset_dir = os.path.join(image_dir, subset)
        lbl_subset_dir = os.path.join(label_dir, subset)
        if not force_repartition and os.path.exists(img_subset_dir) and os.path.exists(lbl_subset_dir):
            print(f'{img_subset_dir} and {lbl_subset_dir} already exists, skipping...')
            return

        print(f'Copying {subset} images into {img_subset_dir}...')
        if os.path.exists(img_subset_dir):
            shutil.rmtree(img_subset_dir)
        os.makedirs(img_subset_dir, exist_ok=True)
        for image in tqdm(images, desc=f'Copying {subset} images'):
            shutil.copy(os.path.join(image_dir, image), os.path.join(img_subset_dir, image))

        print(f'Copying {subset} labels into {lbl_subset_dir}...')
        if os.path.exists(lbl_subset_dir):
            shutil.rmtree(lbl_subset_dir)
        os.makedirs(lbl_subset_dir, exist_ok=True)
        for label in tqdm(labels, desc=f'Copying {subset} labels'):
            shutil.copy(os.path.join(label_dir, label), os.path.join(lbl_subset_dir, label))

    copy_files(train_images, train_labels, 'train')
    copy_files(val_images, val_labels, 'val')

def partition_video_dataset(dataset: str,
                            num_clips: int,
                            num_val: int = 1):
    
    assert num_val < num_clips, 'Number of validation clips must be less than total number of clips'
    
    try:
        dataset_path = pathlib.Path(f'datasets/{dataset}')
    except FileNotFoundError:
        raise Exception(f"Dataset {dataset} not found")

    # Extract labels and names from the dataset
    labels = sorted(dataset_path.glob("labels/*.txt"))

    if dataset == 'NAPLab-LiDAR':
        assert num_clips == 18, 'NAPLab-LiDAR dataset must be split into 18 folds'
        # test frames are 201 - 301, assumes test frames removed from dataset
        video_labels = []
        start = 0
        for i in range(num_clips):
            # Some segments have 101 frames instead of 100
            end = start + 101 if i in {1, 2, 3, 4} else start + 100
            video_labels.append(labels[start:end])
            start = end

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
    
    def copy_files(images, labels, subset):
        img_subset_dir = os.path.join(dataset_path, 'images', subset)
        lbl_subset_dir = os.path.join(dataset_path, 'labels', subset)
        
        print(f'Copying {subset} images into {img_subset_dir}...')
        if os.path.exists(img_subset_dir):
            # Prune the directory
            for filename in os.listdir(img_subset_dir):
                file_path = os.path.join(img_subset_dir, filename)
                if file_path not in images:
                    os.remove(file_path)
        else:
            os.makedirs(img_subset_dir, exist_ok=True)
        for image in tqdm(images, desc=f'Copying {subset} images'):
            shutil.copy(image, img_subset_dir)

        print(f'Copying {subset} labels into {lbl_subset_dir}...')
        if os.path.exists(lbl_subset_dir):
            shutil.rmtree(lbl_subset_dir)
        os.makedirs(lbl_subset_dir, exist_ok=True)
        for label in tqdm(labels, desc=f'Copying {subset} labels'):
            shutil.copy(label, lbl_subset_dir)

    copy_files(train_imgs, train_labels, 'train')
    copy_files(val_imgs, val_labels, 'val')

    

def kfold_crossval_partition(dataset: str, 
                             k: int, 
                             seed: int = 0):
    """Function to partition dataset into k-folds for cross-validation.
    Assumes YOLOv8 dataset structure.
    
    Args:
    dataset_dir (str): Path to the dataset directory
    yaml_dir (str): Path to the YAML file containing the class names
    k (int): Number of folds
    seed (int): Random seed for reproducibility
    
    """

    # Try to load the dataset and YAML file
    try:
        dataset_path = pathlib.Path(f'datasets/{dataset}')
        yaml_path = f'data/{dataset}.yaml'
    except FileNotFoundError:
        raise Exception(f"Dataset {dataset} not found")

    # Extract labels and names from the dataset
    labels = sorted(dataset_path.glob("labels/*.txt"))
        
    with open(yaml_path, 'r', encoding='utf-8') as f:
        classes = yaml.safe_load(f)['names']

    cls_idx = sorted(classes.keys())

    # Create a DataFrame to store number of a class in each fold
    indx = [label.stem for label in labels]
    labels_df = pd.DataFrame([], columns=cls_idx, index=indx)

    for label in labels:
        lbl_counter = collections.Counter()
        
        # Open the label file and count the number of each class in the file
        with open(label, 'r') as f:
            lines = f.readlines()

        # Increment the counter for each class in the label file
        for line in lines:
            lbl_counter[int(line.split(' ')[0])] += 1

        # Insert the counter into the DataFrame
        labels_df.loc[label.stem] = lbl_counter

    # Fill NaN values with 0
    labels_df = labels_df.fillna(0)

    ### K-Fold Splitting ###
    
    kf = KFold(n_splits=k, random_state=seed, shuffle=True)
    
    kfolds = list(kf.split(labels_df))
    
    folds = [f'split_{i}' for i in range(1, k+1)]
    folds_df = pd.DataFrame(index=indx, columns=folds)
    
    for idx, (train, val) in enumerate(kfolds, start=1):
        folds_df[f'split_{idx}'].loc[labels_df.iloc[train].index] = 'train'
        folds_df[f'split_{idx}'].loc[labels_df.iloc[val].index] = 'val'

    fold_lbl_distrb = pd.DataFrame(index=folds, columns=cls_idx)
    
    for n, (train_idx, val_idx) in enumerate(kfolds, start=1):
        train_totals = labels_df.iloc[train_idx].sum()
        val_totals = labels_df.iloc[val_idx].sum()
        
        ratio = val_totals / (train_totals + 1E-7)
        fold_lbl_distrb.loc[f'split_{n}'] = ratio
        
    raise NotImplementedError('Function not yet completed')

if __name__ == "__main__":
    # Temporary test code
    partition_video_dataset('NAPLab-LiDAR', 18, 1)

    
    
        
        
    
    
    
    