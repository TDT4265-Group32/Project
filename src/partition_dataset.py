import os
import random
import shutil
from tqdm import tqdm

def partition_dataset(dataset_dir, train_ratio=0.6, val_ratio=0.2, force_repartition=False):
    print(f'Partitioning dataset in {dataset_dir} into train, val, and test subsets...\n')
    image_dir = os.path.join(dataset_dir, 'images')
    label_dir = os.path.join(dataset_dir, 'labels')

    assert train_ratio + val_ratio <= 1

    all_images = [img for img in os.listdir(image_dir) if img.endswith('.PNG')]
    all_labels = [label for label in os.listdir(label_dir) if label.endswith('.txt')]
    
    assert len(all_images) == len(all_labels), 'Number of images and labels do not match'

    random.shuffle(all_images)
    random.shuffle(all_labels)

    train_size = int(len(all_images) * train_ratio)
    val_size = int(len(all_images) * val_ratio)

    train_images = all_images[:train_size]
    val_images = all_images[train_size:train_size + val_size]
    test_images = all_images[train_size + val_size:]
    
    train_labels = [img.replace('.PNG', '.txt') for img in train_images]
    val_labels = [img.replace('.PNG', '.txt') for img in val_images]
    test_labels = [img.replace('.PNG', '.txt') for img in test_images]

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
    if train_ratio + val_ratio < 1:
        copy_files(test_images, test_labels, 'test')
