import os
import random
import shutil
from tqdm import tqdm

def partition_dataset(dataset_dir, train_ratio=0.6, val_ratio=0.2, force_repartition=False):
    print(f'Partitioning dataset in {dataset_dir} into train, val, and test subsets...\n')
    image_dir = os.path.join(dataset_dir, 'images')

    assert train_ratio + val_ratio <= 1

    all_images = [img for img in os.listdir(image_dir) if img.endswith('.PNG')]
    random.shuffle(all_images)

    train_size = int(len(all_images) * train_ratio)
    val_size = int(len(all_images) * val_ratio)

    train_images = all_images[:train_size]
    val_images = all_images[train_size:train_size + val_size]
    test_images = all_images[train_size + val_size:]

    def copy_images(images, subset):
        subset_dir = os.path.join(image_dir, subset)
        if not force_repartition and os.path.exists(subset_dir):
            print(f'{subset_dir} already exists, skipping...')
            return
        print(f'Copying {subset} images into {subset_dir}...')
        if os.path.exists(subset_dir):
            shutil.rmtree(subset_dir)
        os.makedirs(subset_dir, exist_ok=True)
        for image in tqdm(images, desc=f'Copying {subset} images'):
            shutil.copy(os.path.join(image_dir, image), os.path.join(subset_dir, image))

    copy_images(train_images, 'train')
    copy_images(val_images, 'val')
    if train_ratio + val_ratio < 1:
        copy_images(test_images, 'test')
