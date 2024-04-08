from utils.partition_dataset import partition_dataset
import YOLOv8

from os import path
import argparse
import json

def main(args):
    objd = YOLOv8.ObjectDetection()

    with open(args.train_config) as json_file:
        train_params = json.load(json_file)

    if path.normpath(args.dataset_path) == path.normpath('datasets/NAPLab-LiDAR'):
        # Currently, only NAPLab-LiDAR has the desired structure for the "partition_dataset" function
        partition_dataset(dataset_dir=args.dataset_path, force_repartition=False)
    objd.train(train_params)
    objd.model.export()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Script for training model.')
    parser.add_argument('--dataset_path', type=str, help='Path of the dataset')
    parser.add_argument('--reshuffle_dataset', type=bool, default=False, help='Shuffle the dataset before partitioning')
    parser.add_argument('--train_config', type=str, default='configs/YOLOv8/NAPLab-LiDAR/train.json', help='Training configuration file')

    args = parser.parse_args()
    main(args)
