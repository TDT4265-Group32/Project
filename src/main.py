import random
import os
import argparse
import yaml
from math import ceil

from codecarbon import EmissionsTracker

import FasterRCNN
from YOLOv8.CustomYOLO import CustomYOLO as YOLO
from tools.partition_dataset import partition_dataset
from tools.png_to_video import create_video
from tools.load_dataset import copy_datasets

def main(args):
    ARCHITECTURE = args.type
    MODE = args.mode
    MODEL = args.model_path
    # Only dataset used in the project
    DATASET = 'NAPLab-LiDAR'
    copy_datasets()

    assert ARCHITECTURE in ['YOLOv8', 'FasterRCNN'], 'Invalid architecture. Please choose from: YOLOv8, FasterRCNN'
    assert MODE in ['train', 'val', 'pred', 'bench'], 'Invalid mode. Please choose from: train, validate, predict'

    match ARCHITECTURE:
        case 'YOLOv8':
            # Load the configuration file
            YAML_PATH = os.path.join('configs', 'YOLOv8', MODE + '.yaml')
            with open(YAML_PATH) as yaml_config_file:
                CONFIG_YAML = yaml.safe_load(yaml_config_file)

            # Use custom YOLO model
            model = YOLO(MODEL)
            # Load parameters to be passed onto train, validate, or predict functions
            PARAMS = CONFIG_YAML['params']

            match MODE:
                case 'train':
                    # Set the loss function parameters
                    model.set_dfl(CONFIG_YAML['loss_function']['use_dfl'])
                    model.set_iou_method('giou', CONFIG_YAML['loss_function']['use_giou'])
                    model.set_iou_method('diou', CONFIG_YAML['loss_function']['use_diou'])
                    model.set_iou_method('ciou', CONFIG_YAML['loss_function']['use_ciou'])

                    # Initialize the emissions tracker
                    tracker = EmissionsTracker()
                    tracker.start()

                    # Partition the dataset into seperate training and validation folders
                    partition_dataset()
                    model.train(PARAMS)

                    # Stop the timer and finalize the power consumption
                    tracker.stop()
                case 'val':

                    model.validate(val_params=PARAMS)

                case 'pred':

                    model.predict(PARAMS)
                    if CONFIG_YAML['video']['create_video'] and DATASET == 'NAPLab-LiDAR':
                        # Create path if it doesn't exist
                        if not os.path.exists(CONFIG_YAML['video']['path']):
                            os.makedirs(CONFIG_YAML['video']['path'])

                    # Create video from sequence of PNGs
                    create_video(os.path.join('results', DATASET),
                                filename=os.path.join(CONFIG_YAML['video']['path'],
                                                    CONFIG_YAML['video']['filename']),
                                extension=CONFIG_YAML['video']['extension'],
                                fps=CONFIG_YAML['video']['fps'])
                case 'bench':
                    # Benchmark the model
                    # NOTE: This attempts to export model to all available formats
                    model.benchmark(PARAMS)

        case 'FasterRCNN':
            raise NotImplementedError('Insert FasterRCNN code here')

if __name__ == "__main__":
    # Make results "deterministic"
    random.seed(0)
    # Parse the arguments
    parser = argparse.ArgumentParser(description='Script for training model.', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--type', type=str, 
                        help='Type of architecture to use for training the model \
                            \nOptions: YOLOv8, FasterRCNN')
    parser.add_argument('--mode', type=str, 
                        help='Mode to run the script in \
                            \nOptions: train, val, pred, bench')
    parser.add_argument('--model_path', type=str, 
                        help='Path to the model weights file, e.g. yolov8l.pt')
    args = parser.parse_args()
    main(args)