import random
import os
import argparse
import json
import time
from math import ceil

from codecarbon import EmissionsTracker

import FasterRCNN
from YOLOv8.CustomYOLO import CustomYOLO as YOLO
from tools.partition_dataset import partition_dataset, partition_video_dataset
from tools.png_to_video import create_video

def main(args):
    ARCHITECTURE = args.type
    MODE = args.mode
    MODEL = args.model_path

    assert ARCHITECTURE in ['YOLOv8', 'FasterRCNN'], 'Invalid architecture. Please choose from: YOLOv8, FasterRCNN'
    assert MODE in ['train', 'val', 'pred', 'bench'], 'Invalid mode. Please choose from: train, validate, predict'

    # Load the configuration file
    match ARCHITECTURE:
        case 'YOLOv8':
            JSON_PATH = os.path.join('configs', 'YOLOv8', MODE + '.yaml')
            with open(JSON_PATH) as json_config_file:
                CONFIG_JSON = json.load(json_config_file)

            # Use custom YOLO model
            model = YOLO(MODEL)
            # Load parameters to be passed onto train, validate, or predict functions
            PARAMS = CONFIG_JSON['params']

            if MODE == 'train':
                # Set the loss function parameters
                model.set_dfl(CONFIG_JSON['loss_function']['use_dfl'])
                model.set_iou_method('giou', CONFIG_JSON['loss_function']['use_giou'])
                model.set_iou_method('diou', CONFIG_JSON['loss_function']['use_diou'])
                model.set_iou_method('ciou', CONFIG_JSON['loss_function']['use_ciou'])

                # Initialize the emissions tracker
                tracker = EmissionsTracker()
                tracker.start()

                # Partition the dataset into seperate training and validation folders
                partition_dataset()
                model.train(PARAMS)

                # Stop the timer and finalize the power consumption
                tracker.stop()

            elif MODE == 'val':

                model.validate(val_params=PARAMS)

            elif MODE == 'pred':

                model.predict(PARAMS)

                # Only sensible to create video from sequence of PNGs for NAPLab-LiDAR dataset
                if CONFIG_JSON['video']['create_video'] and DATASET == 'NAPLab-LiDAR':

                    # Create path if it doesn't exist
                    if not os.path.exists(CONFIG_JSON['video']['path']):
                        os.makedirs(CONFIG_JSON['video']['path'])

                    # Create video from sequence of PNGs
                    create_video(os.path.join('results', DATASET),
                                filename=os.path.join(CONFIG_JSON['video']['path'],
                                                    CONFIG_JSON['video']['filename']),
                                extension=CONFIG_JSON['video']['extension'],
                                fps=CONFIG_JSON['video']['fps'])
            elif MODE == 'bench':
                model_path = CONFIG_JSON['model_path']
                
                model.benchmark(PARAMS)

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