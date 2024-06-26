import random
import os
import argparse
import yaml
import atexit

import torch
import lightning.pytorch as pl
from codecarbon import EmissionsTracker
from lightning.pytorch.loggers import WandbLogger, TensorBoardLogger
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pathlib import Path

from FasterRCNN.trainer import CustomFasterRCNN
from FasterRCNN.datamodule import CustomDataModule

from YOLOv8.CustomYOLO import CustomYOLO as YOLO
from tools.data_partitioner import partition_dataset, create_test_dataset
from tools.video_formatter import create_video
from tools.dataloader import extract_dataset, export_data

def main(args):
    ARCHITECTURE = args.arch 
    MODE = args.mode
    MODEL_PATH = args.model_path

    # Extract the dataset and partition it into training and validation sets
    extract_dataset()
    create_test_dataset()
    partition_dataset()

    assert ARCHITECTURE in ['YOLOv8', 'FasterRCNN'], 'Invalid architecture. Please choose from: YOLOv8, FasterRCNN'
    assert MODE in ['train', 'val', 'pred', 'export', 'all'], 'Invalid mode. Please choose from: train, validate, predict, export'

    match ARCHITECTURE:
        case 'YOLOv8':
            # Set the seed for reproducibility
            random.seed(0)
            # Load the configuration file
            YAML_PATH = os.path.join('configs', 'YOLOv8', 'config.yaml')
            with open(YAML_PATH) as yaml_config_file:
                CONFIG_YAML = yaml.safe_load(yaml_config_file)
            # Load the configuration parameters
            TRAIN_CONFIG = CONFIG_YAML['train']
            VAL_CONFIG = CONFIG_YAML['val']
            PRED_CONFIG = CONFIG_YAML['pred']
            EXPORT_CONFIG = CONFIG_YAML['export']
            # Select the mode to run the script in
            match MODE:
                case 'all':
                    # Check if a model path is provided
                    if MODEL_PATH is None:
                        MODEL_PATH = TRAIN_CONFIG['model_path']
                    # Load model and parameters
                    model = YOLO(MODEL_PATH)

                    # Perform training, validation, prediction, and export
                    model.train(TRAIN_CONFIG['params'], TRAIN_CONFIG['loss_function'])
                    model.validate(VAL_CONFIG['params'])
                    model.predict(PRED_CONFIG['params'])
                    if PRED_CONFIG['video']['create_video']:
                        create_video(os.path.join('results', ARCHITECTURE),
                                     **PRED_CONFIG['video'])
                    model.export(EXPORT_CONFIG['params'])
                    export_data(ARCHITECTURE)

                case 'train':
                    # Partition the dataset into random seperate training and validation folders
                    partition_dataset()
                    model.train(TRAIN_CONFIG['params'], TRAIN_CONFIG['loss_function'])

                case 'val':
                    # Validate the model on the test set
                    model.validate(VAL_CONFIG['params'])

                case 'pred':
                    # Perform prediction on the test set
                    model.predict(PRED_CONFIG['params'])
                    if PRED_CONFIG['video']['create_video']:
                        # Create video from sequence of PNGs
                        create_video(os.path.join('results', ARCHITECTURE),
                                     **PRED_CONFIG['video'])
                case 'export':
                    # Export the model to a specified format
                    model.export(EXPORT_CONFIG['params'])
                    export_data(ARCHITECTURE)

        case 'FasterRCNN':
            # Load the configuration file
            YAML_PATH = os.path.join('configs', 'FasterRCNN', 'faster_rcnn_config.yaml')
            with open(YAML_PATH) as yaml_config_file:
                CONFIG_YAML = yaml.safe_load(yaml_config_file)
            # Load the configuration parameters
            MODULE_CONFIG = CONFIG_YAML['module_config']
            CHECKPOINT_PATH = CONFIG_YAML['checkpoint_path']
            TEST_MODEL = CONFIG_YAML['test_model']
            CUSTOMDATAMODULE = CONFIG_YAML['custom_data_module']
            TRAINER_CONFIG = CONFIG_YAML['trainer']
            LOGGER = CONFIG_YAML['logger']
            CALLBACKS = CONFIG_YAML['callbacks']
            
            if MODE != 'all':
                Warning("'all' is the only mode supported for FasterRCNN")
                MODE = 'all'

            # Select the mode to run the script in
            match MODE:
                case 'all':
                    # Set the precision for the model
                    torch.set_float32_matmul_precision('medium')
                    # Set the seed for reproducibility
                    pl.seed_everything(42)
                    # Load the custom data module
                    dm = CustomDataModule(**CUSTOMDATAMODULE)
                    # Load the model
                    if CHECKPOINT_PATH:
                        model = CustomFasterRCNN.load_from_checkpoint(CHECKPOINT_PATH, config=MODULE_CONFIG)
                        print("Loading weights from checkpoint...")
                    else:
                        model = CustomFasterRCNN(MODULE_CONFIG)
                    # Initialize the logger
                    if LOGGER['type'].lower() == 'wandb':
                        logger = WandbLogger(project=LOGGER['project'],
                                             name=LOGGER['name'])
                    elif LOGGER['type'].lower() == 'tensorboard':
                        logger = TensorBoardLogger(save_dir=LOGGER['save_dir'],
                                                   name=LOGGER['name'])
                    else:
                        raise ValueError("Invalid logger type. Please choose from: Wandb, Tensorboard")
                    # Initialize the trainer
                    trainer = pl.Trainer(
                        **TRAINER_CONFIG,
                        logger=logger,
                        callbacks=[
                            EarlyStopping(**CALLBACKS['early_stopping']),
                            LearningRateMonitor(**CALLBACKS['learning_rate_monitor']),
                            ModelCheckpoint(**CALLBACKS['model_checkpoint']),
                        ])

                    # Train the model
                    if not TEST_MODEL:
                        tracker = EmissionsTracker()
                        tracker.start()
                        atexit.register(tracker.stop)
                        trainer.fit(model, train_dataloaders=dm.train_dataloader(), val_dataloaders=dm.val_dataloader())
                        tracker.stop()
                        atexit.unregister(tracker.stop)

                    # trainer.test(model, dataloaders=dm.train_dataloader())

                    pred = trainer.predict(model, dataloaders=dm.test_dataloader(), return_predictions=True)


if __name__ == "__main__":
    # Parse the arguments
    parser = argparse.ArgumentParser(description='Script for training model.', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--arch', type=str, 
                        help='Type of architecture to use for training the model \
                            \nOptions: YOLOv8, FasterRCNN')
    parser.add_argument('--mode', type=str, default='all',
                        help='Mode to run the script in \
                            \nOptions: train, val, pred, export, all')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to the model weights file, e.g. yolov8l.pt')
    args = parser.parse_args()
    main(args)