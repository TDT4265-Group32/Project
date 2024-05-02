import random
import os
import argparse
import yaml

import torch
import lightning.pytorch as pl
import munch
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pathlib import Path

from FasterRCNN.trainer import CustomFasterRCNN
from FasterRCNN.datamodule import CustomDataModule

from YOLOv8.CustomYOLO import CustomYOLO as YOLO
from tools.data_partitioner import partition_dataset
from tools.video_formatter import create_video
from tools.dataloader import extract_dataset, export_data

def main(args):
    ARCHITECTURE = 'FasterRCNN' ##args.arch
    MODE = args.mode
    extract_dataset()
    partition_dataset()

    assert ARCHITECTURE in ['YOLOv8', 'FasterRCNN'], 'Invalid architecture. Please choose from: YOLOv8, FasterRCNN'
    assert MODE in ['train', 'val', 'pred', 'export', 'all'], 'Invalid mode. Please choose from: train, validate, predict, export'

    match ARCHITECTURE:
        case 'YOLOv8':
            # Load the configuration file
            if MODE == 'all':
                # Train model from scratch when all is selected
                YAML_PATH = os.path.join('configs', 'YOLOv8', 'train.yaml')
            else:
                YAML_PATH = os.path.join('configs', 'YOLOv8', MODE + '.yaml')

            with open(YAML_PATH) as yaml_config_file:
                CONFIG_YAML = yaml.safe_load(yaml_config_file)

            # Check if a model path is provided
            if args.model_path is not None:
                MODEL = args.model_path
            else:
                MODEL = CONFIG_YAML['model_path']

            # Load model and parameters
            model = YOLO(MODEL)
            PARAMS = CONFIG_YAML['params']

            match MODE:
                case 'all':
                    # Load all configuration files
                    TRAIN_YAML = os.path.join('configs', 'YOLOv8', 'train.yaml')
                    VAL_YAML = os.path.join('configs', 'YOLOv8', 'val.yaml')
                    PRED_YAML = os.path.join('configs', 'YOLOv8', 'pred.yaml')
                    EXPORT_YAML = os.path.join('configs', 'YOLOv8', 'export.yaml')

                    CONFIGS = {TRAIN_YAML: None, VAL_YAML: None, PRED_YAML: None, EXPORT_YAML: None}
                    
                    for key in CONFIGS:

                        with open(key) as yaml_config_file:
                            CONFIGS[key] = yaml.safe_load(yaml_config_file)

                    # Perform training, validation, prediction, and export
                    model.train(CONFIGS[TRAIN_YAML]['params'], CONFIGS[TRAIN_YAML]['loss_function'])

                    model.validate(CONFIGS[VAL_YAML]['params'])

                    model.predict(CONFIGS[PRED_YAML]['params'])
                    if CONFIGS[PRED_YAML]['video']['create_video']:
                        # Create video from sequence of PNGs
                        create_video(os.path.join('results', ARCHITECTURE),
                                     dst_path=CONFIGS[PRED_YAML]['video']['path'],
                                     filename=CONFIGS[PRED_YAML]['video']['filename'],
                                     extension=CONFIGS[PRED_YAML]['video']['extension'],
                                     fps=CONFIGS[PRED_YAML]['video']['fps'])

                    model.export(CONFIGS[EXPORT_YAML]['params'])
                    export_data(ARCHITECTURE)

                case 'train':
                    # Partition the dataset into random seperate training and validation folders
                    partition_dataset()
                    model.train(PARAMS, CONFIG_YAML['loss_function'])

                case 'val':
                    # Validate the model on the test set
                    model.validate(val_params=PARAMS)

                case 'pred':
                    # Perform prediction on the test set
                    model.predict(PARAMS)
                    if CONFIG_YAML['video']['create_video']:
                        # Create video from sequence of PNGs
                        create_video(os.path.join('results', ARCHITECTURE),
                                     dst_path=CONFIG_YAML['video']['path'],
                                     filename=CONFIG_YAML['video']['filename'],
                                     extension=CONFIG_YAML['video']['extension'],
                                     fps=CONFIG_YAML['video']['fps'])
                case 'export':
                    # Export the model to a specified format
                    model.export(PARAMS)
                    export_data(ARCHITECTURE)

        case 'FasterRCNN':
            torch.set_float32_matmul_precision('medium')
            config = munch.munchify(yaml.load(open("configs/FasterRCNN/faster_rcnn_config.yaml"), Loader=yaml.FullLoader))

            assert MODE in ['train', 'val', 'pred', 'all'], 'Invalid mode. Please choose from: train, val, pred'

            pl.seed_everything(42)
            dm = CustomDataModule(batch_size=32, num_workers=4)
            
            if config.checkpoint_path:
                model = CustomFasterRCNN.load_from_checkpoint(checkpoint_path=config.checkpoint_path, config=config)
                print("Loading weights from checkpoint...")
            else:
                model = CustomFasterRCNN(config)
            
            trainer = pl.Trainer(
                devices=config.devices, 
                max_epochs=config.max_epochs, 
                check_val_every_n_epoch=config.check_val_every_n_epoch,
                enable_progress_bar=config.enable_progress_bar,
                precision="bf16-mixed",
                # deterministic=True,
                logger=WandbLogger(project=config.wandb_project, name=config.wandb_experiment_name, config=config),
                callbacks=[
                    EarlyStopping(monitor="val_acc", patience=config.early_stopping_patience, mode="max", verbose=True),
                    LearningRateMonitor(logging_interval="step"),
                    ModelCheckpoint(dirpath=Path(config.checkpoint_folder, config.wandb_project, config.wandb_experiment_name), 
                                    filename='best_model:epoch={epoch:02d}-val_acc={val_acc:.4f}',
                                    auto_insert_metric_name=False,
                                    save_weights_only=True,
                                    save_top_k=1),
                ])

            if not config.test_model:
                trainer.fit(model, train_dataloaders=dm.train_dataloader(), val_dataloaders=dm.val_dataloader())

            trainer.test(model, dataloaders=dm.train_dataloader())


if __name__ == "__main__":
    # Make results "deterministic"
    random.seed(0)
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