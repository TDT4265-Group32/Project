import os
import argparse
import json
import glob
import time
from math import ceil

from utils.partition_dataset import partition_dataset, partition_video_dataset
import torch
from tools.png_to_video import create_video

from ultralytics import YOLO
from ultralytics.utils.loss import BboxLoss
from ultralytics.utils.metrics import bbox_iou
from ultralytics.utils.tal import bbox2dist
from tqdm import tqdm
from codecarbon import EmissionsTracker
import random

class CustomBboxLoss(BboxLoss):
    """Custom bounding box loss class."""
    def __init__(self, reg_max, use_dfl=False):
        """Use CIoU loss for the bounding box loss."""
        super().__init__(reg_max, use_dfl)
        self.reg_max = reg_max
        self.use_dfl = use_dfl
        self.use_iou_method = {"giou": False,
                                "diou": False,
                                "ciou": False}

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask):
        """Modified forward function for the bounding box loss."""
        
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)

        iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=True,
                        CIoU=self.use_iou_method['ciou'],
                        DIoU=self.use_iou_method['diou'],
                        GIoU=self.use_iou_method['giou'])
        loss_iou = 1 - iou
        if self.use_dfl:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.reg_max)
            loss_dfl = self._df_loss(pred_dist[fg_mask].view(-1, self.reg_max + 1), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)

        return loss_iou, loss_dfl

class CustomYOLO(YOLO):
    """Custom YOLOv8 model."""
    def __init__(self, cfg='models/pretrained/yolov8m.pt'):
        super().__init__(cfg)
        # Choose to get better performance by sacrificing speed
        self.loss = CustomBboxLoss(reg_max=4, use_dfl=True)

    def load(self, weights):
        """Load the YOLOv8 model.

        Args:
        weights (str): Path to the weights file
        """
        super().load(weights)
        self.fuse()
    
    def train(self, train_params):
        """Train the YOLOv8 model.
        
        Args:
        train_params (dict): Dictionary containing training parameters
        """
        result = super().train(**train_params)
        return result

    def validate(self, val_params):
        """Validate the YOLOv8 model.
        
        Args:
        val_params (dict): Dictionary containing validation parameters
        """
        
        results = super().val(**val_params)
        
        return results

    def predict(self, predict_params, dataset: str = None):
        """Predict using the YOLOv8 model.
        Also, save the results if the results_path is provided.

        Args:
        predict_params (dict): Dictionary containing prediction parameters
        """

        results = super().predict(**predict_params, device=self.device)
        
        if dataset is not None:
            results_path = os.path.join('results', dataset)
            if not os.path.exists(results_path):
                os.makedirs(results_path)
            else:
                # Clear the results folder
                for file in os.listdir(results_path):
                    if file.endswith('.PNG'):
                        os.remove(os.path.join(results_path, file))

            for idx, result in tqdm(enumerate(results), total=len(results), desc=f'Saving result frames'):
                img_path = sorted(glob.glob(os.path.join(predict_params['source'], '*.PNG')))[idx]
                original_image_name = os.path.basename(img_path)
                result.save(os.path.join(results_path, original_image_name))

        return results

    def set_dfl(self, active: bool):
        """Set the DFL flag in the loss function.

        Args:
        active (bool): Flag to activate DFL
        """
        self.loss.use_dfl = active

    def set_iou_method(self, method: str, active: bool):
        """Set the IoU method in the loss function.

        Args:
        method (str): Name of the IoU method
        active (bool): Flag to activate the IoU method
        """
        assert method in ['giou', 'diou', 'ciou'], 'Invalid IoU method. Please choose from: giou, diou, ciou'

        self.loss.use_iou_method[method] = active

def main(args):
    MODE = args.mode
    DATASET = args.dataset

    assert MODE in ['train', 'val', 'pred'], 'Invalid mode. Please choose from: train, validate, predict'

    # Load the configuration file
    JSON_PATH = os.path.join('configs', 'YOLOv8', DATASET, MODE + '.json')
    with open(JSON_PATH) as json_config_file:
        CONFIG_JSON = json.load(json_config_file)

    # Use custom YOLOv8 model with overloaded functions, implementations can be seen above
    model = CustomYOLO(cfg=CONFIG_JSON['model_path'])
    # Load parameters to be passed onto train, validate, or predict functions
    PARAMS = CONFIG_JSON['params']

    if MODE == 'train':
        # Set the loss function parameters
        model.set_dfl(CONFIG_JSON['loss_function']['use_dfl'])
        model.set_iou_method('giou', CONFIG_JSON['loss_function']['use_giou'])
        model.set_iou_method('diou', CONFIG_JSON['loss_function']['use_diou'])
        model.set_iou_method('ciou', CONFIG_JSON['loss_function']['use_ciou'])

        # Start the timer and initialize the power consumption
        start_time = time.time()
        tracker = EmissionsTracker()
        tracker.start()

        # Partition functions currently only work with NAPLab-LiDAR
        if DATASET == 'NAPLab-LiDAR':
            
            # Partition the dataset based on the mode
            if CONFIG_JSON['partition']['mode'] == 'video':

                # Evenly distribute the epochs across the number of shuffles
                PARAMS['epochs'] = ceil(PARAMS['epochs'] / CONFIG_JSON['partition']['num_shuffles'])
                for i in range(CONFIG_JSON['partition']['num_shuffles']):
                    print("\n" + "="*50)
                    print("Running shuffle {0}...".format(i + 1).center(50))
                    print("="*50 + "\n")
                    partition_video_dataset(DATASET, 18, seed=i)
                    model.train(PARAMS)
                
            elif CONFIG_JSON['partition']['mode'] == 'images':
                partition_dataset(DATASET)
                model.train(PARAMS)
            else:
                raise ValueError('NAPLab-LiDAR dataset needs to be partitioned by either "video" or "images" mode.')

        # For other datasets, just train the model
        else:
            model.train(PARAMS)

        # Stop the timer and finalize the power consumption
        tracker.stop()
        elapsed_time = time.time() - start_time
        
        # Save the power consumption
        # Save the elapsed time
        hours, remainder = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        # Write the results to a file
        with open('time_elapsed.txt', 'w') as f:
            f.write(f"Time elapsed: {int(hours)}h {int(minutes)}m {seconds}s\n")

    elif MODE == 'val':
        model_path = CONFIG_JSON['model_path']
        model.load(model_path)
        model.validate(val_params=PARAMS)

    elif MODE == 'pred':
        model_path = CONFIG_JSON['model_path']

        model.predict(PARAMS, DATASET)

        # Only sensible to create video from sequence of PNGs for NAPLab-LiDAR dataset
        if CONFIG_JSON['video']['create_video'] and DATASET == 'NAPLab-LiDAR':

            # Create path if it doesn't exist
            if not os.path.exists(CONFIG_JSON['video']['path']):
                os.makedirs(CONFIG_JSON['video']['path'])

            # Create video from sequence of PNGs
            create_video(os.path.join('results', DATASET),
                         dst_path=os.path.join(CONFIG_JSON['video']['path'],
                                               CONFIG_JSON['video']['filename']))

if __name__ == "__main__":
    # Make results "deterministic"
    random.seed(0)
    # Parse the arguments
    parser = argparse.ArgumentParser(description='Script for training model.', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--mode', type=str, 
                        help='Mode to run the script in \
                            \nOptions: train, validate, predict')
    parser.add_argument('--dataset', type=str, default='NAPLab-LiDAR', 
                        help='Name of dataset \
                            \nDefault: NAPLab-LiDAR \
                            \nCheck datasets with available configs in "configs" directory: \
                            \nconfigs/YOLOv8/<name_of_dataset>')
    args = parser.parse_args()
    main(args)