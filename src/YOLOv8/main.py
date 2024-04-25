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

class CustomBboxLoss(BboxLoss):
    """Custom bounding box loss class."""
    def __init__(self, reg_max, use_dfl=False):
        """Use CIoU loss for the bounding box loss."""
        super().__init__(reg_max, use_dfl)
        self.reg_max = reg_max
        self.use_dfl = use_dfl

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask):
        """Modified forward function for the bounding box loss."""
        
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        # Use CIoU loss
        CIoU = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=True, CIoU=True)
        loss_iou = 1 - CIoU
        if self.use_dfl:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.reg_max)
            loss_dfl = self._df_loss(pred_dist[fg_mask].view(-1, self.reg_max + 1), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)

        return loss_iou, loss_dfl

class CustomYOLO(YOLO):
    """Custom YOLOv8 model."""
    def __init__(self, cfg='models/pretrained/yolov8n.pt'):
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

    def predict(self, predict_params, results_path=None):
        """Predict using the YOLOv8 model.
        Also, save the results if the results_path is provided.

        Args:
        predict_params (dict): Dictionary containing prediction parameters
        """

        results = super().predict(**predict_params, device=self.device)
        
        if results_path is not None:
        
            if not os.path.exists(results_path):
                os.makedirs(results_path)

            for idx, result in tqdm(enumerate(results), total=len(results), desc=f'Saving result frames'):
                img_path = sorted(glob.glob(os.path.join(predict_params['source'], '*.PNG')))[idx]
                original_image_name = os.path.basename(img_path)
                result.save(os.path.join(results_path, original_image_name))

        return results

def main(args, start_time):
    assert args.mode in ['train', 'val', 'pred'], 'Invalid mode. Please choose from: train, validate, predict'

    yolo_model = CustomYOLO()
    json_path = os.path.join('configs', 'YOLOv8', args.dataset, args.mode + '.json')
    with open(json_path) as json_file:
        json_content = json.load(json_file)
    
    params = json_content['params']

    if args.mode == 'train':
        dataset = args.dataset
        if dataset == 'NAPLab-LiDAR':
            # Currently, only NAPLab-LiDAR has the desired structure for the "partition_dataset" function
            if json_content['partition']['mode'] == 'video':
                params['epochs'] = ceil(params['epochs'] / json_content['partition']['num_shuffles'])
                for i in range(json_content['partition']['num_shuffles']):
                    print("\n" + "="*50)
                    print("Running shuffle {0}...".format(i + 1).center(50))
                    print("="*50 + "\n")
                    partition_video_dataset(dataset, 18, seed=i)
                    yolo_model.train(params)
                
            elif json_content['partition']['mode'] == 'images':
                partition_dataset(dataset, force_repartition=False)
                yolo_model.train(params)
            else:
                raise ValueError('Invalid partition mode. Please choose from: video, images')

        else:
            yolo_model.train(params)

        yolo_model.export()

    elif args.mode == 'val':
        model_path = json_content['model_path']
        yolo_model.load(model_path)
        yolo_model.validate(val_params=params)

    elif args.mode == 'pred':
        model_path = json_content['model_path']

        results_path = os.path.join('results', args.dataset)
        yolo_model.predict(params, results_path=results_path)
        
        if json_content['video']['create_video']:
            create_video(results_path, dst_path=os.path.join(results_path, json_content['video']['filename']))

    elapsed_time = time.time() - start_time
    with open('elapsed_time.txt', 'w') as f:
        f.write(str(elapsed_time))

if __name__ == "__main__":
    start_time = time.time()
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
    main(args, start_time)