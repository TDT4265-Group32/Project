import os
import glob
import atexit
import logging

import torch
from codecarbon import EmissionsTracker

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
    
    def train(self, train_params, loss_params):
        """Train the YOLOv8 model.
        
        Args:
        train_params (dict): Dictionary containing training parameters
        loss_params (dict): Dictionary containing loss parameters
        """
        # Set the DFL and IoU methods
        self.set_dfl(loss_params['use_dfl'])
        self.set_iou_method('giou', loss_params['use_giou'])
        self.set_iou_method('diou', loss_params['use_diou'])
        self.set_iou_method('ciou', loss_params['use_ciou'])

        # Initialize the emissions tracker
        tracker = EmissionsTracker(log_level=logging.WARNING)
        tracker.start()
        atexit.register(tracker.stop)

        # Start training
        result = super().train(**train_params)

        # Stop the timer and finalize the power consumption
        tracker.stop()
        atexit.unregister(tracker.stop)

        return result

    def validate(self, val_params):
        """Validate the YOLOv8 model.
        
        Args:
        val_params (dict): Dictionary containing validation parameters
        """
        
        results = super().val(**val_params)
        
        return results

    def predict(self,
                predict_params):
        """Predict using the YOLOv8 model.
        Also, save the results if the results_path is provided.

        Args:
        predict_params (dict): Dictionary containing prediction parameters
        """

        results = super().predict(**predict_params, device=self.device)

        results_path = os.path.join('results', 'YOLOv8')
        if not os.path.exists(results_path):
            os.makedirs(results_path)
        else:
            for file in os.listdir(results_path):
                if file.endswith('.PNG'):
                    os.remove(os.path.join(results_path, file))

        for idx, result in tqdm(enumerate(results), total=len(results), desc=f'Saving result frames'):
            img_path = sorted(glob.glob(os.path.join(predict_params['source'], '*.PNG')))[idx]
            original_image_name = os.path.basename(img_path)
            result.save(os.path.join(results_path, original_image_name))

        return results

    def benchmark(self, params):
        results = super().benchmark(**params)

        return results

    def export(self, params):
        results = super().export(**params)

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
