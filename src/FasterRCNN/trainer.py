import lightning.pytorch as pl
#from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
#from lightning.pytorch.loggers import WandbLogger
import torch
#from torch import nn
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights 
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.ops import box_iou

from torchvision.ops import generalized_box_iou, giou_loss

#from torch.utils.data import DataLoader
#from torchvision.transforms import functional as F
#from torchvision.transforms import v2 as T
#from datamodule import CustomDataModule, NAPLabLiDAR

class CustomFasterRCNN(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Define the model
        weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT if config.use_pretrained_weights else None
        self.model = fasterrcnn_resnet50_fpn(weights=weights, progress=True)

        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, config.num_classes)

        # Freeze the parameters of the RPN
        for param in self.model.rpn.parameters():
            param.requires_grad = False

        self.acc_fn = MeanAveragePrecision(box_format="xyxy", iou_type="bbox", iou_thresholds=[0.2])
    
    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.config.max_lr, momentum=self.config.momentum, weight_decay=self.config.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80,200], gamma=0.1)
        #lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)
        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler,
            #"monitor": "Validation_mAP",  # Ensure Validation_mAP is being monitored
        }

    def forward(self, x, y=None):
        return self.model(x, y)

    def training_step(self, batch, batch_idx):
        x, y = batch

        y_hat = self.forward(x, y)

        loss = y_hat['loss_classifier'] + y_hat['loss_box_reg'] + y_hat['loss_objectness'] + y_hat['loss_rpn_box_reg']

        self.log_dict({
            "train_loss": loss,
            "loss_classifier": y_hat['loss_classifier'],
            "loss_box_reg": y_hat['loss_box_reg'],
            "loss_objectness": y_hat['loss_objectness'],
            "loss_rpn_box_reg": y_hat['loss_rpn_box_reg']
        },on_epoch=True, on_step=False, prog_bar=True, sync_dist=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x, y)  # Pass both x and y to forward
        acc = self.acc_fn(y_hat, y)

        self.log_dict({
            "val_acc": acc['map']
        }, on_epoch=True, on_step=False, prog_bar=True, sync_dist=True)
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        acc = self.acc_fn(y_hat, y)
        self.log_dict({
            "test_acc": acc['map'],
        },on_epoch=True, on_step=False, prog_bar=True, sync_dist=True)

