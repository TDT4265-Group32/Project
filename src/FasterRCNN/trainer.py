import lightning.pytorch as pl
#from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
#from lightning.pytorch.loggers import WandbLogger
import torch
#from torch import nn
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchmetrics import Accuracy
from torchvision.ops import box_iou

#from torchvision.ops import generalized_box_iou, giou_loss

#from torch.utils.data import DataLoader
#from torchvision.transforms import functional as F
#from torchvision.transforms import v2 as T
#from datamodule import CustomDataModule, NAPLabLiDAR

class CustomFasterRCNN(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)

        # in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        # self.model.roi_heads.box_predictor = torch.nn.Linear(in_features, num_classes)

        # self.acc_fn = Accuracy(task="multiclass", num_classes=self.config.num_classes)
    
    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.config.max_lr, momentum=self.config.momentum, weight_decay=self.config.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)
        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler,
            "monitor": "Validation_mAP",
        }

    def forward(self, x, y=None):
        return self.model(x, y)

    def training_step(self, batch, batch_idx):
        x, y = batch

        # x = x[batch_idx]
        # y = y[batch_idx]

        y_hat = self.forward(x, y)
        loss = y_hat

        combined_loss = sum(y_hat.values())


        # # Calculate GIoU loss
        # losses = []
        # for pred_boxes, gt_boxes in zip(y_hat['boxes'], y['boxes']):
        #     loss = self.giou_loss(pred_boxes, gt_boxes)
        #     losses.append(loss)
        
        # # Compute average loss for the batch
        # loss = torch.mean(torch.stack(losses))
        
        # Log loss and accuracy
        # acc = self.acc_fn(y_hat, y)
        # self.log_dict({
        #     "train/loss": loss,
        #     "train/acc": acc
        # },on_epoch=True, on_step=False, prog_bar=True, sync_dist=True)
        
        return combined_loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        acc = self.acc_fn(y_hat, y)
        loss = self.loss_fn(y_hat, y)
        self.log_dict({
            "val/loss":loss,
            "val/acc": acc
        },on_epoch=True, on_step=False, prog_bar=True, sync_dist=True)
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        acc = self.acc_fn(y_hat, y)
        self.log_dict({
            "test/acc": acc,
        },on_epoch=True, on_step=False, prog_bar=True, sync_dist=True)

    def giou_loss(self, pred_box, gt_box):
        
        # Convert boxes from (x, y, w, h) to (x1, y1, x2, y2)
        pred_box = torch.stack([pred_box[:, 0], pred_box[:, 1], pred_box[:, 0] + pred_box[:, 2], pred_box[:, 1] + pred_box[:, 3]], dim=1)
        gt_box = torch.stack([gt_box[:, 0], gt_box[:, 1], gt_box[:, 0] + gt_box[:, 2], gt_box[:, 1] + gt_box[:, 3]], dim=1)
        
        iou = box_iou(pred_box, gt_box)
        
        enclosing_x1 = torch.min(pred_box[:, 0], gt_box[:, 0])
        enclosing_y1 = torch.min(pred_box[:, 1], gt_box[:, 1])
        enclosing_x2 = torch.max(pred_box[:, 2], gt_box[:, 2])
        enclosing_y2 = torch.max(pred_box[:, 3], gt_box[:, 3])

        enclosing_area = (enclosing_x2 - enclosing_x1) * (enclosing_y2 - enclosing_y1)
        giou = iou - (enclosing_area - iou) / enclosing_area
        
        return 1 - giou
