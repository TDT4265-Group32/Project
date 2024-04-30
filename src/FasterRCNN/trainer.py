import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger
import torch
from torch import nn
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchmetrics import Accuracy
import munch
import yaml
import glob
from pathlib import Path

from torchvision.ops import generalized_box_iou, giou_loss

import argparse
from torch.utils.data import DataLoader
from torchvision.transforms import functional as F
from torchvision.transforms import v2 as T
from datamodule import CustomDataModule, NAPLabLiDAR

torch.set_float32_matmul_precision('medium')
config = munch.munchify(yaml.load(open("configs/FasterRCNN/faster_rcnn_config.yaml"), Loader=yaml.FullLoader))

class FasterRCNN(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)

        # in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        # self.model.roi_heads.box_predictor = torch.nn.Linear(in_features, num_classes)

        self.acc_fn = Accuracy(task="multiclass", num_classes=self.config.num_classes)
    
    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.config.max_lr, momentum=self.config.momentum, weight_decay=self.config.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.config.max_epochs)
        return [optimizer], [{"scheduler": lr_scheduler, "interval": "epoch"} ]

    def forward(self, x, y=None):
        return self.model(x, y)

    def training_step(self, batch, batch_idx):
        x, y = batch

        x = x[batch_idx]
        y = y[batch_idx]

        y_hat = self.forward(x, y)

        # Calculate GIoU loss
        losses = []
        for pred_boxes, gt_boxes in zip(y_hat['boxes'], y['boxes']):
            loss = giou_loss(pred_boxes, gt_boxes)
            losses.append(loss)
        
        # Compute average loss for the batch
        loss = torch.mean(torch.stack(losses))
        
        # Log loss and accuracy
        acc = self.acc_fn(y_hat, y)
        self.log_dict({
            "train/loss": loss,
            "train/acc": acc
        },on_epoch=True, on_step=False, prog_bar=True, sync_dist=True)
        
        return loss
    
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


if __name__ == "__main__":
    
    pl.seed_everything(42)

    dm = CustomDataModule(batch_size=32, num_workers=4)

    if config.checkpoint_path:
        model = FasterRCNN.load_from_checkpoint(checkpoint_path=config.checkpoint_path, config=config)
        print("Loading weights from checkpoint...")
    else:
        model = FasterRCNN(config)

    trainer = pl.Trainer(
        devices=config.devices, 
        max_epochs=config.max_epochs, 
        check_val_every_n_epoch=config.check_val_every_n_epoch,
        enable_progress_bar=config.enable_progress_bar,
        precision="bf16-mixed",
        # deterministic=True,
        logger=WandbLogger(project=config.wandb_project, name=config.wandb_experiment_name, config=config),
        callbacks=[
            EarlyStopping(monitor="val/acc", patience=config.early_stopping_patience, mode="max", verbose=True),
            LearningRateMonitor(logging_interval="step"),
            ModelCheckpoint(dirpath=Path(config.checkpoint_folder, config.wandb_project, config.wandb_experiment_name), 
                            filename='best_model:epoch={epoch:02d}-val_acc={val/acc:.4f}',
                            auto_insert_metric_name=False,
                            save_weights_only=True,
                            save_top_k=1),
        ])
    
    if not config.test_model:
        trainer.fit(model, train_dataloaders=dm.train_dataloader())
    
    trainer.test(model, test_dataloaders=dm.test_dataloader())
