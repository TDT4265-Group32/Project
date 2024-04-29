import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger
import torch
from torch import nn
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchmetrics import Accuracy
import munch
import yaml
import glob
from pathlib import Path


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

        self.model = fasterrcnn_resnet50_fpn(pretrained=True)

        # in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        # self.model.roi_heads.box_predictor = torch.nn.Linear(in_features, num_classes)

        self.loss_fn = nn.CrossEntropyLoss()
        self.acc_fn = Accuracy(task="multiclass", num_classes=self.config.num_classes)
    
    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.config.max_lr, momentum=self.config.momentum, weight_decay=self.config.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.config.max_epochs)
        return [optimizer], [{"scheduler": lr_scheduler, "interval": "epoch"} ]

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss_fn(y_hat, y)
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
    
    train_img_paths = glob.glob("datasets/NAPLab-LiDAR/images/train/*.PNG")
    train_annotations = glob.glob("datasets/NAPLab-LiDAR/labels/train/*.txt")

    val_img_paths = glob.glob("datasets/NAPLab-LiDAR/images/val/*.PNG")
    val_annotations = glob.glob("datasets/NAPLab-LiDAR/labels/val/*.txt")

    train_transform = T.Compose([
        T.Resize((1024, 1024)),
        T.RandomHorizontalFlip(),
        T.RandomRotation(30),
        T.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.0, hue=0.0),
        T.RandomErasing(p=0.3, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False),
        T.ToTensor(),
    ])

    train_dataset = NAPLabLiDAR(train_img_paths, train_annotations, transform=train_transform)
    val_dataset = NAPLabLiDAR(val_img_paths, val_annotations)

    dm = CustomDataModule(train_dataset, val_dataset, batch_size=32, num_workers=4)

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
        trainer.fit(model, datamodule=dm.train_dataloader())
    
    trainer.test(model, datamodule=dm.test_dataloader())
