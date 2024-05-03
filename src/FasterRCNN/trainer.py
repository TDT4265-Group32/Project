import lightning.pytorch as pl
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights 
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.utils import draw_bounding_boxes
from cv2 import imwrite

class CustomFasterRCNN(pl.LightningModule):
    def __init__(self, config: dict):
        super().__init__()
        self.config = config

        # Define the model
        weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT if self.config['use_pretrained_weights'] else None
        self.model = fasterrcnn_resnet50_fpn(weights=weights, progress=True)

        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.config['num_classes']+1)

        # Freeze the parameters of the RPN
        for param in self.model.rpn.parameters():
            param.requires_grad = False

        self.mAP = MeanAveragePrecision(box_format="xyxy", iou_type="bbox")
    
    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.config['max_lr'], momentum=self.config['momentum'], weight_decay=self.config['weight_decay'])
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
        
        img = x[batch_idx].to(torch.uint8)

        img = draw_bounding_boxes(img, y_hat[batch_idx]['boxes'], y_hat[batch_idx]['labels'])
        imwrite(f"results/{batch_idx}.png", img)

        self.mAP.update(y_hat, y)
        acc = self.mAP.compute()

        self.log_dict({
            "val_acc": acc['map']
        }, on_epoch=True, on_step=False, prog_bar=True, sync_dist=True)
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        self.mAP.update(y_hat, y)
        acc = self.mAP.compute()
        self.log_dict({
            "test_acc": acc['map'],
        },on_epoch=True, on_step=False, prog_bar=True, sync_dist=True)

