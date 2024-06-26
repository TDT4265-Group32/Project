import glob
from PIL import Image

import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2 as T

class NAPLabLiDAR(Dataset):
    def __init__(self, img_paths, annotation_paths, transform=None):
        self.img_paths = img_paths
        self.annotation_paths = annotation_paths
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        annotation_path = self.annotation_paths[idx]
        
        # Load image
        img = Image.open(img_path).convert("RGB")
        
        # Load annotations from YOLO format text file
        with open(annotation_path, 'r') as file:
            boxes = []
            labels = []
            for line in file:
                # Parse annotation line
                parts = line.strip().split(' ')
                class_label = int(parts[0])
                x_center, y_center, width, height = map(float, parts[1:])
                
                # Convert YOLO format to bounding box coordinates (x_min, y_min, x_max, y_max)
                x_min = (x_center - width / 2)
                y_min = (y_center - height / 2)
                x_max = (x_center + width / 2)
                y_max = (y_center + height / 2)
                
                boxes.append([x_min, y_min, x_max, y_max])
                labels.append(class_label)
            
        # Convert to tensors
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        # Prepare targets
        targets = {
            'boxes': boxes,
            'labels': labels
        }

        # Apply transformations
        if self.transform:
            img, targets = self.transform(img, targets)
        
        return img, targets

class CustomDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=32, num_workers=4):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers

        train_img_paths = glob.glob("datasets/NAPLab-LiDAR/images/train/*.PNG")
        train_annotations = glob.glob("datasets/NAPLab-LiDAR/labels/train/*.txt")

        val_img_paths = glob.glob("datasets/NAPLab-LiDAR/images/val/*.PNG")
        val_annotations = glob.glob("datasets/NAPLab-LiDAR/labels/val/*.txt")

        test_img_paths = glob.glob("datasets/NAPLab-LiDAR/images/test/*.PNG")
        test_annotations = glob.glob("datasets/NAPLab-LiDAR/labels/test/*.txt")

        assert len(train_img_paths) > 0, "No training images found"
        assert len(val_img_paths) > 0, "No validation images found"
        assert len(train_img_paths) == len(train_annotations), f"Number of images and annotations do not match for train set"
        assert len(val_img_paths) == len(val_annotations), f"Number of images and annotations do not match for val set"

        self.train_dataset = NAPLabLiDAR(train_img_paths, train_annotations, transform=self.get_transforms("train"))
        self.val_dataset = NAPLabLiDAR(val_img_paths, val_annotations, transform=self.get_transforms("val"))
        self.test_dataset = NAPLabLiDAR(test_img_paths, test_annotations, transform=self.get_transforms("test"))

    def collate_fcn(self, batch):
        x = []
        y = []

        for i in range(len(batch)):
            x.append(batch[i][0])
            y.append(batch[i][1])

        return x, y

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True, shuffle=True, collate_fn=self.collate_fcn)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True, shuffle=False, collate_fn=self.collate_fcn)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True, shuffle=False, collate_fn=self.collate_fcn)
    
    def get_transforms(self, split):
        mean = [0.4696, 0.4696, 0.4696]
        std = [0.0339, 0.0339, 0.0339]
        
        shared_transforms = [
            T.ToTensor(),
            T.Resize((1024, 1024)),
            T.Normalize(mean, std) 
        ]
        
        if split == "train":
            return T.Compose([
                *shared_transforms,
                # T.RandomHorizontalFlip(),
                # T.RandomRotation(30),
                # T.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                # T.RandomErasing(p=0.3, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False),
            ])
            
        elif split == "val" or split == "test":
            return T.Compose([
                *shared_transforms,
            ])


# # Example usage:
# train_dataset = CustomDataset(train_img_paths, train_annotations, transform=train_transform)
# val_dataset = CustomDataset(val_img_paths, val_annotations, transform=val_transform)

# data_module = CustomDataModule(train_dataset, val_dataset, batch_size=32, num_workers=4)
