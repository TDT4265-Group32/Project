import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as F
from PIL import Image
import pytorch_lightning as pl

class NAPLabLiDAR(Dataset):
    def __init__(self, img_paths, annotations, transform=None):
        self.img_paths = img_paths
        self.annotations = annotations
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        annotation = self.annotations[idx]
        
        # Load image
        img = Image.open(img_path).convert("L")

        # TODO: Load annotations
        
        # # Load annotations from text file
        # with open(self.annotations, 'r') as file:
        #     # Parse annotations from the text file
        #     # This depends on the format of your annotation file
        #     # For example, if your annotations are in COCO format, you need to parse it accordingly
        #     annotation = parse_annotations(file)

        # Apply transformations
        if self.transform:
            img = self.transform(img)
        
        # Prepare targets (annotations)
        # You need to ensure that targets are returned for training mode
        if self.annotations is not None:
            target = annotation  # Assuming your annotation is in the correct format
        else:
            target = None  # If targets are not available, return None
        
        return img, target

class CustomDataModule(pl.LightningDataModule):
    def __init__(self, train_dataset, val_dataset, batch_size=32, num_workers=4):
        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.num_workers = num_workers

    # TODO: rewrite to datamodules

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)
    
    # TODO: Implement test_dataloader with a test dataset
    # def test_dataloader(self):
    #     return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

# # Example usage:
# train_dataset = CustomDataset(train_img_paths, train_annotations, transform=train_transform)
# val_dataset = CustomDataset(val_img_paths, val_annotations, transform=val_transform)

# data_module = CustomDataModule(train_dataset, val_dataset, batch_size=32, num_workers=4)
