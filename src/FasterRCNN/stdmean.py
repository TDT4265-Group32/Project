import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os

class CustomImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = [os.path.join(root_dir, img) for img in os.listdir(root_dir)]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image

# Define transform to convert images to tensors
transform = transforms.Compose([transforms.ToTensor()])

# Load dataset from the specified directory
train_dataset = CustomImageDataset(root_dir='datasets/NAPLab-LiDAR/images/train', transform=transform)

# Create DataLoader for the dataset
data_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)

class MeanStdCalculator:
    def __init__(self):
        self.count = 0
        self.mean = 0
        self.M2 = 0

    def update(self, x):
        self.count += 1
        delta = x - self.mean
        self.mean += delta / self.count
        delta2 = x - self.mean
        self.M2 += delta * delta2

    def compute(self):
        variance = self.M2 / (self.count - 1)
        std = torch.sqrt(variance)
        return self.mean, std

# Initialize mean and std calculator
mean_std_calculator = MeanStdCalculator()

# Calculate mean and std
for image in data_loader:
    image = image.float()  # Ensure conversion to float
    mean_std_calculator.update(torch.mean(image))
mean, std = mean_std_calculator.compute()

# Print mean and std
print("Mean:", mean)
print("Std:", std)

# Now you can use mean and std for normalization
