from ultralytics import YOLO
from partition_dataset import partition_dataset
import os
import torch
import random
random.seed(0)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load YOLOv8 model from pre-trained weights file
model = YOLO('models/pretrained/yolov8n.pt')
# We can also use our own custom model: YOLO('path/to/model.pt')

partition_dataset(dataset_dir='datasets/NAPLab-LiDAR', force_repartition=False)
result = model.train(data='data/NAPLab-LiDAR.yaml', epochs=10, imgsz=640, device=device)

# Since the model is already trained, we can use the model to predict (no input data is required as it is already trained)
metrics = model.val()

# Test prediction on an image
test_results = model.predict('datasets/NAPLab-LiDAR/images/test/')

# Export trained model
model.export()
