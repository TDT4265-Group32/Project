from ultralytics import YOLO
import os
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load YOLOv8 model from pre-trained weights file
model = YOLO('models/pretrained/yolov8n-seg.pt')
# We can also use our own custom model: YOLO('path/to/model.pt')

# Example of arguments, others can also be "CUDA", etc.. (Can be seen in configs/default_format.yml)
result = model.train(data='coco128-seg.yaml', epochs=100, imgsz=640, device=device)

# Since the model is already trained, we can use the model to predict (no input data is required as it is already trained)
metrics = model.val()
# Show metrics
print(metrics.box.map)
print(metrics.box.maps)
print(metrics.seg.map)
print(metrics.seg.maps)

# Test prediction on an image
results = model('https://ultralytics.com/images/bus.jpg')

if os.path.exists("models/examples") == False:
    os.makedirs("models/examples")

# Export trained model
model.export()
