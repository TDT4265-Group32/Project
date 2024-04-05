from ultralytics import YOLO
import os

# Load YOLOv8 model from pre-trained weights file
model = YOLO('models/pretrained/yolov8n-seg.pt')
# We can also use our own custom model: YOLO('path/to/model.pt')

# Example of arguments, others can also be "CUDA", etc.. (Can be seen in configs/default_format.yml)
result = model.train(data='coco128-seg.yaml', epochs=100, imgsz=640)

# Since the model is already trained, we can use the model to predict (no input data is required as it is already trained)
metrics = model.val()
# Show metrics
metrics.box.map
metrics.box.maps
metrics.seg.map
metrics.seg.maps

# Test prediction on an image
results = model('https://ultralytics.com/images/bus.jpg')

if os.path.exists("examples") == False:
    os.makedirs("examples")

# Export trained model
model.export("examples/custom-coco128-seg.pt")
