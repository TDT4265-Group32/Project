from partition_dataset import partition_dataset
import YOLOv8
import os
import torch
import random
random.seed(0)

def main():

    objd = YOLOv8.ObjectDetection()
    objd.predict('datasets/ultralytics/bus.jpg', save_path='results/ultralytics/bus_pred.jpg')

if __name__ == "__main__":
    main()
