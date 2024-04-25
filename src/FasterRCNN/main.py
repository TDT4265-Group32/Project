import argparse

import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn, faster_rcnn

def main(args):
    # Load the Faster R-CNN model
    assert args.mode in ['train', 'validate', 'predict'], 'Invalid mode. Please choose from: train, validate, predict'
    
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    
    # Set the model to evaluation
    model.eval()
    
    # Load the image
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Faster R-CNN')
    parser.add_argument('--mode', type=str, help='Mode to run the script in \nOptions: train, validate, predict')
    parser.add_argument('--dataset', type=str, default='NAPLab-LiDAR', help='Name of dataset \nDefault: NAPLab-LiDAR \nCheck datasets with available configs in "configs" directory: \nconfigs/YOLOv8/<name_of_dataset>')
    
    args = parser.parse_args()

    main(args)