import YOLOv8

import argparse
import json

def main(args):
    objd = YOLOv8.ObjectDetection()

    objd.load_model(model_path=args.model_path)

    with open(args.val_config) as json_file:
        val_params = json.load(json_file)

    objd.validate(val_params=val_params)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Validate a given model.')
    parser.add_argument('--model_path', type=str, help='Path to the model (Ex. in "runs/detect/train*/weights/best.pt")')
    parser.add_argument('--val_config', type=str, default='configs/YOLOv8/NAPLab-LiDAR/train.json', help='Validation configuration file (Recommended to use the same as training)')


    args = parser.parse_args()
    main(args)
