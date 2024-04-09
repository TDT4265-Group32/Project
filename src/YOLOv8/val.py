import YOLOv8

from os import path
import argparse
import json

def main(args):
    objd = YOLOv8.ObjectDetection()

    objd.load_model(model_path=args.model_path)

    json_path = path.join('configs', 'YOLOv8', args.dataset, 'val.json')
    with open(json_path) as json_file:
        val_params = json.load(json_file)

    objd.validate(val_params=val_params)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Validate a given model.', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--model_path', type=str,
                        help='Path to the model \
                            \n(Ex. in "runs/detect/train*/weights/best.pt")')
    parser.add_argument('--dataset', type=str, default='NAPLab-LiDAR', 
                        help='Validation configuration file \
                            \n(Recommended to format "configs/YOLOv8/<dataset>/val.json" similar to train.json \
                            \nthat was used for the chosen model in <model_path>)')


    args = parser.parse_args()
    main(args)
