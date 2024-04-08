from utils.png_to_video import create_video
import YOLOv8

from os import path
import argparse
import json

def main(args):
    objd = YOLOv8.ObjectDetection()

    objd.load_model(model_path=args.model_path)
    
    with open(args.pred_config) as json_file:
        predict_params = json.load(json_file)    

    objd.predict(predict_params, results_path=args.results_path)
    
    if create_video:
        create_video(args.results_path, destination=path.join(args.results_path, 'pred_video.mp4'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--model_path', type=str, help='Path to the model (Usually in "runs/detect/train*/weights/best.pt")')
    parser.add_argument('--pred_config', type=str, default='configs/YOLOv8/NAPLab-LiDAR/predict.json', help='Prediction configuration file')

    parser.add_argument('--results_path', type=str, default='results/NAPLab-LiDAR/', help='Path to save the results')
    parser.add_argument('--create_video', type=bool, default=False, help='Create video from the results')

    args = parser.parse_args()
    main(args)
