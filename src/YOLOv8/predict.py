from utils.png_to_video import create_video
import YOLOv8

from os import path
import argparse
import json

def main(args):
    objd = YOLOv8.ObjectDetection()

    objd.load_model(model_path=args.model_path)
    
    json_path = path.join('configs', 'YOLOv8', args.dataset, 'predict.json')
    with open(json_path) as json_file:
        predict_params = json.load(json_file)    

    results_path = path.join('results', args.dataset)
    objd.predict(predict_params, results_path=results_path)
    
    if args.create_video:
        create_video(args.results_path, destination=path.join(results_path, 'pred_video.mp4'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--model_path', type=str, 
                        help='Path to the model \
                            \n(Ex. in "runs/detect/train*/weights/best.pt")')
    parser.add_argument('--dataset', type=str, default='NAPLab-LiDAR', help='Dataset to perform prediction/test')
    parser.add_argument('--create_video', type=bool, default=False, help='Create video from the results')

    args = parser.parse_args()
    main(args)
