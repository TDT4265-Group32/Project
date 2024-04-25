import os
import argparse
import json
import glob

from utils.partition_dataset import partition_dataset, partition_video_dataset
import torch
from tools.png_to_video import create_video

from ultralytics import YOLO
from tqdm import tqdm

class YOLOv8():
    
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.load_model()

    def save_model(self, path):

        raise NotImplementedError('This function is not yet implemented.')

    def load_model(self, model_path='models/pretrained/yolov8n.pt'):
        """Load the YOLOv8 model.
        Default model path is the pretrained YOLOv8n model.
        
        Args:
        model_path (str): Path to the model file
    
        """
        model = YOLO(model_path)
        model.fuse()
        
        self.model = model
        
        return model
    
    def train(self, train_params):
        """
        For more, check out: https://docs.ultralytics.com/modes/train/
        """
        result = self.model.train(**train_params, device=self.device)
        
        return result

    def run_inference(self, show=True, conf=0.4):
        """
        For more, check out: https://docs.ultralytics.com/modes/predict/#key-features-of-predict-mode
        """
        self.model(source=0, show=show, conf=conf)


    def validate(self, val_params):
        """
        For more, check out: https://docs.ultralytics.com/modes/val/
        """
        results = self.model.val(**val_params, device=self.device)
        print(f'Precision: {results.p}')
        print(f'Recall: {results.r}')
        print(f'mAP: {results.map}')
        
        return results

    def predict(self, predict_params, results_path=None):
        """
        For more, check out: https://docs.ultralytics.com/modes/predict/#key-features-of-predict-mode
        """
        # Split the file path into root and extension

        results = self.model.predict(**predict_params, device=self.device)
        
        if results_path is not None:
        
            if not os.path.exists(results_path):
                os.makedirs(results_path)

            for idx, result in tqdm(enumerate(results), total=len(results), desc=f'Saving result frames'):
                img_path = sorted(glob.glob(os.path.join(predict_params['source'], '*.PNG')))[idx]
                original_image_name = os.path.basename(img_path)
                result.save(os.path.join(results_path, original_image_name))

        return results

def main(args):
    """Main function for running the script.
    This function is responsible for running YOLOv8 model in the desired mode for the specified dataset.
    Configurations for each mode can be found in the respective JSON files in the "configs" directory.
    
    Args:
    args (argparse.Namespace): Arguments passed to the script containing the mode and dataset name
    
    """
    assert args.mode in ['train', 'val', 'pred'], 'Invalid mode. Please choose from: train, validate, predict'

    yolo_model = YOLOv8()
    json_path = os.path.join('configs', 'YOLOv8', args.dataset, args.mode + '.json')
    with open(json_path) as json_file:
        json_content = json.load(json_file)
    
    params = json_content['params']

    if args.mode == 'train':
        dataset = args.dataset
        if dataset == 'NAPLab-LiDAR':
            # Currently, only NAPLab-LiDAR has the desired structure for the "partition_dataset" function
            # partition_dataset(dataset_dir=os.path.join('datasets', args.dataset), force_repartition=False)
            # partition_video_dataset(args.dataset, 18)
            if json_content['partition'] == 'video':
                params['epochs'] = json_content['video']['epochs_per_seg']
                for _ in range(json_content['video']['num_shuffles']):
                    partition_video_dataset(dataset, 18)
                    yolo_model.train(params)
                
            elif json_content['partition'] == 'images':
                partition_dataset(dataset, force_repartition=False)
                yolo_model.train(params)

        else:
            yolo_model.train(params)
        
        yolo_model.export()

    elif args.mode == 'val':
        model_path = json_content['model_path']
        yolo_model.load_model(model_path=model_path)
        yolo_model.validate(val_params=params)

    elif args.mode == 'pred':
        model_path = json_content['model_path']

        results_path = os.path.join('results', args.dataset)
        yolo_model.predict(params, results_path=results_path)
        
        if json_content['video']['create_video']:
            create_video(results_path, dst_path=os.path.join(results_path, json_content['video']['filename']))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Script for training model.', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--mode', type=str, 
                        help='Mode to run the script in \
                            \nOptions: train, validate, predict')
    parser.add_argument('--dataset', type=str, default='NAPLab-LiDAR', 
                        help='Name of dataset \
                            \nDefault: NAPLab-LiDAR \
                            \nCheck datasets with available configs in "configs" directory: \
                            \nconfigs/YOLOv8/<name_of_dataset>')

    args = parser.parse_args()
    main(args)