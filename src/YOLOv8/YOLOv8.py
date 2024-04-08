from ultralytics import YOLO

import os
import torch
from tqdm import tqdm

"""
This example script was inspired by Ultralytics tutorial on how to use yolov8
"""

class ObjectDetection:
    
    def __init__(self):
        
        # Use this if we want to run OBD on camera
        # self.capture_index = capture_index
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Using Device: ", self.device)
        
        self.load_model()

    def load_model(self, model_path='models/pretrained/yolov8n.pt'):
        
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
        
        return results

    def predict(self, predict_params, results_path=None):
        """
        For more, check out: https://docs.ultralytics.com/modes/predict/#key-features-of-predict-mode
        """
        # Split the file path into root and extension

        results = self.model.predict(**predict_params, device=self.device)
        
        if results_path is not None:
            root, ext = os.path.splitext(results_path)

            # If the extension is not empty, get the directory name
            if ext:
                save_dir = os.path.dirname(root)
            else:
                save_dir = root
        
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            for idx, result in tqdm(enumerate(results), total=len(results), desc=f'Saving result frames'):
                result.save(f'{save_dir}/frame_{idx:06}.PNG')

        return results
