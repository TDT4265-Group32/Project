from ultralytics import YOLO

import os
import torch
import cv2

"""
This example script was inspired by Ultralytics tutorial on how to use yolov8
"""

class ObjectDetection:
    
    def __init__(self):
        
        # Use this if we want to run OBD on camera
        # self.capture_index = capture_index
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Using Device: ", self.device)
        
        self.model = self.load_model()

    def load_model(self, model_path='models/pretrained/yolov8n.pt'):
        
        model = YOLO(model_path)
        model.fuse()
        
        return model
    
    def train(self, yaml_path, epochs=100, imgsz=640):
        """
        For more, check out: https://docs.ultralytics.com/modes/train/
        """
        result = self.model.train(data=yaml_path, epochs=epochs, imgsz=imgsz, device=self.device)
        
        return result

    def run_inference(self, show=True, conf=0.4):
        """
        For more, check out: https://docs.ultralytics.com/modes/predict/#key-features-of-predict-mode
        """
        self.model(source=0, show=show, conf=conf)
        
        # for result in results:
        #     boxes = result.boxes
        #     masks = result.masks
        #     keypoints = results.keypoints
        #     probs = results.probs
    

    def predict(self, path, show=False, conf=0.4, save_path=None):
        """
        For more, check out: https://docs.ultralytics.com/modes/predict/#key-features-of-predict-mode
        """
        # Split the file path into root and extension

        results = self.model(source=path, show=show, conf=conf)
        
        if save_path is not None:
            root, ext = os.path.splitext(save_path)

            # If the extension is not empty, get the directory name
            if ext:
                save_dir = os.path.dirname(root)
            else:
                save_dir = root
        
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            for idx, result in enumerate(results):
                result.save(f'{save_dir}/prediction_{idx}.PNG')

        return results
    
    # def plot_bboxes(self, results, frame):
        
    #     xyxys = []
    #     confidences = []
    #     class_ids = []
        
    #     # Extract detections for person class
    #     # Converting to cpu and numpy for plotting
    #     for result in results:
    #         boxes = result.boxes.cpu().numpy()
            
    #         xyxys.append(boxes.xyxy)
    #         confidences.append(boxes.conf)
    #         class_ids.append(boxes.cls)
            
            
    #     return results[0].plot(), xyxys, confidences, class_ids

    def export_model(self):
        """
        For more, check out: https://docs.ultralytics.com/modes/export/
        """
        self.model.export()
