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
        
        model = YOLO('models/pretrained/yolov8n.pt')
        #model.fuse()
        
        return model

    def run_inference(self, source=1, show=True, conf=0.4, save=True):
        """
        For more, check out: https://docs.ultralytics.com/modes/predict/#key-features-of-predict-mode
        """
        results = self.model(source=source, show=show, conf=conf, save=save)
        
        # for result in results:
        #     boxes = result.boxes
        #     masks = result.masks
        #     keypoints = results.keypoints
        #     probs = results.probs
    

    # def predict(self, frame):
    #     """
    #     For more, check out: https://docs.ultralytics.com/modes/predict/#key-features-of-predict-mode
    #     """
    #     results = self.model(frame)
        
    #     return results
    
    # def plot_bboxes(self, results, frame):
        
    #     xyxys = []
    #     confidences = []
    #     class_ids = []
        
    #     # Extract detections for person class
    #     # Converting to cpu and numpy for plotting
    #     for result in results:
    #         boxes = result.boxes.cpu().numpy()
            
    #         # Bounding boxes (index 0-3 contains the coordinate for the four corners)
    #         #xyxys = boxes.xyxy
            
    #         #for xyxy in xyxys:
    #             # cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])), (0, 255, 0))
    #             # confidences.append(xyxy[4])
    #             # class_ids.append(xyxy[5])
            
    #         xyxys.append(boxes.xyxy)
    #         confidences.append(boxes.conf)
    #         class_ids.append(boxes.cls)
            
            
    #     return results[0].plot(), xyxys, confidences, class_ids
    
if __name__ == 'main':
    objd = ObjectDetection()
    
    objd.run_inference()