from partition_dataset import partition_dataset
import YOLOv8
import random
random.seed(0)

def main():

    objd = YOLOv8.ObjectDetection()
    partition_dataset(dataset_dir='datasets/NAPLab-LiDAR', force_repartition=False)
    objd.train('data/NAPLab-LiDAR.yaml', epochs=500, imgsz=640)
    #objd.load_model('runs/detect/train6/weights/best.pt')
    objd.predict('datasets/NAPLab-LiDAR/images/', save_path='results/NAPLab-LiDAR/')
    objd.export_model()

if __name__ == "__main__":
    main()
