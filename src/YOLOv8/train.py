from utils.partition_dataset import partition_dataset
import YOLOv8

from os import path
import argparse
import json

def main(args):
    objd = YOLOv8.ObjectDetection()

    json_path = path.join('configs', 'YOLOv8', args.dataset, 'train.json')
    with open(json_path) as json_file:
        train_params = json.load(json_file)

    if args.dataset == 'NAPLab-LiDAR':
        # Currently, only NAPLab-LiDAR has the desired structure for the "partition_dataset" function
        partition_dataset(dataset_dir=path.join('datasets', args.dataset), force_repartition=False)

    objd.train(train_params)
    objd.model.export()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Script for training model.', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--dataset', type=str, default='NAPLab-LiDAR', 
                        help='Name of dataset \
                            \nDefault: NAPLab-LiDAR \
                            \nCheck the "configs" directory for available datasets: \
                            \n\nconfigs/YOLOv8/<name_of_dataset>')

    args = parser.parse_args()
    main(args)
