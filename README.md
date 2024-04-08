# CV&DL Project by Christian Le and Aleksander Klund

Chosen project: Object detection with LiDAR Data from Trondheim

## Project Rules

1. The code should be developed by yourself, but you are free to use open-source repositories and libraries. If you take code from anywhere else, please attribute the original authors in your source code and write a notice in your report. 
2. You are allowed to use open-source architectures from PyTorch or MONAI etc. 
3. Annotating the test data by yourself and using it to train/validate your model is NOT allowed. 
4. It is allowed to use any pre-trained model. However, you cannot use models pre-trained on any of the respective datasets for each track. 
5. Training your final model should not take more than 12 hours on the IDUN cluster using 1 GPU or on the computers at the Cybele lab.

## Project Outline

The objective of this project is to perform object detection for 8 classes on a LiDAR dataset collected by the NAPLab at NTNU.

The dataset themselves cannot be found in this repository, but within the devices used to train the models.

Specific for our project, we use two different architectures:

1. YOLOv8
2. TBD

## Using YOLOv8

The scripts can be run using the CLI, where we currently have 3 different ones,

```bash
$ python src/YOLOv8/train.py --dataset_path <(dir)required> --reshuffle_dataset <(bool)optional> --train_config <(.json)optional>

$ python src/YOLOv8/val.py --model_path <(dir)required> --val_config <(.json)optional>

$ python src/YOLOv8/predict.py --model_path <(dir)required> --pred_config <(.json)optional> --results_path <(dir)optional> --create_video <(bool)optional>
```

The .json files can be found in the "configs" folder and contains changes to the default settings. A file containing all the default parameters for "train" and "validation" can be found in "configs/default_train-val.yaml" and for "predict" in "configs/default_pred.yaml".

