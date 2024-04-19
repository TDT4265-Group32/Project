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
$ python src/YOLOv8/train.py --dataset <(str/name)optional>

$ python src/YOLOv8/val.py --model_path <(dir)required> --dataset <(str/name)optional>

$ python src/YOLOv8/predict.py --model_path <(dir)required> --dataset <(str/name)optional> --create_video <(bool)optional>
```

The default dataset is NAPLab-LiDAR and create_video is false, the rest needs to be specified.

The .json files can be found in the "configs" folder and contains changes to the default settings. A file containing all the default parameters for "train" and "validation" can be found in "configs/default_train-val.yaml" and for "predict" in "configs/default_pred.yaml".

## Tensorboard

To check the tensorboard when using SSH into a remote server, it is required to forward the port of the remote server to the local to check,
this can be done by doing the following in the **local machine**,

```bash
$ ssh -L 6006:localhost:6006 <remote address>
```

Here, it is assumed that the Tensorboard is located in "localhost:6006" on the remote server and it is forwarded to the local "6006" port.
When connected, the tensorboard can be accessed locally in "localhost:6006".


## Train, Validation and Test set

* Test set is chosen to be from frame 201 to frame 301 ("edge" frames are included), which represents motion from a seperate scene from other parts of the dataset (a good fit for a test set).
* All other remaining images are in the "datasets/NAPLab-LiDAR/images" folder which is free to be partitioned into "train" and "val" folders through "src/YOLOv8/utils/partition_dataset.py"