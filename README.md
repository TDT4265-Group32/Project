# Object Detection with LiDAR Data from Trondheim

by Christian Le and Aleksander Klund

## Project Outline

The objective of this project is to perform object detection for 8 classes on a LiDAR dataset collected by the NAPLab at NTNU.

The dataset themselves cannot be found in this repository, but within the devices used to train the models.

Specific for our project, we use two different architectures:

1. YOLOv8
2. TBD

## Using YOLOv8

The scripts can be run using the CLI,

```bash
$ python src/YOLOv8/main.py --mode <(str/mode)required> --dataset <(str/name)optional>
```

Currently, there are three different modes:
* train
* val
* pred

The dataset specifies which dataset they will run for and further configurations for the MODE-DATASET pair can be found in [configs](configs/YOLOv8/), where the ones related to LiDAR data can be found [here](configs/YOLOv8/NAPLab-LiDAR/).
The "params" section in each .json file needs to follow the standard format of YOLOv8 train, val and pred.

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

