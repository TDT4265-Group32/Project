# Object Detection with LiDAR Data from Trondheim

## Introduction

This is a project for the course [TDT4265 - Computer Vision and Deep Learning](https://www.ntnu.edu/studies/courses/TDT4265#tab=omEmnet) at NTNU, Trondheim by Christian Le and Aleksander Klund

This repository contains the code for object detection on LiDAR data from the NAPLab at NTNU. The project is divided into two parts, where the first part is using YOLOv8 and the second part is using Faster R-CNN.

## Project Outline

The objective of this project is to perform object detection for 8 classes on a LiDAR dataset collected by the NAPLab at NTNU.

Specific for our project, we use two different architectures:

1. YOLOv8
2. Faster R-CNN

**NOTE**: The dataset themselves cannot be found in this repository, but within the devices used to train the models as redistribution of the dataset is prohibited.

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

### Tensorboard

To check the tensorboard when using SSH into a remote server, it is required to forward the port of the remote server to the local to check,
this can be done by doing the following in the **local machine**,

```bash
$ ssh -L 6006:localhost:6006 <remote address>
```

Here, it is assumed that the Tensorboard is located in "localhost:6006" on the remote server and it is forwarded to the local "6006" port.
When connected, the tensorboard can be accessed locally in "localhost:6006".


### Train, Validation and Test set

* Test set is chosen to be from frame 201 to frame 301 ("edge" frames are included), which represents motion from a seperate scene from other parts of the dataset (a good fit for a test set).
* All other remaining images are in the "datasets/NAPLab-LiDAR/images" folder which is free to be partitioned into "train" and "val" folders through "src/YOLOv8/utils/partition_dataset.py"

## Running code in the background

To run the training in the background, it is recommended to use TMUX. To start a new session, run

```bash
$ tmux new -s <session_name>
```

The usual session name is "group32" for our project. To detach from the session, press "Ctrl + B" and then "D". To reattach to the session, run

```bash
$ tmux attach -t <session_name>
```