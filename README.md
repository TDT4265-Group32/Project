# Object Detection with LiDAR Data from Trondheim

## About the Project

This is a project for the course [TDT4265 - Computer Vision and Deep Learning](https://www.ntnu.edu/studies/courses/TDT4265#tab=omEmnet) at NTNU, Trondheim by Christian Le and Aleksander Klund.

This repository contains the code for object detection on LiDAR data from the NAPLab at NTNU. The project is divided into two main modules, each using **YOLOv8** and **Faster R-CNN**.

## About the Dataset

The dataset contains 19 clips of LiDAR data, where each clip contains ~10 seconds/~100 frames of data and contains 8 different classes, bounding boxes are annotated for each frame. The dataset is collected by the NAPLab at NTNU.
The objective of this project is to perform object detection on this dataset.

The test set is chosen to be from frame 201 to and including frame 301, which represents motion from a seperate scene from other parts of the dataset (a good fit for a test set).
All other remaining images are in the [datasets/NAPLab-LiDAR/images](datasets/NAPLab-LiDAR/images/) folder which is free to be partitioned into "train" and "val" folders through functions in [src/tools/data_partitioner.py](src/tools/data_partitioner.py).

**NOTE**: The dataset themselves cannot be found in this repository, but within the devices used to train the models as redistribution of the dataset is prohibited.

## Configuration

Before running the code, the configuration files need to be set up. The configuration files are in YAML format and contain the parameters for running the different modes of each architecture. The configuration files are located in the [configs/](configs/) folder.

### YOLOv8

The configuration files for the YOLOv8 model can be found in [configs/YOLOv8/](configs/YOLOv8). The configuration files are in YAML format and contain the parameters for training, validation, prediction and exporting.

The **"params"** section in each .yaml file needs to follow the standard format of YOLOv8 train, val and pred parameters. <u>[They can be found here](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/default.yaml)</u>. Unspecified parameters will be set to default values. Other fields are specific for this project.

#### [Train](configs/YOLOv8/train.yaml)

* **"model_path"**: Chosen model for the training. Usually the pretrained [yolov8m.pt](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/data/explorer/gui/dash.py#L37-L41) is used.
* **"loss_function"**: Contains booleans to indicate which loss functions to be used.

**NOTE**: Results of training are saved in "runs/detect/trainX", where "X" is specified in the terminal upon starting and ending training, where highest X is usually the most recent run. This can be changed by specifying "project" and "name" in "params".

#### [Val](configs/YOLOv8/val.yaml)

* **"model_path"**: Path to the model to be validated.

**NOTE**: Validation data are saved in "runs/detect/valX", where "X" follows the same logic as in training. This can be changed by specifying "project" and "name" in "params".

**NOTE**: The model is validated on the test set, which is specified in "data" in "params".

#### [Pred](configs/YOLOv8/pred.yaml)

* **"model_path"**: Path to the model to be used for prediction.
* **"video.create_video"**: Boolean to indicate whether to create a video from the predictions.
* **"video.path"**: Path to save the video.
* **"video.filename"**: Filename of the video.
* **"video.fps"**: Frames per second of the video.
* **"video.extension"**: Extension of the video, e.g. ".mp4, gif, etc.".

**NOTE**: Prediction images are by default saved in "results/nameOfArchitecture".

#### [Export](configs/YOLOv8/export.yaml)

* **"model_path"**: Path to the model to be exported.

**NOTE**: This function also attempts to export training, validation, prediction, configuration and emissions.csv to the [export](export/) folder (appears after running the export mode).

### Faster R-CNN

The configuration files for the Faster R-CNN model can be found in [configs/FasterRCNN/](configs/FasterRCNN). The configuration files are in YAML format and contain the parameters for training, validation, prediction and exporting.

**WORK IN PROGRESS**

## Running the Code

The scripts are run using the CLI,

```bash
$ python src/main.py --arch <str(required)> --mode <str(optional)> --model <str(optional)>
```

Explanation of the arguments:
* **--arch**: The architecture to be used. Can be "YOLOv8" or "FasterRCNN".
* **--mode**: The mode to be run. Can be "train", "val", "pred" and "export". If not specified, the script will run all modes.
* **--model**: Path to the model to be used. If not specified, the script will use models specified in the configuration files.

**TIP**: Often when these functions are performed on a remote machine, the export function is very useful to collect the results and models in one place, which can then be transferred by doing the following on the local machine,

```bash
$ scp -r <user>@<remote_address>:<path_to_export_folder> <local_path>
```

## Useful tools

### Tensorboard

YOLO supports tensorboard. There are multiple ways to check the tensorboard when using SSH into a remote server. 

#### Port Forwarding
Tensorboard can be checked on local machine by port forwarding by doing the following on the **local machine**,

```bash
$ ssh -L <port>:localhost:<port> <remote address>
```

Here, it is assumed that the Tensorboard is located in "localhost:<port>" on the remote server and it is forwarded to the local port. The port is usually given in the terminal when the tensorboard is started. When connected, the tensorboard can be accessed locally in [localhost:6006](https://localhost:6006) (6006 is an example).

#### In VSCode

Open the **command line** in VSCode (Ctrl + Shift + P) and type **"Python: Open Tensorboard"**. This will open a new tab in the VSCode where the tensorboard can be accessed.

#### In CLI

To start tensorboard, run the following command in the terminal,

```bash
$ tensorboard --logdir <path_to_tensorboard_logs>
```

The terminal will show the link to the tensorboard, which can be accessed in the browser.

## Running the Training in the Background

To run the training in the background and avoid interruptions due to possible issues with local machine, it is recommended to use TMUX. To start a new session, run

```bash
$ tmux new -s <session_name>
```

To detach from the session, press "Ctrl + B" and then "D". To reattach to the session, run

```bash
$ tmux attach -t <session_name>
```
