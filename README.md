# Object Detection with LiDAR Data from Trondheim

## About the Project

This is a project for the course [TDT4265 - Computer Vision and Deep Learning](https://www.ntnu.edu/studies/courses/TDT4265#tab=omEmnet) at NTNU, Trondheim by Christian Le and Aleksander Klund.

This repository contains the code for object detection on LiDAR data from the NAPLab at NTNU. The project is divided into two main modules, each using **YOLOv8** and **Faster R-CNN**.

## About the Dataset

The dataset contains 19 clips of LiDAR data, where each clip contains ~10 seconds/~100 frames of data and contains 8 different classes, bounding boxes are annotated for each frame. The dataset is collected by the NAPLab at NTNU.
The objective of this project is to perform object detection on this dataset.

**NOTE**: The dataset themselves cannot be found in this repository, but within the devices used to train the models as redistribution of the dataset is prohibited.

## Configuration

Before running the code, the configuration files need to be set up. The configuration files are in YAML format and contain the parameters for running the different modes of each architecture. The configuration files are located in the [configs/](configs/) folder.

### YOLOv8

The configuration files for the YOLOv8 model can be found in [configs/YOLOv8/](configs/YOLOv8). The configuration files are in JSON format and contain the parameters for training, validation and prediction.

The **"params"** section in each .json file needs to follow the standard format of YOLOv8 train, val and pred parameters. They can be found [here](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/default.yaml). Unspecified parameters will be set to default values. Other fields are specific for this project.

#### [Train](configs/YOLOv8/NAPLab-LiDAR/train.json)

* **"pretrained_model"**: Chosen pretrained model for the training. Usually "yolov8m.pt" is used.
* **"loss_function"**: Contains booleans to indicate which loss functions to be used.
* **partition.mode**: The mode of partitioning the dataset. Can be "video" or "images", where "video" partitions the 18 clips, while "images" partition for all images independently.
* **partition.num_shuffles**: Number of times the dataset is shuffled, thus training is performed the same number of times.
Epochs per training is given by ceil(epochs / num_shuffles).

**NOTE: Results of training are saved in "runs/detect/trainX", where "X" is specified in the terminal upon starting and ending training.**

#### [Val](configs/YOLOv8/NAPLab-LiDAR/val.json)

* **"model_path"**: Path to the model to be validated.

**NOTE: Validation data is vaed in "runs/detect/valX", where "X" is specified in the terminal upon starting and ending validation.**

#### [Pred](configs/YOLOv8/NAPLab-LiDAR/pred.json)

* **"model_path"**: Path to the model to be used for prediction.
* **"video.create_video"**: Boolean to indicate whether to create a video from the predictions.
* **"video.save_path"**: Path to save the video.
* **"video.filename"**: Filename of the video.

**NOTE: Prediction images are by default saved in "results/NameOfDataset".**

## Using YOLOv8

The scripts can be run using the CLI,

```bash
$ python src/main.py --mode <(str/mode)required> --dataset <(str/name)optional>
```

Currently, there are three different modes:
* train
* val
* pred

The dataset specifies which dataset they will run for. More information of configurations can be found [[here]](#configuration).



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

### Train, Validation and Test set

* Test set is chosen to be from frame 201 to frame 301 ("edge" frames are included), which represents motion from a seperate scene from other parts of the dataset (a good fit for a test set).
* All other remaining images are in the "datasets/NAPLab-LiDAR/images" folder which is free to be partitioned into "train" and "val" folders through functions in [partition_dataset.py](src/YOLOv8/utils/partition_dataset.py).



## Running code in the background

To run the training in the background and avoid interruptions due to possible issues with local machine, it is recommended to use TMUX. To start a new session, run

```bash
$ tmux new -s <session_name>
```

To detach from the session, press "Ctrl + B" and then "D". To reattach to the session, run

```bash
$ tmux attach -t <session_name>
```
