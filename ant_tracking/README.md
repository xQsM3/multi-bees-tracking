# Ant_Tracking

## Introduction

This repository contains code for online tracking ants. The approach is described in *Online Tracking of Ants Based on Deep Association
Metrics: Method, Dataset and Evaluation*. 


## Dependencies

The code is compatible with Python 2.7 and 3. The following dependencies are needed to run the tracker:

* NumPy
* sklearn
* OpenCV

Additionally, feature generation requires TensorFlow (>= 2.0). if your machine runs on 1.X and you have issues with this
code, download ant_tracking from https://github.com/holmesww/multi-ants-tracking and implement it in this framework

#


## generating detections with rcnn

python generate_detections.py --seq_dir=/home/linx123-rtx/multi-ants-tracking/ant_tracking/BUMBLEBEES/bbox_test


   
Check `python ant_tracking_app.py -h` for an overview of available options.


There are also scripts in the repository to visualize results, generate videos, and evaluate the MOT challenge benchmark.





## Generating appearance descriptors

Beside the main tracking application, this repository contains a script to generate features for ants identification, suitable to compare the visual appearance of ants�� bounding boxes using cosine similarity.
The following example generates these features from standard MOT challenge detections.

```
python tools/generate_appearance_descriptors.py \
    --model=resources/networks/ants.pb \
    --mot_dir=./ANTS/bbox_test \
    --output_dir=./resources/detections/
    
python tools/generate_appearance_descriptors.py \
    --model=resources/networks/bumblebees.pb \
    --mot_dir=./BUMBLEBEES/bbox_test \
    --output_dir=./resources/detections/    
```
The model has been generated with TensorFlow 1.4. If you run into
incompatibility, re-export the frozen inference graph to obtain a new
`ants.pb` that is compatible with your version:


## Running 

The repository contains the sample pre-training weights, stored in `./resources/networks/`. And generated features of detections, stored in `./resources/detections/`.
We assume the [ANTS] datasets have been stored in the repository root directory, which is including ``bbox_train.zip`` and ``bbox_test.zip``. Unzip ``bbox_test.zip`` to the ANTS directory for tracking.

```
python ant_tracking_app.py \
    --sequence_dir=./ANTS/bbox_test/Indoor1 \
    --detection_file=./resources/detections/Indoor1.npy \
    --min_confidence=0.3 \
    --nn_budget=100 \
    --display=True
    
python ant_tracking_app.py \
    --sequence_dir=./ANTS/bbox_test/Outdoor \
    --detection_file=./resources/detections/Outdoor.npy \
    --min_confidence=0.3 \
    --nn_budget=100 \
    --display=True
```

python ant_tracking_app.py \
    --sequence_dir=./BUMBLEBEES/bbox_test/video3 \
    --detection_file=./resources/detections/video3.npy \
    --min_confidence=0.3 \
    --max_cosine_distance=99999.999 \
    --nn_budget=500 \
    --display=True
    


```
python tools/freeze_model.py
```
The ``generate_detections.py`` stores for each sequence of the ANTS dataset a separate binary file in NumPy native format. Each file contains an array of shape `Nx138`, where N is the number of detections in the corresponding MOT sequence. The first 10 columns of this array contain the raw MOT detection copied over from the input file. The remaining 128 columns store the appearance descriptor. The files generated by this command can be used as input for the `ant_tracking_app.py`.


## Highlevel overview of source files

In the top-level directory are executable scripts to execute, evaluate, and visualize the tracker. The main entry point is in `ant_tracking_app.py`.
This file runs the tracker on a ANTS sequence.

In package `Ant_Tracking ` is the main tracking code:

* `detection.py`: Detection base class.
* `kalman_filter.py`: A Kalman filter implementation and concrete
   parametrization for image space filtering.
* `linear_assignment.py`: This module contains code for min cost matching and the matching cascade.
* `iou_matching.py`: This module contains the IOU matching metric.
* `nn_matching.py`: A module for a nearest neighbor matching metric.
* `track.py`: The track class contains single-target track data such as Kalman state, number of hits, misses, hit streak, associated feature vectors, etc.
* `tracker.py`: This is the multi-target tracker class.
* `ant_tracking_app.py` expects detections in a custom format, stored in .npy files. These can be computed from ANTS detections using `generate_detections.py`.


## Evaluation

The method for calculating MOT indicators in this article can be found [here]
( https://bitbucket.org/amilan/motchallenge-devkit/src/default/).

python evaluate_motchallenge.py     --mot_dir /home/linx123-rtx/multi-ants-tracking/ant_tracking/BUMBLEBEES/bbox_test --detection_dir=/home/linx123-rtx/multi-ants-tracking/ant_tracking/resources/detections --output_dir=/home/linx123-rtx/multi-ants-tracking/ant_tracking/tmp/output --min_confidence=0.3     --max_cosine_distance=99999.999     --nn_budget=100

## show paths 
python show_results.py --sequence_dir=./BUMBLEBEES/bbox_test/video1 --result_file=/home/linx123-rtx/multi-ants-tracking/ant_tracking/tmp/output/video1.txt --show_false_alarms=True

## save videos

python generate_videos.py --mot_dir=/home/linx123-rtx/multi-ants-tracking/ant_tracking/BUMBLEBEES/bbox_test --result_dir=/home/linx123-rtx/multi-ants-tracking/ant_tracking/tmp/output --output_dir=/home/linx123-rtx/multi-ants-tracking/ant_tracking/tmp/output



         
