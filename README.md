## multi-bees-tracking

multi-bees-tracking is an implementation of the multi-ants-tracking https://github.com/holmesww/multi-ants-tracking
it can track on single sequences or on a dataset containing multiple sequences, reaching state of the art benchmarks
running on a dataset with multiple sequences, the framework uses 3 cameras to reconstruct the 3 dimensional coordinates 
## tested on machine
RTX 3090
CUDA 11.0
tensorflow 2.X
ubuntu 20.04

## Installation

# bash installation

clone repository
$ cd multi-bees-tracking
$ bash install.sh

# trouble shooting to install the tracker on RTX

then run the official tensorflow upgrade notebook on the tracker package (just google upgrade tensorflow v1->v2) to upgrade code from tensorflow v1 to v2. add save_format='h5' to the save() functions as suggested in the
output print of the upgrade notebook

for the error:
No module named 'tensorflow.contrib'
on tensorflow v2:
1) install tf_slim https://github.com/google-research/tf-slim 
2) replace import tensorflow.contrib.slim as slim with: 
import tf_slim as slim

for the error:
RuntimeError: tf.placeholder() is not compatible with eager execution.
on tensorflow v2:
replace the error line with:
tf.compat.v1.disable_eager_execution()

for the error:
The name 'net/images:0' refers to a Tensor which does not exist. 
change:
Try to change "net/%s:0" => "%s:0" 83 & 85 lines in 'tools/generate_detections.py 
pip install scikit-learn==0.22.2


for the error:
save() got an unexpected keyword argument 'save_format'
delete:
save_format='h5' in the corresponding np.save call

## models 

download the appearance descriptor model for bumblebees:
https://fh-aachen.sciebo.de/s/azh8VO32sH3GRvG
this model was trained by the API in ./cosine_metric_learning
put the model in directory:
./ant_tracking/resources/networks


download the detector model for bumblebees:
https://fh-aachen.sciebo.de/s/azh8VO32sH3GRvG
this model was trained by the Detectron2 API
put the model in directory:
./ant_tracking/rcnn/models
  
## datasets
the datasets for both detection and appearance description are published here:
https://fh-aachen.sciebo.de/s/azh8VO32sH3GRvG


## Usage

the repository contains an implementation of the original ant tracker and the novel bumblebee tracker. 

you can evaluate the tracker on a single sequence by following the Readme in:
./ant_tracking/README.md

you can run the tracker on a set of sequences by following the readme in:
./bb_framework/README.md

you can calculate benchmarks for the tracker following the readme in:
./py-motmetrics/README.md




## Authors
Luc Stiemer (email luc.stiemer@alumni.fh-aachen.de)
Andreas Thoma (email a.thoma@fh-aachen.de)

feel free to contact us if you have issues with the tracker
