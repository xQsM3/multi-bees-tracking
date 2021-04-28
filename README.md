# multi-bees-tracking

multi-bees-tracking is a python implementation of the multi-ants-tracking https://github.com/holmesww/multi-ants-tracking method.  
It can track on single sequences or on a dataset containing multiple sequences, reaching state of the art benchmarks  
running on a dataset with multiple sequences, the framework uses 3 cameras to reconstruct the 3 dimensional coordinates of the bumblebees  
by triangulation  
  
the tracker is based on Tracking-By-Detection. The detections are provided by a Faster R-CNN model with ResNeXt-101 backbone. The  
detector model was trained with the Detectron 2 API, using >10000 bumblebee images. The data association between the detections 
and the existing trajectories is done by an implementation of the multi-ant tracker. It consist of two branches, one for  
motion matching and one for appearance matching. The motion matcher is based on a Kalman Filter exploiting a constant velocity  
model. The appearance matcher uses a fully convolutional neural network for appearance description and cosine similarity matching  
between the detections and the trajectories. 3D reconstruction is then performed by triangulation, using camera calibration matrices  
provided by matlab calibrator app.  


## tested on machine  
RTX 3090  
CUDA 11.0  
tensorflow 2.X  
ubuntu 20.04  

# Installation  
  
## bash installation  

clone repository  
$ cd multi-bees-tracking  
$ bash install.sh  
  
## trouble shooting to install the tracker on RTX  
  
then run the official tensorflow upgrade notebook on the tracker package (just google upgrade tensorflow v1->v2) to upgrade code from tensorflow v1 to v2.  
add save_format='h5' to the save() functions as suggested in the output print of the upgrade notebook  
note that this upgrade procedure was allready conducted for this repository  
  
  
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
  
# models  
  
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
    
# datasets  
the datasets for both detector and appearance descriptor training are published here:  
https://fh-aachen.sciebo.de/s/azh8VO32sH3GRvG  

the bumblebee training dataset for the faster R-CNN detector consists of over 10 000 images with 15 000 annotations
the bumlbebee training dataset for the appearance descriptor consists of 144 images with 12 bumblebee identities
  
# Usage  
  
the repository contains an implementation of the original ant tracker and the novel bumblebee tracker.  
  
you can evaluate the tracker on a single sequence by following the Readme in:  
./ant_tracking/README.md  
  
you can run the tracker on a set of sequences by following the readme in:  
./bb_framework/README.md  
  
you can calculate benchmarks for the tracker following the readme in:  
./py-motmetrics/README.md  
  
  
  
  
# Authors  
Luc Stiemer (email luc.stiemer@alumni.fh-aachen.de)  
Andreas Thoma (email a.thoma@fh-aachen.de)  
  
feel free to contact us if you have issues with the tracker  
  
