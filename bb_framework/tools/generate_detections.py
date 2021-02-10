# import standard libraries
from torchvision.transforms import transforms
from PIL import Image
from pathlib import Path
import cv2 as cv
import ntpath
import argparse
import datetime
import os
import errno
import numpy as np
import glob
import sys
# import torch
import torch
from torch.autograd import Variable
# import modules
from rcnn.load_rcnn_model import load_model
from rcnn.detector import Detector




def generate_detections(seq_dir,conf_thresh,bs):
    """Generate detections with rcnn detector on a single sequence.

    Parameters
    ----------
    seq_dir : str
        path to the sequence.
    conf_thresh
        confidence score threshold for detector
    bs
        batch size 
    """
    

    
    sequence = ntpath.basename(seq_dir)

    # get model predictor object        
    model,predictor = load_model(float(conf_thresh))
    detector = Detector(model,predictor)

    # detection list
    det_list = []
    print("Processing %s" % sequence)
    image_filenames = sorted(glob.glob(seq_dir+"/*.jpg"))

    pointer = 0
    while pointer <len(image_filenames):
        if pointer+bs>len(image_filenames):
            bs = len(image_filenames)-pointer

        batch = image_filenames[pointer:pointer+bs]
        detector.predict_on_batch(batch)

        print("Frame %05d/%05d" % (pointer, len(image_filenames)))
        pointer+=bs

    detector.outputs_instances_to_cpu()
    detector.force_bbox_size()
    for frame_idx,output in enumerate(detector.outputs_cpu):
        for box_pred,score_pred,classes_pred in \
        zip(output["pred_boxes"],output["scores"],output["pred_classes"]):
            det_list.append([frame_idx,-1,round(box_pred[0]),round(box_pred[1]),
                             round(box_pred[2]),round(box_pred[3]),1])
    return detector.outputs_cpu
