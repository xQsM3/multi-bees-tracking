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
import datetime
# import torch
import torch
from torch.autograd import Variable
# import modules
from detector.load_model import load_model
from detector.detector import Detector



def generate_detections(det_model,seq_dir,conf_thresh,bs,imdim):
    """Generate detections with rcnn/retina detector on a single sequence.

    Parameters
    ----------
    seq_dir : str
        path to the sequence.
    conf_thresh
        confidence score threshold for detector
    bs
        batch size 
    """

    # get model predictor object       
    model,predictor = load_model(float(conf_thresh),det_model)
    detector = Detector(model,predictor)

    # detection list
    det_list = []
    #print("Processing %s" % sequence)
    image_filenames = sorted(glob.glob(seq_dir+"/*.jpg"))

    # frame pointer
    pointer = 0

    while pointer <len(image_filenames):
        if pointer+bs>len(image_filenames):
            bs = len(image_filenames)-pointer

        #slice image filenames to batch
        batch = image_filenames[pointer:pointer+bs]
        #get system time before prediction
        starttime = datetime.datetime.now()
        #predict on batch
        detector.predict_on_batch(batch,imdim)
        #compute frames / seconds fp/s
        sec = (datetime.datetime.now()-starttime).total_seconds()
        fps = len(batch) / sec

        print("generate detections in frame %05d/%05d \
                %01f [fp/s]" % (pointer,len(image_filenames),
                                                          fps),end="\r")
        pointer+=bs
    detector.outputs_instances_to_cpu()
    '''
    for frame_idx,output in enumerate(detector.outputs_cpu):
        for box_pred,score_pred,classes_pred in \
        zip(output["pred_boxes"],output["scores"],output["pred_classes"]):
            det_list.append([frame_idx,-1,round(box_pred[0]),round(box_pred[1]),
                             round(box_pred[2]),round(box_pred[3]),1])
    '''
    return detector.outputs_cpu
