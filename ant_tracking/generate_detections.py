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
from detector.load_model import load_model
from detector.detector import Detector




def generate_detections(seq_dir,conf_thresh,model_name,bs):
    """Generate detections with rcnn/retina detector on sequences.

    Parameters
    ----------
    seq_dir : str
        Path to the MOTChallenge sequences.
    output_dir
        Path to the output directory. Will be created if it does not exist.
    conf_thresh
        confidence score threshold for detector
    """
    

    
    for sequence in os.listdir(seq_dir):
        
        # get model predictor object        
        model,predictor = load_model(float(conf_thresh),model_name)
        detector = Detector(model,predictor)
        
        # detection list
        det_list = []
        print("Processing %s" % sequence)
        sequence_dir = os.path.join(seq_dir, sequence)
        image_dir = os.path.join(sequence_dir, "img"+sequence[-1::])
        image_filenames = sorted(glob.glob(image_dir+"/*.jpg"))
        
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
            
        

        with open(ntpath.join(seq_dir,sequence,"det/det.txt").replace("\\","/"),"w") as f:
            for row in det_list:
                string = ','.join(list(map(str,row)))
                string +="\n"
                f.write(string)


def parse_args():
    """Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="RCNN/Retina detector")
    parser.add_argument(
        "--model",
        default="rcnn",
        help="model name, either retina or rcnn (retina is faster)")
    parser.add_argument(
        "--seq_dir", help="Path to sequences with structure <seq_dir>/sequenceX/img1/000000X.jpg",
        required=True)
    parser.add_argument(
        "--conf_thresh", default=0.95,help="confidence score threshold for detector")
    parser.add_argument(
        "--bs",default=1,help="batch size")
    return parser.parse_args()

def main():
    args = parse_args()
    starttime = datetime.datetime.now()
    generate_detections(args.seq_dir,args.conf_thresh,args.model,args.bs)
    endtime = datetime.datetime.now()
    print(endtime - starttime)

if __name__ == "__main__":
    main()
