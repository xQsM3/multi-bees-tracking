# vim: expandtab:ts=4:sw=4
from __future__ import division, print_function, absolute_import

import argparse
import os
import sys
import cv2
import numpy as np
import ntpath
current = os.path.dirname(os.getcwd())
sys.path.append(current)
sys.path.append(ntpath.join(current,"ant_tracking").replace("\\","/"))

from tools.generate_detections import generate_detections
from tools.generate_appearance_descriptors import get_appearance_descriptors
from application_util import preprocessing
from application_util import visualization
from ant_tracking.ant_tracking import nn_matching
from ant_tracking.ant_tracking.detection import Detection
from ant_tracking.ant_tracking.tracker import Tracker
from sequence import Sequence
import datetime

def gather_sequence_info(seq_dir, sequence):
    """Gather sequence information, such as image filenames, detections,
    groundtruth to pass it to the ant tracker.

    Parameters
    ----------
    sequence_dir : str
        Path to the MOTChallenge sequence directory.

    Returns
    -------
    Dict
        A dictionary of the following sequence information:

        * sequence_name: Name of the sequence
        * image_filenames: A dictionary that maps frame indices to image
          filenames.
        * appearances: A numpy array of detections bbox + features of appearance descriptors
        * groundtruth: A numpy array of ground truth in MOTChallenge format.
        * image_size: Image size (height, width).
        * min_frame_idx: Index of the first frame.
        * max_frame_idx: Index of the last frame.

    """
    image_dir = sequence.seq_dir
    image_filenames = {}
    for i,path in enumerate(sequence.frame_paths):
        image_filenames[i] = path
    groundtruth_file = None

    appearances = sequence.appearances

    groundtruth = None

    if len(image_filenames) > 0:
        image = cv2.imread(next(iter(image_filenames.values())),
                           cv2.IMREAD_GRAYSCALE)
        image_size = image.shape
    else:
        image_size = None

    if len(image_filenames) > 0:
        min_frame_idx = min(image_filenames.keys())
        max_frame_idx = max(image_filenames.keys())
    else:
        min_frame_idx = int(appearances[:, 0].min())
        max_frame_idx = int(appearances[:, 0].max())

    info_filename = os.path.join(seq_dir, "seqinfo.ini")
    if os.path.exists(info_filename):
        with open(info_filename, "r") as f:
            line_splits = [l.split('=') for l in f.read().splitlines()[1:]]
            info_dict = dict(
                s for s in line_splits if isinstance(s, list) and len(s) == 2)

        update_ms = 1000 / int(info_dict["frameRate"])
    else:
        update_ms = None
        
    feature_dim = appearances.shape[1] - 10 if appearances is not None else 0
    seq_info = {
        "sequence_name": sequence.sequence_name,
        "image_filenames": image_filenames,
        "appearances": appearances,
        "groundtruth": groundtruth,
        "image_size": image_size,
        "min_frame_idx": min_frame_idx,
        "max_frame_idx": max_frame_idx,
        "feature_dim": feature_dim,
        "update_ms": update_ms
    }
    return seq_info

def create_appearances(appearances_mat, frame_idx, min_height=0):
    """Create detections for given frame index from the raw detection matrix.

    Parameters
    ----------
    appearances_mat : ndarray
        Matrix of detections + appearance descriptors. The first 10 columns of the detection matrix are
        in the standard MOTChallenge detection format. In the remaining columns
        store the feature vector (appearance descriptors) associated with each detection.
    frame_idx : int
        The frame index.
    min_height : Optional[int]
        A minimum detection bounding box height. Detections that are smaller
        than this value are disregarded.

    Returns
    -------
    List[tracker.Detection]
        Returns detection responses at given frame index.

    """
    frame_indices = appearances_mat[:, 0].astype(np.int)
    mask = frame_indices == frame_idx

    appearances_list = []
    for row in appearances_mat[mask]:
        bbox, confidence, feature = row[2:6], row[6], row[10:]
        if bbox[3] < min_height:
            continue
        appearances_list.append(Detection(bbox, confidence, feature))
    return appearances_list

def run(seq_dir,
        nms_max_overlap, min_detection_height, max_cosine_distance,
        nn_budget, conf_thresh,bs,app_model,display):
    """Run multi-target tracker on a particular sequence.

    Parameters
    ----------
    seq_dir : str
        Path to the MOTChallenge sequence directory.
    min_confidence : float
        Detection confidence threshold. Disregard all detections that have
        a confidence lower than this value.
    nms_max_overlap: float
        Maximum detection overlap (non-maxima suppression threshold).
    min_detection_height : int
        Detection height threshold. Disregard all detections that haveTypeError: run() missing 1 required positional 
        a height lower than this value.
    max_cosine_distance : float
        Gating threshold for cosine distance metric (object appearance).
    nn_budget : Optional[int]
        Maximum size of the appearance descriptor gallery. If None, no budget
        is enforced.
    display : bool
        If True, show visualization of intermediate tracking results.

    """
    sequence = Sequence(seq_dir)
    # predict detections
    sequence.detections = generate_detections(seq_dir,conf_thresh,bs)
    # predict appearance descriptors
    sequence.appearances = get_appearance_descriptors(sequence,app_model)
    # safe information in ant tracker syntax such that the tracker can handle it
    seq_info = gather_sequence_info(seq_dir, sequence)
    metric = nn_matching.NearestNeighborDistanceMetric(
        "cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)
    results = []
    results_centered = []
    def frame_callback(vis, frame_idx):
        
        
        # Load image and store appearances in Detector object.
        appearances = create_appearances(
            seq_info["appearances"], frame_idx, min_detection_height)
        #appearances = [d for d in appearances if d.confidence >= min_confidence]
        # Run non-maxima suppression.
        boxes = np.array([d.tlwh for d in appearances])
        scores = np.array([d.confidence for d in appearances])
        indices = preprocessing.non_max_suppression(
            boxes, nms_max_overlap, scores)
        appearances = [appearances[i] for i in indices]

        # Update tracker.
        tracker.predict()
        # tracker.update(detections)
        tracker.update(appearances, frame_idx)

        # Update visualization.
        if display:
            print("Visualize frame {0}".format(frame_idx),end="\r")
            image = cv2.imread(
                seq_info["image_filenames"][frame_idx], cv2.IMREAD_COLOR)
            vis.set_image(image.copy())
            vis.draw_detections(appearances)
            vis.draw_trackers(tracker.tracks)
        # Store results.
        for track in tracker.tracks:
            if (not track.is_confirmed() or track.time_since_update > 1) and frame_idx > 1:  # 第一帧参与
                continue
            bbox = track.to_tlwh()
            results.append([
                frame_idx, track.track_id, bbox[0], bbox[1], bbox[2], bbox[3]])
            results_centered.append([
                frame_idx, track.track_id, round(bbox[0]+bbox[2]/2), round(bbox[1]+bbox[3]/2)])            
    # Run tracker.
    if display:
        visualizer = visualization.Visualization(seq_info, update_ms=5)
    else:
        visualizer = visualization.NoVisualization(seq_info)

    visualizer.run(frame_callback)

    sequence.tracks = np.array(results_centered)
    cv2.destroyAllWindows()
    return sequence
    

def parse_args():
    """ Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Deep SORT")
    parser.add_argument(
        "--seq_dir", help="Path to MOTChallenge sequence directory",
        default=None, required=True)
    parser.add_argument(
        "--min_detection_height", help="Threshold on the detection bounding "
        "box height. Detections with height smaller than this value are "
        "disregarded", default=0, type=int)
    parser.add_argument(
        "--nms_max_overlap",  help="Non-maxima suppression threshold: Maximum "
        "detection overlap.", default=1.0, type=float)
    parser.add_argument(
        "--max_cosine_distance", help="Gating threshold for cosine distance "
        "metric (object appearance).", type=float, default=0.2)
    parser.add_argument(
        "--nn_budget", help="Maximum size of the appearance descriptors "
        "gallery. If None, no budget is enforced.", type=int, default=None)
    parser.add_argument(
        "--conf_thresh", help="confidence threshold "
        "gallery. If None, no budget is enforced.", type=int, default=0.95)
    parser.add_argument(
        "--batch_size", help="batch size for detection, currently only bs=1 working ", type=int,default=1)
    parser.add_argument(
        "--appearance_model", help="path to appearance model", default="/home/linx123-rtx/multi-ants-tracking/ant_tracking/resources/networks/bumblebees.pb")   
    parser.add_argument(
        "--display", help="Show intermediate tracking results",
        default=True, type=bool)
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    starttime = datetime.datetime.now()

    sequence = run(
        args.seq_dir,args.nms_max_overlap,args.min_detection_height, 
        args.max_cosine_distance, args.nn_budget,args.conf_thresh,args.batch_size, args.appearance_model,args.display)
        
    endtime = datetime.datetime.now()
    print(endtime - starttime)
