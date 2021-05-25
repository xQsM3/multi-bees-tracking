# Luc Stiemer 2020
# THIS SCRIPT RUNS TRACKING AND RECONSTRUCTION WITH AUTO DETECT MOTION ALGORITHM

#import python modules
import glob
import os
import ntpath
from pathlib import Path
import sys
import argparse
import datetime
from tqdm import tqdm
import traceback
#import own modules
import tracking_app
import tools.track_matching as track_matching
import sequence
import tools.loadData
from log import info
from tools import loadData
from tools import reconstruct
from tools import plot
from log import log

def main_loop(main_dir,
        nms_max_overlap, min_detection_height, max_cosine_distance,
        nn_budget, conf_thresh,bs,app_model,det_model,imdim,display,app_resize):
    
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
    #global values
    cameras = '159','160','161'
    maskCoordinates = {}
    videosTicked = 0 #for progress bar in terminal
    
    #print welcome info
    info.welcome()


    #get directories for analysis
    if Path(main_dir).is_dir():
        day_dirs = sorted(glob.glob(main_dir+'/*') )
    else:
        sys.exit("[WARNING]: main_dir directory does not exist")

    #check if boundary Conditions are available (calibration and masks) and counts video total quantity
    videosTotal,day_dirs = loadData.checkData(day_dirs)

    #initialize Logger object
    logfile = log.Logger()

    #ask if user wants to continue 
    if not input('[INPUT:] press y to start tracking for every folder with located calibrations\n') == 'y':
        sys.exit()

    
    ## START ANALYSIS
    
    # iterate through days
    for day_dir in day_dirs:

        #inform user about process
        info.info('analysing day '+ntpath.basename(day_dir+"\n"))

        ## READ IN
        #read masks
        maskCoordinates['159'],maskCoordinates['160'],maskCoordinates['161'] = loadData.readMasks(day_dir)
        #read calib data of stereo pair a "159160"
        K159_a,K160_a,D159_a,D160_a,R_a,T_a,F_a,E_a,calibImageSize = loadData.readCalib(day_dir,159,160)
        #read calib data of stereo pair b "159161"
        K159_b,K161_b,D159_b,D161_b,R_b,T_b,F_b,E_b,calibImageSize = loadData.readCalib(day_dir,159,161)

        #get seq directories in a dictionary with key 159,160,161 for each camera
        seq_dirs_dic = loadData.getVideoDirectory(day_dir)
        #if quantity of videos of each camera are not equal, give error and skip date
        if not (len(seq_dirs_dic['159']) == 
                len(seq_dirs_dic['160']) == 
                len(seq_dirs_dic['161'])):
                info.error('quantity of videos of the cameras 159/160/161 not equal for the day ' 
                           + ntpath.basename(day_dir))
                continue

        ## START TRACKING
        # iterate through sequences
        pbar = tqdm( seq_dirs_dic['159'])
        for i,seq_dir in enumerate(pbar):
            seq_start_time = datetime.datetime.now()
            try:

                # init tracking for the current sequence (from each camera angle)
                sequence_dic = {}
                for cam in cameras:
                    sequence_dic[cam] = tracking_app.run(seq_dirs_dic[cam][i],nms_max_overlap, 
                                     min_detection_height, max_cosine_distance,nn_budget, 
                                     conf_thresh,bs,app_model,det_model,imdim,display,calibImageSize,app_resize)
                if len(sequence_dic['159'].frame_paths) != len(sequence_dic['160'].frame_paths) or \
                    len(sequence_dic['159'].frame_paths) != len(sequence_dic['161'].frame_paths):
                    info.warning('sequence 159 / 160 / 161 different length for '+sequence_dic['159'].sequence_name[:-6])
                    continue
                # calibration data into calib objects
                stereo159160 = reconstruct.SceneReconstruction3D(K159_a,D159_a,K160_a,D160_a,R_a,T_a,calibImageSize)
                stereo159161 = reconstruct.SceneReconstruction3D(K159_b,D159_b,K161_b,D161_b,R_b,T_b,calibImageSize)

                #match 2D tracks between cameras
                info.info("match 2D tracks                                                              ")
                match_matrix159160 = track_matching.match2D(sequence_dic["159"],sequence_dic["160"],stereo159160)
                match_matrix159161 = track_matching.match2D(sequence_dic["159"],sequence_dic["161"],stereo159161)

                info.info("perform reconstruction")
                sequence_dic["159160"] = sequence.Sequence3D(seq_dir,"159","160",match_matrix159160)
                sequence_dic["159160"].tracks = stereo159160.triangulate_tracks(sequence_dic["159"],sequence_dic["160"],sequence_dic["159160"])

                sequence_dic["159161"] = sequence.Sequence3D(seq_dir,"159","161",match_matrix159161)
                sequence_dic["159161"].tracks = stereo159161.triangulate_tracks(sequence_dic["159"],sequence_dic["161"],sequence_dic["159161"])

                info.info("draw 3D plot")
                plot.save_3D_plt(day_dir,sequence_dic["159160"],sequence_dic["159161"],display)

                info.info("save tracks")
                for key in sequence_dic:
                    sequence_dic[key].write_sequence(day_dir)


            except Exception as e:
                info.error("unexpected error occured while analysing video "+
                           ntpath.basename(ntpath.basename(seq_dir))[:-5]+"in day "+ ntpath.basename(day_dir))
                print(traceback.format_exc())

            total_fps = 3*sequence_dic['159'].max_frame_idx/(datetime.datetime.now()-seq_start_time).total_seconds()
            pbar.set_description("OVERALL SPEED %01f [fp/s]" % total_fps)
            #updating progress bar ticker
            videosTicked += 3


def parse_args():
    """ Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="MOT with RCNN")
    parser.add_argument(
        "--main_dir", help='''Path to sequences with following structure:\n
        <MAIN_DIR>/<DAY_X>/<SEQUENCE_X>/<FRAME_X> \n
        where <SEQUENCE_X> should have the structure: <VX_YYYYYYYYY_ZZZ_1> \n
        where X is the number of the sequence YYYYYYYYYY is an arbitrary name of \n
        the sequence, YYY is either 159,160 or 161 (the number of camera) and _1 in the\n
        end has to be there allways. \n
        <FRAME_X> has structure <VX_YYYYYYYYY_ZZZ_0000000W>, W is the frame number''',
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
        "--app_resize", help='''appearance bounding box resize. e.g. 0.2 increases the detection bboxes by 20%,\n"
                              -0.2 decreases. the resized bboxes are the input for the appearance descriptor. Default is 0% (no resizing).
                              reducing the size can improve inference time of appearance descriptor. however, it is not remommended''', type=float,default=False)
    parser.add_argument(
        "--conf_thresh", help="confidence threshold "
        "gallery. If None, no budget is enforced.", type=float)
    parser.add_argument(
        "--batch_size", help="batch size for detection, currently only bs=1 working ", type=int,default=1)
    parser.add_argument(
        "--appearance_model", help="path to appearance model", default="./ant_tracking/resources/networks/bumblebees.pb")
    parser.add_argument(
        "--detection_model", help="name of detection model, choose yolov5l,rcnn or retina", default="yolov5l")
    parser.add_argument(
        "--im_dim", help="image largest dimension (either width or height, aspect ration will remain same)", type=int,default=640)
    parser.add_argument(
        '--display', help="Show intermediate tracking results",
        default=True, action='store_true')
    parser.add_argument(
        '--no-display',
        dest='display', action='store_false')
    return parser.parse_args()
    
if __name__ == '__main__':
    args = parse_args()
    starttime = datetime.datetime.now()

    print(args.appearance_model)
    print(args.detection_model)
    #replace ./ in appearance_model directory 
    if args.appearance_model[0] == '.':
    	basedir = ntpath.dirname( os.path.dirname(os.path.abspath( __file__ )) )
    	args.appearance_model = basedir + args.appearance_model[1::]
    	
    main_loop(
        args.main_dir,args.nms_max_overlap,args.min_detection_height, 
        args.max_cosine_distance, args.nn_budget,args.conf_thresh,args.batch_size,
        args.appearance_model,args.detection_model,args.im_dim,args.display,args.app_resize)
