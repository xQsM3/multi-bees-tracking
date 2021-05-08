import glob
import cv2 as cv
import os
import pandas as pd
import ntpath
import numpy as np
import traceback

from log import info

class Sequence():
    '''
    stores informations of 2d sequence such as detections, descriptors tracks..
    '''
    def __init__(self,seq_dir):
        frame_paths = sorted(glob.glob(seq_dir+"/*.jpg") )
        image_shape = cv.imread(frame_paths[0]).shape
        max_frame_idx = len(frame_paths)
        
        self.seq_dir = seq_dir # string with directory path
        self.sequence_name = os.path.basename(seq_dir) # basename of directory path
        self.frame_paths = frame_paths # list of frame paths
        self.image_shape = image_shape # tuple (h,w,c)
        self.frame_width = image_shape[1]
        self.frame_height = image_shape[0]
        self.min_frame_idx = 0 # int
        self.max_frame_idx = len(frame_paths)-1 # int
        self.detections = [] 
        self.appearances = [] #Matrix of detections+appearance descriptor. The first 10 columns of the detection matrix are the detections.
        self.tracks = [] #np.array with [frame_idx, track_id, x,y]
    def resize_tracks(self,CalibrationImageSize):
        # resize track pixel coordinates if calibration image size varies from inference frame size
        if CalibrationImageSize != self.image_shape:
            calib_width = CalibrationImageSize[1]
            calib_height = CalibrationImageSize[0]
            width_factor = calib_width / self.frame_width
            height_factor = calib_height / self.frame_height
            self.tracks[:,2] = self.tracks[:,2] * width_factor
            self.tracks[:,3] = self.tracks[:,3] * height_factor

    def write_sequence(self,day_dir):
        output_dir = ntpath.join(day_dir,"analysis/2Dtracks").replace("\\","/")

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        tracks = self.tracks
        pandaPath = {'frame_idx':tracks[:,0],'ID':tracks[:,1],'x':tracks[:,2],'y':tracks[:,3]}
        pandaPath = pd.DataFrame(pandaPath)
        pandaPath.to_csv(ntpath.join(output_dir,self.sequence_name+'.csv').replace("\\","/"))
        self._generate_video(output_dir)
        
    def _generate_video(self,output_dir,imreadFlag=cv.IMREAD_REDUCED_COLOR_2):
        # create rainbow colors for IDs
        id_min = 1
        id_max = max(self.tracks[:,1])
        rbg = np.array([255,0,0])
        id_colors = []
        for ID in range(id_min,id_max+1):
            # manipulate rbg
            if (ID-1)>0 and (ID-1)%3 == 0:

                rbg = rbg//2
                for e,num in enumerate(rbg):

                    if num ==0:
                        rbg[e] = 255
                        continue
                    elif e ==2:
                        rbg[rbg.tolist().index(min(rbg))] = 255

            id_colors.append(tuple(map(int,rbg)))
            # shift
            rbg = np.array([rbg[1],rbg[2],rbg[0]])

        frame_list = []

        for frame_idx in range(self.min_frame_idx,self.max_frame_idx+1):
            frame = cv.imread(self.frame_paths[frame_idx],imreadFlag)

            text_dis_w = 30 #text distance for ID number written on frame in width
            text_dis_h = 15 #text distance for ID number written on frame in height

            for ID in range(id_min,id_max+1):
                # display ID colors on frame
                font = cv.FONT_HERSHEY_SIMPLEX
                cv.putText(frame,str(ID),(text_dis_w,text_dis_h), font, 0.5, id_colors[ID-1], 1, cv.LINE_AA)
                text_dis_w += 30
                if text_dis_w >= frame.shape[1] - 30:
                    text_dis_h += 15
                    text_dis_w = 30

                # get track with corresponding ID
                trackID = self.tracks[self.tracks[:,1]==ID]

                # draw ID lines on frame
                line_points = trackID[trackID[:,0]<=frame_idx]
                line_points = line_points[:,2:4]
                if len(line_points) > 1:
                    if imreadFlag == cv.IMREAD_REDUCED_COLOR_2:
                        line_points = line_points // 2
                    cv.polylines(frame, [line_points],isClosed=False,color=id_colors[ID-1], thickness=1)

            frame_list.append(frame)


        fourcc = cv.VideoWriter_fourcc(*"mp4v")
        #fourcc = cv.VideoWriter_fourcc(*'XVID')

        out = cv.VideoWriter(ntpath.join(output_dir,self.sequence_name+'.mp4').replace("\\","/"),
                             fourcc, 20, (frame.shape[1],frame.shape[0]))

        for frame in frame_list:
            out.write(frame)

        out.release()
        
class Sequence3D():
    '''
    stores information of 3D tracks of a given sequenceprint(traceback.format_exc())
    '''
    def __init__(self,seq_dir,cam1,cam2,match_matrix):
        
        frame_paths = sorted(glob.glob(seq_dir+"/*.jpg") )
        image_shape = cv.imread(frame_paths[0]).shape
        max_frame_idx = len(frame_paths)
        
        self.seq_dir = seq_dir # string with directory path
        
         
        sequence_name = os.path.basename(seq_dir) # basename of directory path
        self.sequence_name = sequence_name.replace(cam1,cam1+cam2)
        self.frame_paths = frame_paths # list of frame paths
        self.image_shape = image_shape # tuple (h,w,c)
        self.min_frame_idx = 0 # int
        self.max_frame_idx = len(frame_paths)-1 # int
        self.frame_width = image_shape[1]
        self.frame_height = image_shape[0]
        
        self.tracks = [] #np.array with [frame_idx, track_id, x,y,z]
        self.match_matrix = match_matrix
    def write_sequence(self,day_dir):
        output_dir = ntpath.join(day_dir,"analysis/3Dtracks").replace("\\","/")
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        tracks = self.tracks
        try:
            pandaPath = {'frame_idx':tracks[:,0].astype(int),'ID':tracks[:,1].astype(int),'x':tracks[:,2],'y':tracks[:,3],'z':tracks[:,4]}
            
            pandaPath = pd.DataFrame(pandaPath)
            pandaPath.to_csv(ntpath.join(output_dir,self.sequence_name+'.csv').replace("\\","/"))
        except Exception as e:
            info.error("unexpected error occured while writing 3D sequence "+ self.sequence_name)
            print(traceback.format_exc())

    
