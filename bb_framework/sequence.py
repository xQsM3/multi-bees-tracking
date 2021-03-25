import glob
import cv2 as cv
import os
import pandas as pd
import ntpath
import numpy as np

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
        self.min_frame_idx = 0 # int
        self.max_frame_idx = len(frame_paths)-1 # int
        self.detections = [] 
        self.appearances = [] #Matrix of detections+appearance descriptor. The first 10 columns of the detection matrix are the detections.
        self.tracks = [] #np.array with [frame_idx, track_id, x,y]
    
    def write_sequence(self,day_dir):
        output_dir = ntpath.join(day_dir,"analysis/2Dtracks").replace("\\","/")
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        tracks = self.tracks
        pandaPath = {'frame_idx':tracks[:,0],'ID':tracks[:,1],'x':tracks[:,2],'y':tracks[:,3]}
        pandaPath = pd.DataFrame(pandaPath)
        pandaPath.to_csv(ntpath.join(output_dir,self.sequence_name+'.csv').replace("\\","/")) 
        self._generate_video(output_dir)
        
    def _generate_video(self,output_dir):
        # create rainbow colors for IDs
        id_min = min(self.tracks[:,1])
        id_max = max(self.tracks[:,1])
        
        rbg = np.array([255,0,0])
        id_colors = []
        for i,ID in enumerate(range(id_min,id_max+1)):
            # manipulate rbg
            if i>0 and i%3 == 0:
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
            
            frame = cv.imread(self.frame_paths[frame_idx])
            for ID in range(id_min,id_max+1):
                
                # display ID colors on frame
                font = cv.FONT_HERSHEY_SIMPLEX
                
                cv.putText(frame,str(ID),(50+ID*50,50), font, 2, id_colors[ID-1], 2, cv.LINE_AA)
                
                trackID = self.tracks[self.tracks[:,1]==ID]
                
                for idx in range(self.min_frame_idx,frame_idx):
                    # check if there is an entry for track with ID at frame frame_idx
                    if not len(trackID[trackID[:,0]==idx]) == 0 \
                            and not len(trackID[trackID[:,0]==idx-1]) == 0:
                        # get points of track at frame_idx
                        pt1 = trackID[trackID[:,0]==idx-1][0,2],trackID[trackID[:,0]==idx-1][0,3]
                        pt2 = trackID[trackID[:,0]==idx][0,2],trackID[trackID[:,0]==idx][0,3]
                        cv.line(frame,pt1,pt2,id_colors[ID-1],2)
                                  
            frame_list.append(frame)         
        
        fourcc = cv.VideoWriter_fourcc(*"mp4v")
        #fourcc = cv.VideoWriter_fourcc(*'XVID')
        out = cv.VideoWriter(ntpath.join(output_dir,self.sequence_name+'.mp4').replace("\\","/"),
                             fourcc, 20, (self.image_shape[1],self.image_shape[0]))       
        for frame in frame_list:
            out.write(frame)
        out.release()
        
class Sequence3D():
    '''
    stores information of 3D tracks of a given sequence
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
        
        
        self.tracks = [] #np.array with [frame_idx, track_id, x,y,z]
        self.match_matrix = match_matrix
    def write_sequence(self,day_dir):
        output_dir = ntpath.join(day_dir,"analysis/3Dtracks").replace("\\","/")
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        tracks = self.tracks
        try:
            pandaPath = {'frame_idx':list(map(tracks[:,0],int)),'ID':list(map(tracks[:,1],int)),'x':tracks[:,2],'y':tracks[:,3],'z':tracks[:,4]}
            pandaPath = pd.DataFrame(pandaPath)
            pandaPath.to_csv(ntpath.join(output_dir,self.sequence_name+'.csv').replace("\\","/"))
        except:
            pass

    
