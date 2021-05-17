# Luc Stiemer 2020
# THIS SCRIPTS MATCHES TRACKS BETWEEN 2D AND 3D TRACKS

#import python modules
import numpy as np
import math
import os
import pathlib
import cv2 as cv
import ntpath
import sys
import pandas as pd
from scipy.stats import iqr


#check for similarity between each position of the stereo pairs
def match3D(path12,path13):

    #subtract paths
    diffs = np.subtract(path12,path13)

    #check if difference is bigger then five millimeter
    similar = True
    count = 0
    for diff in diffs:
        if abs(diff).any() > 5 and not np.isnan(diff).any():
            similar = False
        else:
            if not np.isnan(diff).any():
                count += 1
    #calculate similarity in percent
    percent = count // len(path12) * 100
    return similar,percent
'''
def match2D(sequence1,sequence2,stereo):
    this solution is not working dude to bug in cv.stereorectify, it would be faster then the suggested solution
    so you might check for bug fix in later opencv versions then 4.4

    max_track_id_1 = max(sequence1.tracks[:,1])
    min_track_id_1 = min(sequence1.tracks[:,1])
    max_track_id_2 = max(sequence2.tracks[:,1])
    min_track_id_2 = min(sequence2.tracks[:,1])
    
    # init empty match matrix
    match_matrix = np.zeros((max_track_id_1+1,max_track_id_2+1))
    thresh = 3
    match_matrix[:] = thresh +1
    #perform rectification
    R1, R2, P1, P2, _, _, _ = stereo._perform_rectification()
    #check if vertical or horizontal stereo pair
    vertical_stereo = stereo._check_for_vertical_stereo(P2)
    
    for id1 in range(min_track_id_1,max_track_id_1+1):
        for id2 in range(min_track_id_2,max_track_id_2+1):
            track1 = sequence1.tracks[sequence1.tracks[:,1]==id1]
            track2 = sequence2.tracks[sequence2.tracks[:,1]==id2]
            # remap the tracks with rectification
            track1,track2=stereo._rectify_tracks(track1,track2,R1,R2,P1,P2)
            frame_indices_1 = track1[:,0]
            frame_indices_2 = track2[:,0]

            # calculate diffs
            diffs = []
            for idx1 in frame_indices_1:
                if len(track2[track2[:,0]==idx1]) == 0:
                    continue
                #for vertical stereo, bee should have same x value in both rectified camera images
                # (since epilines are vertical)
                if vertical_stereo:
                    diff = abs(track1[track1[:,0]==idx1][0,2]-track2[track2[:,0]==idx1][0,2])
                #for horizontal stereo, bee should have same y value
                # (since epilines are horizontal)
                else:
                    diff = abs(track1[track1[:,0]==idx1][0,3]-track2[track2[:,0]==idx1][0,3])
                diffs.append(diff)


            #remove outliers with IQR algorithm
            diffsc = diffs.copy()
            if diffs:
                diffs = np.array(diffs)
                IQR = iqr(diffs)
                Q3 = np.percentile(diffs, 75, interpolation = 'midpoint')
                Q = Q3 + IQR*1.5
                diffs = diffs[diffs<Q]
                if diffs.size > 0:
                    mean_dissmatch = np.mean(diffs)
                    match_matrix[id1,id2] = mean_dissmatch
                else:
                    print(diffsc)
                    print(diffs_iqr)
                    print(diffs)
    
    # convert match matrix to boolean matrix, mean_dissmatch should be smaller then given threshold, to count as True match
    print(match_matrix)
    match_matrix = match_matrix < thresh
    print(match_matrix)
    return match_matrix
'''
def match2D(sequence1,sequence2,stereo):
    '''
    mapping the points in the rectified plane and then compute the horizontal distance between camera1 point and camera2 point
    would be faster then following solution, but rectifying is buggy in opencv 4.4. therefore, here epilines of camera2 points are
    calculated, and the distances between epilines and camera1 points are computed to see if path1 and path2 match
    
    #returns a boolean matrix, True means track of ID1x matchs with ID2x
    '''
    max_track_id_1 = max(sequence1.tracks[:,1])
    min_track_id_1 = min(sequence1.tracks[:,1])
    max_track_id_2 = max(sequence2.tracks[:,1])
    min_track_id_2 = min(sequence2.tracks[:,1])
    
    # init empty match matrix
    match_matrix = np.zeros((max_track_id_1+1,max_track_id_2+1))
    thresh = 13
    match_matrix[:] = thresh +1

    #resize tracks to calibration image size
    tracks1 = sequence1.tracks_to_calibSize()
    tracks2 =sequence2.tracks_to_calibSize()
    for id1 in range(min_track_id_1,max_track_id_1+1):
        for id2 in range(min_track_id_2,max_track_id_2+1):
            track1 = tracks1[tracks1[:,1]==id1]
            track2 = tracks2[tracks2[:,1]==id2]

            frame_indices_1 = track1[:,0]
            frame_indices_2 = track2[:,0]

            # calculate diffs
            distances = []
            for idx1 in frame_indices_1:
                if len(track2[track2[:,0]==idx1]) == 0:
                    continue
                pt2 = track2[track2[:,0]==idx1][0,2:4]
                epiline1 = stereo.get_epipolar_lines(pt2)
                
                # compute epiline in y=mx+b format
                m_1 = -epiline1[0,1]/epiline1[0,0]
                b_1 = - epiline1[0,2]/epiline1[0,0]
                # compute ortogonal to epiline going through track point x_1,y_1
                x_1 = track1[track1[:,0]==idx1][0,3]
                y_1 = track1[track1[:,0]==idx1][0,2]
                m_ort = -1/m_1
                b_ort = y_1 -m_ort * x_1
                # compute intersection
                x_is = (b_1-b_ort) / (m_ort-m_1)
                y_is = (m_1*(b_1-b_ort)/(m_ort-m_1))+b_1
                # compute distance between intersection point and point1
                d_is1 = np.sqrt((x_1-x_is)**2+(y_1-y_is)**2)
                #for vertical stereo, bee should have same x value in both rectified camera images
                # (since epilines are vertical)
                distances.append(d_is1)

            #remove outliers with IQR algorithm
            distancesc = distances.copy()

            if distances:
                distances = np.array(distances)
                IQR = iqr(distances)
                Q3 = np.percentile(distances, 75, interpolation = 'midpoint')
                Q = Q3 + IQR*0.5
                distances = distances[distances<Q]
                if distances.size > 0:
                    mean_dissmatch = np.mean(distances)
                    match_matrix[id1,id2] = mean_dissmatch
    # convert match matrix to boolean matrix, mean_dissmatch should be smaller then given threshold, to count as True match
    match_matrix = match_matrix < thresh
    return match_matrix