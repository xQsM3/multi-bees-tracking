# Luc Stiemer 2020
# THIS SCRIPT LOADS DATA

#import librarys
import scipy.io as spio
import numpy as np
import sys
import cv2 as cv
import os
import ntpath
import glob
from os import path
import pandas as pd
#import scripts
from log import info
## HELP FUNCTIONS
def loadmat(filename):
    '''
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    '''
    data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)

def _check_keys(dict):
    '''
    checks if entries in dictionary are mat-objects. If yes
    todict is called to change them to nested dictionaries
    '''
    for key in dict:
        if isinstance(dict[key], spio.matlab.mio5_params.mat_struct):
            dict[key] = _todict(dict[key])
    return dict        

def _todict(matobj):
    '''
    A recursive function which constructs from matobjects nested dictionaries
    '''
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, spio.matlab.mio5_params.mat_struct):
            dict[strg] = _todict(elem)
        else:
            dict[strg] = elem
    return dict

## DEFINE FUNCTIONS

def checkData(dayDirectory):
    '''checkData(dayDirectory) -> dayDirectoryNew
.   takes in dayDirectory
.   returns new Directory
.   checks if Calibration and mask Coordinates exists
.   counts total number of videos
.   '''
    dayDirectoryNew = []
    
    #count total quantity of videos
    videoQuantity = 0
    
    for day in dayDirectory:
        boConDirectory = day+ "/analysis/boundaryConditions"
        if ( path.exists(boConDirectory+"/stereoCalib159161.mat") and 
            path.exists(boConDirectory+"/stereoCalib159160.mat") and 
            path.exists(boConDirectory+"/soloCalib159.mat") and 
            path.exists(boConDirectory+"/soloCalib160.mat") and 
            path.exists(boConDirectory+"/soloCalib161.mat") ):
            
              
            info.info('Calibration found for the day ' + ntpath.basename(day))
            dayDirectoryNew = np.append(dayDirectoryNew,day)
            
            videoDirectory = glob.glob(day +'/*')
            
            camera = 159
            while camera <= 161:
                if not path.exists(boConDirectory + "/maskCoordinates"+str(camera)+'.npy'):
                    i = 0
                    while True:
                        
                        if str(camera) == ntpath.basename(videoDirectory[i])[-5:-2]:
                            imgPath = glob.glob(videoDirectory[i]+'/*.jpg')[0]
                            img = cv.imread(imgPath)
                             
                            _ = defineMask(img, boConDirectory + "/maskCoordinates"+str(camera)+'.npy')
                            break
                        i += 1
                        if i >= len(videoDirectory):
                            info.warning('no image found to draw a mask for the day ' + ntpath.basename(day) +' camera {0}'.format(camera) )
                            break
                else:
                    info.info('mask found for the day ' + ntpath.basename(day))                
                camera += 1
        else:
            info.warning('no Calibration found for the day ' + ntpath.basename(day) + ' in directory ' + boConDirectory)
    
    #counts video total number
    for day in dayDirectoryNew:
        for _, dirnames, filenames in os.walk(str(day)):
            for dirname in dirnames:
                if dirname[0] == 'V':
                    videoQuantity += 1      
    
    return videoQuantity,dayDirectoryNew     


def defineMask(frame,boConDirectory):
    '''defineMask(frame) -> mask,maskCoordinates
.   takes in a frame and the saveDirectory for the Mask
.   returns a boolean mask, with a rectengular region of interest in white colour
.   returns maskCoordinates, coordinates of the region of interest
.   The function defineMask will open a frame to draw a rectangle on it. it's important to draw the 
.   rectangle starting from top left corner, going down to bottom right corner. Draw the rectangle wisely,
.   choose a rectangle which is including all the area of the tunnel where the bee is able to fly. not more
.   '''
    IX,IY,W,H = (0,0,0,0)
    
    #check if mask is allready saved, if yes, load it and skip the defining process
    exist = True
    try:
        maskCoordinates = np.load(boConDirectory)
    except IOError:
        exist = False
    if exist == True:
        (IX,IY,W,H) = maskCoordinates
        mask = np.zeros((frame.shape))
        mask[IY:IY+H,IX:IX+W] = 255
        return maskCoordinates
    info.info('''start "pulling" rectangle from TOP LEFT corner by press and hold 
        left mouse bottom pull towards BOTTOM RIGHT corner and release
        make sure that all important tunnel areas are covered
        the algorithm will ignore everything outside this rectangle''')
    #defines a function which will draw the rectangle on the image
    def drawRec(event,mx,my,flags,param):
        global drawing,ix,iy,w,h
        #start drawing if left mouse button is clicked
        if event == cv.EVENT_LBUTTONDOWN:
            drawing = True
            ix,iy = mx,my
        #continue drawing while mouse movement
        elif event == cv.EVENT_MOUSEMOVE:
            if 'drawing' in globals() and drawing:
                cv.rectangle(frame,(ix,iy),(mx,my),(255,0,0),-1)
        #stop drawing when left mouse button up, save the top left corner and bottom right corner of recangle
        elif event == cv.EVENT_LBUTTONUP:
            drawing = False
            w,h = mx-ix,my-iy
            cv.rectangle(frame,(ix,iy),(mx,my),(0,0,0),3)
            
    #define the window size
    shapeWindow(frame,'mask')
    #connects the window, the drawRec function, and the mouse
    cv.setMouseCallback('mask',drawRec)
    
    #shows the image until user hits escape
    while True:
        cv.imshow('mask',frame)

        
        if cv.waitKey(20) & 0xFF == 27:
            break

    cv.destroyAllWindows()
    
    #safe the recangle coordinates
    maskCoordinates = (ix,iy,w,h)
    np.save(boConDirectory,maskCoordinates)
    
    return maskCoordinates


def getFinalPathJPGDirectory(mainDirectory,logDic):
    '''etFinalPathJPGDirectory(mainDirectory,logDic) -> finalPathsJPGDirectory
.   combines video name and day name in logfile to the directory of 
.   the corresponding video
.   '''    
    finalPathsJPGDirectory = {"159": [],"160": [],"161": [] }
    cameras = '159','160','161'
    
    for day,video,checkDigit in zip(logDic['day'],logDic['video'],logDic['checkDigit']):
        if not checkDigit[0] == 'e':
            for cam in cameras:
                finalPathsJPGDirectory[cam].append(mainDirectory +'/' + str(day)+ '/analysis/paths 2D/jpg/' + video +'-'+cam+'_1.jpg' )
    return finalPathsJPGDirectory


def getVideoDirectory(day,reviseDirectory=0,flag='detect'):
    '''getVideoDirectory(day) -> videoDirectoryNew dictionary
.   takes in dayDirectory
.   returns new Directories for each camera
.   checks folders for 159 160 and 161 in name and sorts them by camera
.   flag "detect" is for tracker with auto detect motion
.   flag "select" is for tracker with manual select motion
.   '''
    if flag == 'detect':
        #get all folder paths in day folder
        videoDirectory = sorted(glob.glob(day +'/*'))
        videoDirectoryNew = {"159": [],"160": [],"161": [] }

        #sort out all folders without a "V" as first letter in name
        for folder in videoDirectory:
            if ntpath.basename(folder)[0] == 'V':
                camera = 159
                while camera <= 161:
                    if ntpath.basename(folder)[-5:-2] == str(camera):
                        videoDirectoryNew[str(camera)].append(folder)
                        break
                    else:
                        camera += 1
    if flag == 'select':
        #get base name of day
        dayBaseName = ntpath.basename(day)
        #extract the directories which belong to the current day
        videoDirectoryNew = {"159": [],"160": [],"161": [] }
        cameras = '159','160','161'
        for cam in cameras:
            for directory in reviseDirectory[cam]:
                if dayBaseName in directory:
                    videoDirectoryNew[cam].append(directory)          
    return videoDirectoryNew

def readCalib(day,cam1,cam2): 
    '''readCalib(day,cam1,cam2) -> K1,K2,D1,D2,R,T,F,E,imageSize
.   reads in calibration data from matlab 
.   follow these steps in matlab:
.   perform solo Calibrations in the Calibrator App
.   export Calibration Params into Matlab Workspace
.   type struct(NAME_OF_CALIBPARAMS)
.   an "ans" structure will appear in workspace, rename it to "soloCalib"
.   right klick on soloCalib, save as "soloCalib159" or 160, 161 in directory:
.   day/analysis/boundaryConditions
.   do the same for stereo Calibration but give it the name stereoCalib and save it as
.   stereoCalib159160 or stereoCalib159161
.   '''    
    boConDirectory = day+ "/analysis/boundaryConditions"
    #load the .mat files
    matlabStereoCalib = loadmat(boConDirectory+'/stereoCalib{0}{1}'.format(cam1,cam2))
    matlabSoloCalib1 = loadmat(boConDirectory+'/soloCalib{0}'.format(cam1))
    matlabSoloCalib2 = loadmat(boConDirectory+'/soloCalib{0}'.format(cam2))

    #extract the important matrices 
    R = matlabStereoCalib['stereoCalib']['RotationOfCamera2'].T
    T = matlabStereoCalib['stereoCalib']['TranslationOfCamera2'].reshape(3,1)
    F = matlabStereoCalib['stereoCalib']['FundamentalMatrix']
    F = F / F[2][2]
    E = matlabStereoCalib['stereoCalib']['EssentialMatrix']

    imageSize = tuple(matlabSoloCalib2['soloCalib']['ImageSize'])

    #the reference between camera matrix definition python / matlab is: matlabCameraMatrix^T = pythonCameraMatrix
    K1 =  np.transpose(matlabSoloCalib1['soloCalib']['IntrinsicMatrix']) 
    K2 =  np.transpose(matlabSoloCalib2['soloCalib']['IntrinsicMatrix'])
    K1 = K1.astype('float64') #important! there is a bug in opencv if K is unit16
    K2 = K2.astype('float64') #important! there is a bug in opencv if K is unit16
    
    #load the radial distortion parameters k1,k2,k3 and the tangential distortion parameters p1,p2
    k1_1 = matlabSoloCalib1['soloCalib']['RadialDistortion'][0]
    k2_1 = matlabSoloCalib1['soloCalib']['RadialDistortion'][1]
    try:
        k3_1 = matlabSoloCalib1['soloCalib']['RadialDistortion'][2]
    except:
        k3_1 = 0
    p1_1 = matlabSoloCalib1['soloCalib']['TangentialDistortion'][0]
    p2_1 = matlabSoloCalib1['soloCalib']['TangentialDistortion'][1]

    k1_2 = matlabSoloCalib2['soloCalib']['RadialDistortion'][0]
    k2_2 = matlabSoloCalib2['soloCalib']['RadialDistortion'][1]
    try:
        k3_2 = matlabSoloCalib2['soloCalib']['RadialDistortion'][2]
    except:
        k3_2 = 0
    p1_2 = matlabSoloCalib2['soloCalib']['TangentialDistortion'][0]
    p2_2 = matlabSoloCalib2['soloCalib']['TangentialDistortion'][1]

    #store the distortion parameters
    D1 = np.asarray([k1_1,k2_1,p1_1,p2_1,k3_1]).reshape(1,5).astype('float64')
    D2 = np.asarray([k1_2,k2_2,p1_2,p2_2,k3_2]).reshape(1,5).astype('float64')
    
    return K1,K2,D1,D2,R,T,F,E,imageSize


def readLogFile(directory,name):
    #reads log file from csv file into python dictionary
    logDic = pd.read_csv(directory+'/'+name, sep = ',',index_col=0,squeeze=True).to_dict()
    #change inner (pandas) dictionary to list
    for key in logDic.keys():
        dic = logDic[key]
        l = []
        for k in dic.keys():
            l.append(dic[k])
        logDic[key] = l   
    return logDic

def readMasks(day):
    #reads in the masks
    boConDirectory = day+ "/analysis/boundaryConditions"
    
    #load masks
    maskCoordinates159 = np.load(boConDirectory+"/maskCoordinates159.npy")
    maskCoordinates160 = np.load(boConDirectory+"/maskCoordinates160.npy")
    maskCoordinates161 = np.load(boConDirectory+"/maskCoordinates161.npy")
    return maskCoordinates159,maskCoordinates160,maskCoordinates161
def shapeWindow(frame,winname):
    '''shapeWindow(frame,winname)) -> nan
.   takes in a frame
.   takes in a winname
.   returns nan
.   The function shapeWindow is scaling the Window size since the default Window size is too big
.   to display it properly
.   '''
    shapeWindow = tuple(np.asarray(frame.shape[:2])//2)
    cv.namedWindow(winname, cv.WINDOW_NORMAL)
    cv.resizeWindow(winname,shapeWindow)