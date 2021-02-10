# Luc Stiemer 2020
# THIS SCRIPT WRITES A GLOBAL LOG FILE TO INFORM USER ABOUT PATH QUALITIES

#import python modules
import ntpath
import os
import pandas as pd
from csv import writer
from csv import reader
#import own modules

class Logger:
    def __init__(self):
        self.logDic = {'day':[],'video':[],'checkDigit':[],'evaInfo':[]}
    
    
    def _addEvaInfoToLogfile(self,evaInfo):
        self.logDic['evaInfo'] = evaInfo    
    
    def _addVideoToLogfile(self,videoDirectory,checkDigit):
        
        #get day from directory
        dayDirectory = os.path.dirname(videoDirectory)
        day = ntpath.basename(dayDirectory)
        
        #get video from directory
        video = ntpath.basename(videoDirectory[:-6])
        
        #append log information to class object
        self.logDic['day'].append(day)
        self.logDic['video'].append(video)
        self.logDic['checkDigit'].append(checkDigit)
        self.logDic['evaInfo'].append('-')
        
    def _changeRowInLogFile(self,videoDirectory,checkDigit,evaInfo='keep'):
        #get video base name
        video = ntpath.basename(videoDirectory[:-6])
        #get day namey
        dayDirectory = os.path.dirname(videoDirectory)
        day = ntpath.basename(dayDirectory)
        #change the check digit and evaInfo in log file
        for i,d in enumerate(self.logDic['day']):
            if d == day and self.logDic['video'][i] == video:
                self.logDic['checkDigit'][i] = checkDigit
                if not evaInfo == 'keep': 
                    self.logDic['evaInfo'][i] = evaInfo
                break   
                
    def _createLogTable(self):
        logTable = pd.DataFrame(self.logDic)
        return logTable
    
    def _writeLogReadMe(self,directory):
        
        readMe = open(directory+"/logReadMe.txt","w+")
        readMe.write("#################################################################\n")
        readMe.write("                      check digit explanations                   \n")
        readMe.write("#################################################################\n")
        
        readMe.write("aX:\n")
        readMe.write("      a means 2D and 3D paths' qualities satisfy all requirements\n")
        readMe.write("              the X gives percentage of 3D paths coverage between the stereo pairs\n")
        readMe.write("              the higher the percentage the more significant is the comparison between\n")
        readMe.write("              the stereo pairs\n")
        
        readMe.write("bXY:\n")
        readMe.write("      b means at least one of the 2D paths is not satisfying requirements\n")
        readMe.write("              X=1 means checkYsimilarity2D in the lib/path.py script is not satisfied\n")
        readMe.write("                      Y can be either 12 or 13 for the corresponding camera pair\n")
        
        readMe.write("cXY:\n")
        readMe.write("      c means 3D paths are not satisfying requirements\n")
        readMe.write("              X=1 means checkSimilarity3D in the lib/path.py script is not satisfied\n")
        readMe.write("              X=2 means checkVmax         in the lib/path.py script is not satisfied\n")
        readMe.write("                      Y can be either 12 or 13 for the corresponding camera pair\n")
        
        readMe.write("eX:\n")
        readMe.write("      e means error in analysis\n")
        readMe.write("              X=1 means 2D paths have different lengths\n")     
        readMe.write("              X=2 means evaluateFinalPathsJPG in path.py could not jpg or csv path files\n") 
        
        
        readMe.write("#################################################################\n")
        readMe.close()
        
    
    def _loadExistingLogFileToLogger(self,existingLogDic):
        #loads allready existing log files into the class object
        self.logDic['day']= existingLogDic['day']
        self.logDic['video'] = existingLogDic['video']
        self.logDic['checkDigit'] = existingLogDic['checkDigit']
        self.logDic['evaInfo'] = existingLogDic['evaInfo']
                
    def writeLogFile(self,directory,name):
            #write togfile to csv
            logTable = self._createLogTable()
            logTable.to_csv(directory+'/'+name)
            self._writeLogReadMe(directory)
