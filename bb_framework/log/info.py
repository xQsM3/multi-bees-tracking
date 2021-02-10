# Luc Stiemer 2020
# THIS SCRIPT GIVES ERRORS / WARNING / INFORMATION IN TERMINAL

#import python modules
from sty import fg, bg, ef, rs, RgbFg
import sys
import numpy as np
import time

#global values
progressBarStarted = False
printCounter = 0

## DEFINE USER MESSAGE FUNCTIONS

#print error messages in colour red
def error(message):
    global progressBarStarted,printCounter
    
    print(fg.red + "[ERROR]: "+message+ "\n"+fg.rs)
    print("        0:", sys.exc_info()[0])
    print("        1:", sys.exc_info()[1])
    print("        2:", sys.exc_info()[2])
    
    if progressBarStarted:
        printCounter +=5 
        
#print info messages in colour white
def info(message):
    global progressBarStarted,printCounter
    print("[INFO]: "+message+"\n")
    if progressBarStarted:
        printCounter +=1  
    
# The MIT License (MIT)
# Copyright (c) 2016 Vladimir Ignatev
#
# Permission is hereby granted, free of charge, to any person obtaining 
# a copy of this software and associated documentation files (the "Software"), 
# to deal in the Software without restriction, including without limitation 
# the rights to use, copy, modify, merge, publish, distribute, sublicense, 
# and/or sell copies of the Software, and to permit persons to whom the Software 
# is furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included 
# in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR 
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
# FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT
# OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE 
# OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
def progress(count, total, start_time,status='analyis progress: '):
    global progressBarStarted,printCounter
    
    elapsed_time = time.time() - start_time
    #go as many carriages back as there were prints between the last bar and the current
    carriageReturn = ''
    for i in np.arange(0,printCounter):
        carriageReturn += '\r'
        
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)
    
    
    print('[%s] %s%s %s%s' % (bar, status,percents, '%',carriageReturn))
    print(time.strftime("                                                               elapsed time    : %H:%M:%S",
                        time.gmtime(elapsed_time)),flush=True)
    
    #sys.stdout.write('[%s] %s%s ...%s%s' % (bar, percents, '%', status,carriageReturn))
    
    print('\n')
    sys.stdout.flush()  # As suggested by Rom Ruben (see: http://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console/27871113#comment50529068_27871113)
    
    if progressBarStarted == False:
        progressBarStarted = True

#start time for progress bar
def startTime():
    return time.time() 

#print warning messages in colour yellow
def warning(message):
    
    global progressBarStarted,printCounter
    
    print (fg.yellow+"[WARNING]: "+message+ "\n"+fg.rs)

    if progressBarStarted:
        printCounter +=1
        
#welcome message    
def welcome():
    #https://manytools.org/hacker-tools/ascii-banner/ Calvin S Normal Normal
    print('\n\n\n┌┐ ┬ ┬┌┬┐┌┐ ┬  ┌─┐┌┐ ┌─┐┌─┐  ┌┬┐┬─┐┌─┐┌─┐┬┌─┌─┐┬─┐\n├┴┐│ ││││├┴┐│  ├┤ ├┴┐├┤ ├┤    │ ├┬┘├─┤│  ├┴┐├┤ ├┬┘\n└─┘└─┘┴ ┴└─┘┴─┘└─┘└─┘└─┘└─┘   ┴ ┴└─┴ ┴└─┘┴ ┴└─┘┴└─\n\n\n')