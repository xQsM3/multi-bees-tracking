#from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
import ntpath 
import sys
import plotly.express as px
import plotly.express as px
import pandas as pd
import os
'''
def save_3D_plt(day_dir,sequence3D_1,sequence3D_2):
    NOT WORKING WITH OPENCV......... THREAD ERROR
    output_dir = ntpath.join(day_dir,"analysis/3Dtracks").replace("\\","/")

    tracks1 = sequence3D_1.tracks
    tracks2 = sequence3D_2.tracks
    color1 = (255,0,0)
    color2 = (0,255,0)
    
    id_min1 = min(tracks1[:,1])
    id_max1 = max(tracks1[:,1])  
    id_min2 = min(tracks2[:,1])
    id_max2 = max(tracks2[:,1])  

    fig = plt.figure()
    
    fig = plt.figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
    ax = plt.axes(projection='3d')
    
    for frame_idx in range(sequence3D_1.min_frame_idx,sequence3D_1.max_frame_idx+1):
        for ID in range(id_min1,id_max1+1):
            trackID1 = tracks1[tracks1[:,1]==ID]
            zline = trackID1[:,4]
            yline = trackID1[:,3]
            xline = trackID1[:,2]
            ax.plot3D(xline, yline, zline, color1)
    
    for frame_idx in range(sequence3D_2.min_frame_idx,sequence3D_2.max_frame_idx+1):
        for ID in range(id_min2,id_max2+1):
            trackID2 = tracks2[tracks2[:,1]==ID]
            zline = trackID2[:,4]
            yline = trackID2[:,3]
            xline = trackID2[:,2]
            ax.plot3D(xline, yline, zline, color2)
    plt.savefig(ntpath.join(output_dir,sequence3D_1.sequence_name[:-5]+'.png'))
'''
def save_3D_plt(day_dir,sequence3D_1,sequence3D_2):
    output_dir = ntpath.join(day_dir,"analysis/3Dtracks").replace("\\","/")
    
    try:
        tracks1 = sequence3D_1.tracks
        tracks2 = sequence3D_2.tracks

        id_min1 = int(min(tracks1[:,1]))
        id_max1 = int(max(tracks1[:,1]))
        id_min2 = int(min(tracks2[:,1]))
        id_max2 = int(max(tracks2[:,1])) 

        d = {'x':[],'y':[],'z':[],'camerapair':[],'ID':[],'size':[]}
        #for frame_idx in range(sequence3D_1.min_frame_idx,sequence3D_1.max_frame_idx+1):
        for ID in range(id_min1,id_max1+1):
            trackID1 = tracks1[tracks1[:,1]==ID]
            if len(trackID1) == 0:
                continue
            color1 = []
            #color1 = np.array(color1)

            d['x'].extend(trackID1[:,2].tolist())
            d['y'].extend(trackID1[:,3].tolist())
            d['z'].extend(trackID1[:,4].tolist())
            d['camerapair'].extend([1]*len(trackID1))
            d['ID'].extend([ID]*len(trackID1))
            #d['species'].extend(["virginica"]*len(trackID1))
            d['size'].extend([2]*len(trackID1))

        #for frame_idx in range(sequence3D_2.min_frame_idx,sequence3D_2.max_frame_idx+1):
        for ID in range(id_min2,id_max2+1):
            trackID2 = tracks2[tracks2[:,1]==ID]
            if len(trackID2) == 0:
                continue
            color2 = []
            for i in range(0,len(trackID2)):
                color2.append((0,255,0))
            #color2 = np.array(color2)
            d['x'].extend(trackID2[:,2].tolist())
            d['y'].extend(trackID2[:,3].tolist())
            d['z'].extend(trackID2[:,4].tolist())
            d['camerapair'].extend([2]*len(trackID2))
            d['ID'].extend([ID]*len(trackID2))
            #d['species'].extend(["setosa"]*len(trackID2))
            d['size'].extend([2]*len(trackID2))
        df = pd.DataFrame(data=d)
        df["ID"] = df["ID"].astype(str)
        #fig = px.line_3d(df, x="x", y="y", z="z", color='camerapair')
        fig = px.scatter_3d(df, x='x', y='y', z='z',
                  color='ID', size='size', size_max=8,
                  symbol='camerapair', opacity=0.7)

        fig.update_layout(
            scene = {"aspectmode":"manual","aspectratio":dict(x=1, y=8/3, z=1),
                    "xaxis":dict(nticks=4, range=[30,-270],),
                    "yaxis":dict(nticks=4, range=[-400,400],),
                    "zaxis":dict(nticks=4, range=[1960,1660],)})

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)    
        #fig.update_layout(scene1_aspectmode='auto')
        fig.show()
        #fig.write_image(ntpath.join(output_dir,sequence3D_1.sequence_name[:-5]+'.png').replace("\\","/"))
        fig.write_html(ntpath.join(output_dir,sequence3D_1.sequence_name[:-6]+'.html').replace("\\","/"))
    except:
        pass
