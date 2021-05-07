import cv2 as cv
import sys
import os
import torch
import numpy as np
current = os.path.dirname(__file__)
parentparent = os.path.dirname(os.path.dirname(current))
sys.path.append(parentparent)

from bb_framework.tools import iou

class Detector():
    def __init__(self,model,predictor):
        self.predictor = predictor
        self.model = model
        self.outputs_gpu = []
    def predict_on_batch(self,batch):
            inputs = []
            for image in batch:
                bgr_image = cv.imread(image, cv.IMREAD_COLOR)
                image = np.transpose(bgr_image,(2,0,1))
                image_tensor = torch.from_numpy(image)
                inputs.append({"image":image_tensor})
                
            batch_outputs = self.predictor.batch_prediction(batch)
            self.outputs_gpu.extend(batch_outputs)
    def outputs_instances_to_cpu(self):
        outputs_cpu = []

        for output in self.outputs_gpu:
            output = output["instances"].to("cpu")
            boxes_pred = output.pred_boxes if output.has("pred_boxes") else None
            boxes_pred = boxes_pred.tensor.numpy()
            # transform boxes from x1,y1,x2,y2 to x1,y2,w,h coordinates
            for i,box in enumerate(boxes_pred):
                boxes_pred[i] = [box[0],box[1],box[2]-box[0],box[3]-box[1]]
                
            scores_pred = output.scores if output.has("scores") else None
            scores_pred = scores_pred.numpy()
            classes_pred = output.pred_classes if output.has("pred_classes") else None
            classes_pred = classes_pred.numpy()
            
            outputs_cpu.append({"pred_boxes":boxes_pred,"scores":scores_pred,"pred_classes":classes_pred})
        self.outputs_cpu = outputs_cpu
            
    def force_bbox_size(self):
        for frame_idx,output_cpu in enumerate(self.outputs_cpu):
            if frame_idx == 0:
                continue
            iou_matrix = iou.calculate_iou_matrix(self.outputs_cpu[frame_idx-1],output_cpu)
            if iou_matrix.size == 0:
                continue
            iou_max_matrix = iou.get_iou_max_matrix(iou_matrix)
            
            
            for j,row in enumerate(iou_max_matrix):
                for i,element in enumerate(row):
                    if element != None:
                        bbox1 = self.outputs_cpu[frame_idx-1]["pred_boxes"][j].copy()
                        bbox2 = output_cpu["pred_boxes"][i].copy()
                        # if bbox in next frame is more then 2 pixels wider or higher then in the last frame, force bbox size
                        if bbox1[2]-bbox2[2] > 1:
                            self.outputs_cpu[frame_idx]["pred_boxes"][i][2] = bbox1[2]-1
                            self.outputs_cpu[frame_idx]["pred_boxes"][i][0] +=0.5
                        if bbox1[2]-bbox2[2] < -1:
                            self.outputs_cpu[frame_idx]["pred_boxes"][i][2] = bbox1[2]+1
                            self.outputs_cpu[frame_idx]["pred_boxes"][i][0] -=0.5                          
                        if bbox1[3]-bbox2[3] > 1:
                            self.outputs_cpu[frame_idx]["pred_boxes"][i][3] = bbox1[3]-1
                            self.outputs_cpu[frame_idx]["pred_boxes"][i][1] +=0.5                           
                        if bbox1[3]-bbox2[3] < -1:
                            self.outputs_cpu[frame_idx]["pred_boxes"][i][3] = bbox1[3]+1
                            self.outputs_cpu[frame_idx]["pred_boxes"][i][1] -=0.5                        
