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
    def predict_on_batch(self,batch,imdim):
            inputs = []
            for image in batch:
                bgr_image = cv.imread(image, cv.IMREAD_COLOR)
                image = np.transpose(bgr_image,(2,0,1))
                image_tensor = torch.from_numpy(image)
                inputs.append({"image":image_tensor})
                
            batch_outputs = self.predictor.batch_prediction(batch,imdim)
            self.outputs_gpu.extend(batch_outputs)

    def outputs_instances_to_cpu(self):
        outputs_cpu = []
        for output in self.outputs_gpu:
            # check if output comes from a model out of detectron model zoo (cause then the output is a dictionary)
            if type(output) is not dict: #model is NOT from detectron
                # pass output to cpu
                output = output.to("cpu")

                # transform tensors to numpy arrays
                boxes_pred = output[:,:4].numpy() if len(output[:,:4].size()) != 0 else None
                # transform boxes from x1,y1,x2,y2 to x1,y2,w,h coordinates
                for i, box in enumerate(boxes_pred):
                    boxes_pred[i] = [box[0], box[1], box[2] - box[0], box[3] - box[1]]

                scores_pred = output[:,4] if len(output[:,4].size()) != 0 else None
                scores_pred = scores_pred.numpy()
                classes_pred = output[:,5] if len(output[:,5].size()) != 0 else None
                classes_pred = classes_pred.numpy()

            else: # model is in detectron model zoo
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

