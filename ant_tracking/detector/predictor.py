from PIL import Image
from detectron2.engine import DefaultPredictor
from detectron2.structures.boxes import Boxes
import cv2 as cv
import torch
import numpy as np
import time
import torchvision


class DetectronPredictor(DefaultPredictor):
    """
    modified detectron predictor, works for all models from the detectron2 model zoo. if you have an extern mode
    which is not in the model zoo, use the Predictor class instead
    """

    def __init__(self, cfg):
        super(DetectronPredictor,self).__init__(cfg)

    def batch_prediction(self,batch,imdim):
        """
        Args:
            batch list : str list with paths of batch frames

        Returns:
            predictions [dics]:
                the output of the model for the batch. each dic is for one frame
                See :doc:`/tutorials/models` for details about the format.
        """

        with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
            inputs = []
            images = []

            for im in batch:
                #read image
                im = cv.imread(im,cv.IMREAD_COLOR)
                # resize image
                imshape0 = im.shape
                im = self.resize_image(im,imdim)
                #im = np.asarray(Image.open(im))
                if self.input_format == "RGB":
                    # whether the model expects BGR inputs or RGB
                    im = im[:, :, ::-1]
                width,height = im.shape[:2]

                #transform image to tensor
                image = self.aug.get_transform(im).apply_image(im)
                images.append(image.astype("float32").transpose(2, 0, 1))

            #prepare input for model
            images = torch.as_tensor(np.array(images),device='cpu')
            for image in images:
                inputs.append({"image": image, "height": height, "width": width})

            #predict on batch
            predictions = self.model(inputs)
            # reshape bboxes from imdim to original image dimensions
            for batchIDX,pred in enumerate(predictions):
                predictions[batchIDX]["instances"].set("pred_boxes",
                                                       self.reshape_boxes(pred["instances"].pred_boxes.tensor[:, :4],
                                                                          imdim,imshape0) )
            return predictions

    def resize_image(self,im,imdim):
        # resize image, where imdim is the max dimendsion of resized im
        if max(im.shape) == imdim:
            return im

        if im.shape[1] > im.shape[0]:
            im = cv.resize(im, (round(im.shape[0] / im.shape[1] * imdim),imdim),
                           interpolation=cv.INTER_AREA)
        else:
            im = cv.resize(im, (imdim,round(im.shape[1] / im.shape[0] * imdim)),
                           interpolation=cv.INTER_AREA)
        return im

    def reshape_boxes(self,boxes,imdim,imshape0):
        if max(imshape0) == imdim:
            return Boxes(boxes)
        #reshape boxes from imdim to imshape0
        height0,width0 = imshape0[:2]
        if imshape0[0] > imshape0[1]:
            boxes *= (height0/imdim)
        else:
            boxes *= (weight0 / imdim)
        return Boxes(boxes)
class ExternPredictor():
    """
    predictor class for all detectors which are pytorch based, but NOT implemented in the detectron modelzoo if
    you want to add a model from the detectron model zoo use DetectronPredictor class
    """
    def __init__(self,cfg):
        # build model
        self.device = torch.device('cuda:0')
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=cfg.MODEL_WEIGHTS).to(self.device)
        self.model.eval()
        self.cfg = cfg
    def batch_prediction(self,batch,imdim):

        with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
            images = []

            for im in batch:
                # read image
                image = Image.open(im)
                images.append(image)

            # predict on batch
            predictions = self.model(images,imdim).pred # includes NMS
            return predictions




    def box_iou(self,box1, box2):
        # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
        """
        Return intersection-over-union (Jaccard index) of boxes.
        Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
        Arguments:
            box1 (Tensor[N, 4])
            box2 (Tensor[M, 4])
        Returns:
            iou (Tensor[N, M]): the NxM matrix containing the pairwise
                IoU values for every element in boxes1 and boxes2
        """

        def box_area(box):
            # box = 4xn
            return (box[2] - box[0]) * (box[3] - box[1])

        area1 = box_area(box1.t())
        area2 = box_area(box2.t())

        # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
        inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
        return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)

