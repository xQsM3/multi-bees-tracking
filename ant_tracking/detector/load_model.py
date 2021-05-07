from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2.engine import DefaultPredictor
from detector.predictor import Predictor
from detectron2.checkpoint import DetectionCheckpointer
import os
current = os.path.dirname(os.getcwd())

def load_model(conf_thresh,det_model='rcnn'):
    if det_model == 'rcnn':
        # load config
        cfg = get_cfg()
        cfg.merge_from_file(current+"/ant_tracking/detector/cfg/faster_rcnn_X_101_32x8d_FPN_3x.yaml")
        cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[32*0.75, 64*0.75, 128*0.75, 256*0.75, 512*0.75]]
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = conf_thresh # Set threshold for this model
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
        cfg.MODEL.WEIGHTS = current+"/ant_tracking/detector/models/bb_rcnn.pth"
        #load trained weights
        model = build_model(cfg) # returns a torch.nn.Module
        model.eval()
            
        DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
        model.train(False) # inference mode
        # create predictor
        predictor = Predictor(cfg)
        return model,predictor
    elif det_model == 'retina':
        # load config
        cfg = get_cfg()
        cfg.merge_from_file(current+"/ant_tracking/detector/cfg/retinanet_R_101_FPN_3x.yaml")
        cfg.MODEL.RETINANET.SCORE_THRESH_TEST = conf_thresh # Set threshold for this model 
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
        cfg.MODEL.WEIGHTS = current+"/ant_tracking/detector/models/bb_retina.pth"
        #load trained weights
        model = build_model(cfg) # returns a torch.nn.Module
        model.eval()
    
        DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
        model.train(False) # inference mode
        # create predictor
        predictor = Predictor(cfg)
        model = predictor.model
        return model,predictor
