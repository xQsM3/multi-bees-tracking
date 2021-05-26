from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detector.predictor import DetectronPredictor
from detector.predictor import ExternPredictor
from detectron2.checkpoint import DetectionCheckpointer
import os
import yaml
scriptpath = os.path.dirname(os.getcwd())
# class for models which are not part of detectron model zoo
class Config():
    def __init__(self):
        pass
    def merge_from_file(self,path):
        with open(path,mode="r") as y:
            ydic = yaml.load(y)
            self.MODEL_WEIGHTS = ydic["MODEL"]["WEIGHTS"]
            self.MODEL_NC = ydic["MODEL"]["NC"]
            self.MODEL_CLASSNAMES = ydic["MODEL"]["CLASSNAMES"]
            self.MODEL_ARCHITECTURE = ydic["MODEL"]["ARCHITECTURE"]
def load_model(conf_thresh,det_model):
    if det_model == 'rcnn':
        # load config
        cfg = get_cfg()
        cfg.merge_from_file(scriptpath + "/ant_tracking/detector/cfg/faster_rcnn_X_101_32x8d_FPN_3x.yaml")
        cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[32 * 0.75, 64 * 0.75, 128 * 0.75, 256 * 0.75, 512 * 0.75]]
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = conf_thresh  # Set threshold for this model
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
        cfg.MODEL.WEIGHTS = scriptpath+"/ant_tracking/detector/"+cfg.MODEL.WEIGHTS[1::]
        cfg.MODEL.DEVICE = "cuda"
        # load trained weights
        model = build_model(cfg)  # returns a torch.nn.Module
        model.eval()

        DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
        model.train(False)  # inference mode
        # create predictor
        predictor = DetectronPredictor(cfg)
        return model, predictor

    # load retina from detectron model zoo
    elif det_model == 'retina':
        # load config
        cfg = get_cfg()
        cfg.merge_from_file(scriptpath+"/ant_tracking/detector/cfg/retinanet_R_101_FPN_3x.yaml")
        cfg.MODEL.RETINANET.SCORE_THRESH_TEST = conf_thresh # Set threshold for this model
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
        cfg.MODEL.WEIGHTS = scriptpath+"/ant_tracking/detector/"+cfg.MODEL.WEIGHTS[1::]
        cfg.MODEL.DEVICE = "cuda"
        #load trained weights
        model = build_model(cfg) # returns a torch.nn.Module
        model.eval()
    
        DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
        model.train(False) # inference mode
        # create predictor
        predictor = DetectronPredictor(cfg)
        model = predictor.model
        return model,predictor

    # load yolo (not in detectron modelzoo)
    elif det_model == 'yolov5l':
        cfg = Config()
        cfg.merge_from_file(scriptpath+"/ant_tracking/detector/cfg/yolov5_l.yaml")
        cfg.MODEL_YOLO_CONF_THRESH = conf_thresh
        cfg.MODEL_YOLO_IOU_THRESH = 0.45 # default iou thresh for nms
        cfg.MODEL_DEVICE = "cuda"
        cfg.MODEL_WEIGHTS = scriptpath+"/ant_tracking/detector/"+cfg.MODEL_WEIGHTS[1::]
        # create predictor
        predictor = ExternPredictor(cfg)
        model = predictor.model
        return model,predictor
