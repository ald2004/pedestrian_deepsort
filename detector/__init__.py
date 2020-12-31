from .yolov4 import yolov4
from .yolov4 import darknet
import os

__all__ = ['build_detector', 'darknet', 'yolov4']


def build_detector(cfg, use_cuda):
    cfgfile = os.path.join(os.environ.get('DARKNET_PATH', './'), "cfg", "yolov4.cfg")
    weightsfile = os.path.join(os.environ.get('DARKNET_PATH', './'), "cfg", "coco.data")
    namesfile = os.path.join(os.environ.get('DARKNET_PATH', './'), "cfg", "yolov4.weights")
    return yolov4(cfgfile, weightsfile, namesfile,
                  # score_thresh=cfg.YOLOV3.SCORE_THRESH, nms_thresh=cfg.YOLOV3.NMS_THRESH,
                  is_xywh=True, use_cuda=use_cuda)
