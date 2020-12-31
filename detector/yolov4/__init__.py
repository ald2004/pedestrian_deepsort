import sys

sys.path.append("detector/yolov4")

from .detector import yolov4
from . import darknet

__all__ = ['yolov4']
