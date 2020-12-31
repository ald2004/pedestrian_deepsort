import sys

sys.path.append('.')
sys.path.append('../..')
import darknet
from utils.log import get_logger
import cv2
import time
import torch


class yolov4(object):
    def __init__(self, cfgfile, weightfile, datafile, score_thresh=0.7, conf_thresh=0.01, nms_thresh=0.45,
                 is_xywh=False, use_cuda=True):
        # net definition
        self.net, self.class_names, self.class_colors = darknet.load_network(
            cfgfile,
            weightfile,
            datafile,
            batch_size=1
        )
        # self.net = Darknet(cfgfile)
        # self.net.load_weights(weightfile)
        logger = get_logger("root.detector")
        logger.info('Loading weights from %s... Done!' % (weightfile))
        self.device = "cuda" if use_cuda else "cpu"
        if use_cuda:
            darknet.set_gpu(0)
        # self.net.eval()
        # self.net.to(self.device)

        # constants
        self.width = darknet.network_width(self.net)
        self.height = darknet.network_height(self.net)
        logger.debug('yolov4 width is %s height is  %s ... Done!' % (self.width,self.height))
        self.size = self.width, self.height
        self.score_thresh = score_thresh
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.use_cuda = use_cuda
        self.is_xywh = is_xywh
        # self.num_classes = self.net.num_classes
        self.num_classes = len(self.class_names)
        # self.class_names = self.load_class_names(namesfile)
        self.darknet_image = darknet.make_image(self.width, self.height, 3)

    def __call__(self, ori_img):
        frame_rgb = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        darknet.copy_image_from_bytes(self.darknet_image, frame_resized.tobytes())

        # prev_time = time.time()
        detections = darknet.detect_image(self.net, self.class_names, self.darknet_image, thresh=self.score_thresh)
        # detections_queue.put(detections)
        # fps = int(1 / (time.time() - prev_time))
        # fps_queue.put(fps)
        # print("FPS: {}".format(fps))  # FPS: 20
        # print(detections)
        '''
        [('person', '51.17', (256.9013366699219, 147.636962890625, 33.269134521484375, 119.59164428710938)), 
         ('person', '51.23', (515.8123168945312, 141.8507537841797, 16.541969299316406, 44.08515548706055)), 
         ('person', '70.73', (205.21104431152344, 101.26861572265625, 29.76723861694336, 40.8036994934082)), 
         ('person', '82.67', (285.16668701171875, 146.8859405517578, 28.292022705078125, 126.650390625)), 
         ('person', '84.11', (436.45367431640625, 138.91932678222656, 17.87806510925293, 120.42742919921875)), 
         ('person', '86.5', (429.0495300292969, 429.708740234375, 71.49530029296875, 212.34127807617188)), 
         ('person', '89.65', (404.0263366699219, 136.01242065429688, 23.12635612487793, 123.673583984375)), 
         ('person', '97.56', (310.4417419433594, 139.56553649902344, 24.66164779663086, 126.3392333984375))]
        '''

        # if len(detections) == 0:
        # bbox = torch.FloatTensor([]).reshape([0, 4])
        # cls_conf = torch.FloatTensor([])
        # cls_ids = torch.LongTensor([])
        # else:
        height, width = ori_img.shape[:2]
        # bbox = boxes[:, :4]
        bbox = torch.FloatTensor([x[2] for x in detections]).reshape(-1,4)
        # if self.is_xywh:
        #     # bbox x y w h
        #     bbox = xyxy_to_xywh(bbox)
        # print('444444444444444')
        # print(bbox)
        '''
            tensor([[462.4090, 313.6196,  45.8091,  89.3959],
                    [266.6826, 547.0832,  80.4798, 119.5234],
                    [119.7864, 323.3072,  61.8375, 157.7023],
                    [ 42.7051, 283.3626,  51.8350, 103.8230],
                    [193.3966, 434.6747,  52.6424, 278.7227]])
        '''

        # (226/self.width, 487, 306, 479)

        # bbox *= torch.FloatTensor(
        #     [[self.width / width, self.height / height, self.width / width, self.height / height]])
        bbox *= torch.FloatTensor(
            [[ width/self.width, height/self.height, width/self.width, height/self.height]])
        # cls_conf = boxes[:, 5]
        cls_conf = torch.FloatTensor([float(x[1]) for x in detections]).reshape(-1)
        # cls_ids = boxes[:, 6].long()
        cls_ids = torch.IntTensor([float(x[0]) for x in detections]).reshape(-1)
        a, b, c = bbox.numpy(), cls_conf.numpy(), cls_ids.numpy()
        # print('*' * 88)
        # print(a)  # (-1,4)
        # print(type(a), a.shape)  # (-1,4)
        # print('-' * 22)
        # print(b)
        # print('-' * 22)
        # print(c)
        return a, b, c

    def load_class_names(self, namesfile):
        with open(namesfile, 'r', encoding='utf8') as fp:
            class_names = [line.strip() for line in fp.readlines()]
        return class_names


def demo():
    import os
    cfgfile = os.path.join(os.environ.get('DARKNET_PATH', './'), "cfg", "yolov4.cfg")
    weightsfile = os.path.join(os.environ.get('DARKNET_PATH', './'), "cfg", "coco.data")
    namesfile = os.path.join(os.environ.get('DARKNET_PATH', './'), "cfg", "yolov4.weights")
    yolo = yolov4(cfgfile, weightsfile, namesfile)
    print("yolo.size =", yolo.size)  # yolo.size = (608, 608)
    root = "./demo"
    resdir = os.path.join(root, "results")
    os.makedirs(resdir, exist_ok=True)
    files = [os.path.join(root, file) for file in os.listdir(root) if file.endswith('.jpg')]
    files.sort()
    for filename in files:
        img = cv2.imread(filename)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # bbox, cls_conf, cls_ids = yolo(img)
        yolo(img)

        # image = darknet.draw_boxes(detections, frame_resized, class_colors)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


if __name__ == "__main__":
    demo()
