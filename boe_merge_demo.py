import cv2
import multiprocessing
from itertools import cycle
import glob
import torch
from utils.log import get_logger
import warnings
import argparse
from detector import build_detector
from deep_sort import build_tracker
from utils.draw import draw_boxes
from utils.parser import get_config
from utils.io import write_results
import os
import time
import ffmpeg
import numpy as np
import asyncio
import websockets
import threading
import datetime
import json

height, width, channels = 480, 640, 3
HOST = "0.0.0.0"
# np.set_printoptions(threshold=9223372036854775807)

connected = set()
total_count = 0
q0_count, q1_count, q2_count, q3_count, \
q4_count, q5_count = 0, 0, 0, 0, 0, 0
h0, h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, \
h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23 = \
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
realHeat = []
realTrack = {
    'id': [],
    'point': [],
}


class VideoTracker(object):
    def __init__(self, cfg, args, i_image_q, o_image_q, q_dets_results_q):
        self.cfg = cfg
        self.args = args
        self.i_image_q = i_image_q
        self.o_image_q = o_image_q
        self.q_dets_results_q = q_dets_results_q
        self.logger = get_logger("root")
        self.im_c = 3

        use_cuda = args.use_cuda and torch.cuda.is_available()
        if not use_cuda:
            warnings.warn("Running in cpu mode which maybe very slow!", UserWarning)

        if args.display:
            cv2.namedWindow("test", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("test", args.display_width, args.display_height)

        if args.cam != -1:
            print("Using webcam " + str(args.cam))
            self.vdo = cv2.VideoCapture(args.cam)
        # else:
        #     self.vdo = cv2.VideoCapture()
        self.detector = build_detector(cfg, use_cuda=use_cuda)
        self.deepsort = build_tracker(cfg, use_cuda=use_cuda)
        self.class_names = self.detector.class_names

    def __enter__(self):
        if self.args.cam != -1:
            ret, frame = self.vdo.read()
            assert ret, "Error: Camera error"
            self.im_width = frame.shape[1]
            self.im_height = frame.shape[0]

        else:
            # assert os.path.isfile(self.video_path), "Path error"
            # self.vdo.open(self.video_path)
            # self.im_width = int(self.vdo.get(cv2.CAP_PROP_FRAME_WIDTH))
            # self.im_height = int(self.vdo.get(cv2.CAP_PROP_FRAME_HEIGHT))
            # assert self.vdo.isOpened()
            # read from queue
            frame = self.i_image_q.get()
            self.im_height, self.im_width, self.im_c = frame.shape
            assert self.im_c == 3

        if self.args.save_path:
            os.makedirs(self.args.save_path, exist_ok=True)

            # path of saved video and results
            self.save_video_path = os.path.join(self.args.save_path, "results.avi")
            self.save_results_path = os.path.join(self.args.save_path, "results.txt")

            # create video writer
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            self.writer = cv2.VideoWriter(self.save_video_path, fourcc, 20, (self.im_width, self.im_height))

            # logging
            self.logger.info("Save results to {}".format(self.args.save_path))

        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if exc_type:
            print(exc_type, exc_value, exc_traceback)

    def run(self):
        results = []
        idx_frame = 0
        while 1:
            ori_im = self.i_image_q.get()
            idx_frame += 1
            # if idx_frame % self.args.frame_interval:
            #     continue

            start = time.time()
            # _, ori_im = self.vdo.retrieve()
            im = cv2.cvtColor(ori_im, cv2.COLOR_BGR2RGB)

            # do detection
            bbox_xywh, cls_conf, cls_ids = self.detector(im)
            # print(bbox_xywh,cls_conf,cls_ids)
            '''
                [[1.4130478e+01 4.5147302e+02 2.9447939e+01 4.3295568e-01]
                 [4.4670758e+02 3.9561349e+02 5.8724094e+01 1.3942775e-01]
                 [9.6704277e+01 5.3988617e+02 5.2978405e+01 3.7567344e-01]
                 [2.3981223e+02 5.1301807e+02 4.7807354e+01 4.3671408e-01]] [0.8002365  0.82209855 0.9805381  0.9935718 ] [0 0 0 0]
            '''
            # os._exit(0)

            # select person class
            mask = cls_ids == 0

            bbox_xywh = bbox_xywh[mask]
            # bbox dilation just in case bbox too small, delete this line if using a better pedestrian detector
            # bbox_xywh[:, 3:] *= 1.2
            cls_conf = cls_conf[mask]

            # do tracking
            try:
                outputs = self.deepsort.update(bbox_xywh, cls_conf, im)
            except:
                # print(im)
                # print(im.shape)#(480, 640, 3)
                raise

            # draw boxes for visualization
            if len(outputs) > 0:
                bbox_tlwh = []
                bbox_xyxy = outputs[:, :4]
                identities = outputs[:, -1]
                ori_im = draw_boxes(ori_im, bbox_xyxy, identities)

                for bb_xyxy in bbox_xyxy:
                    bbox_tlwh.append(self.deepsort._xyxy_to_tlwh(bb_xyxy))
                    # {frame},{id},{x1},{y1},{w},{h}
                    # results -1,1,236,211,54,217
                    '''
                        (-1,
                         [(236, 211, 54, 217),
                          (82, 249, 62, 192),
                          (463, 213, 64, 65),
                          (0, 180, 31, 205)],
                         array([1, 2, 3, 4]))
                    (idx_frame - 1, bbox_tlwh, identities)
                    '''
                    self.q_dets_results_q.put((idx_frame - 1, bbox_tlwh, identities))
                    # print(len(results))

                # results.append((idx_frame - 1, bbox_tlwh, identities))

            end = time.time()

            if self.args.display:
                cv2.imshow("test", ori_im)
                cv2.waitKey(1)

            if self.args.save_path:
                self.writer.write(ori_im)

            # save results
            # write_results(self.save_results_path, results, 'mot')

            # logging
            self.logger.info("time: {:.03f}s, fps: {:.03f}, detection numbers: {}, tracking numbers: {}" \
                             .format(end - start, 1 / (end - start), bbox_xywh.shape[0], len(outputs)))

            self.o_image_q.put(ori_im[:, :, ::-1])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("VIDEO_PATH", type=str)
    parser.add_argument("--config_detection", type=str, default="./configs/yolov3.yaml")
    parser.add_argument("--config_deepsort", type=str, default="./configs/deep_sort.yaml")
    # parser.add_argument("--ignore_display", dest="display", action="store_false", default=True)
    parser.add_argument("--display", action="store_true")
    parser.add_argument("--frame_interval", type=int, default=1)
    parser.add_argument("--display_width", type=int, default=800)
    parser.add_argument("--display_height", type=int, default=600)
    parser.add_argument("--save_path", type=str, default="")
    parser.add_argument("--cpu", dest="use_cuda", action="store_false", default=True)
    parser.add_argument("--camera", action="store", dest="cam", type=int, default="-1")
    return parser.parse_args()


def vt(cfg, args, i_image_q, o_image_q, q_dets_results_q):
    with VideoTracker(cfg, args, i_image_q, o_image_q, q_dets_results_q) as vdo_trk:
        vdo_trk.run()


def vt_push_rtmp(o_image_q: multiprocessing.Queue):
    while 1:
        images = o_image_q.get()
        # print(images.shape) # (480, 640, 3)
        if not isinstance(images, np.ndarray):
            images = np.asarray(images)
        ffmpegprocess.stdin.write(images.astype(np.uint8).tobytes())


async def server(websocket, path: str):
    if len(connected) > 50:
        return
    global total_count, q0_count, q1_count, q2_count, q3_count, \
        q4_count, q5_count, h0, h1, h2, h3, h4, h5, \
        h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, \
        h16, h17, h18, h19, h20, h21, h22, h23, realHeat, realTrack

    q0_count, q1_count, q2_count, q3_count, \
    q4_count, q5_count = 10, 20, 30, 40, 50, 60
    if path.endswith('realTrack'):
        connected.add(websocket)
    try:
        if path.endswith('reportDetail'):
            xx = {"areaCount": {"a": 10, "b": 10, "c": 11, "d": 10},
                  "ageCount": {"46-60": "10%", "61": "15%", "31-45": "5%", "21-30": "50%", "0-20": "20%"},
                  "genderCount": {"men": {"num": 2, "percent": "10%"}, "women": {"num": 4, "percent": "20%"}},
                  "totalCount": f"{total_count}", "faceCount": 0,
                  "personCount": {"60-120": f"{q4_count}", "0-15": f"{q0_count}", "30-45": f"{q2_count}",
                                  "15-30": f"{q1_count}", ">120": f"{q5_count}",
                                  "45-60": f"{q3_count}"}}
            await websocket.send(json.dumps(xx))
        elif path.endswith('reportByHour'):
            xx = {"0": h0, "1": h1, "2": h2, "3": h3, "4": h4, "5": h5, "6": h6, "7": h7, "8": h8, "9": h9, "10": h10,
                  "11": h11, "12": h12, "13": h13, "14": h14, "15": h15, "16": h16, "17": h17, "18": h18, "19": h19,
                  "20": h20, "21": h21, "22": h22, "23": h23}
            await websocket.send(json.dumps(xx))
        elif path.endswith('realHeat'):
            xx = []
            for box in realHeat:
                x1 = box[0]
                y1 = box[1]
                bw = box[2]
                bh = box[3]
                xx.append({"x": int(x1 + bw / 2), "y": int(y1 + bh), "num": 1})
            await websocket.send(json.dumps(xx))

        else:
            nowhour = datetime.datetime.now().hour
            while 1:
                for conn in connected:
                    realbox = realTrack['point']
                    realname = realTrack['id']
                    '''
                    [(236, 211, 54, 217),
                      (82, 249, 62, 192),
                      (463, 213, 64, 65),
                      (0, 180, 31, 205)],
                       [1, 2, 3, 4])
                    '''

                    jsondumpbatch = []
                    for i in range(len(realbox)):
                        # bname = realname[i]
                        # bname = '1_136'
                        # bname = f"1_{realname[i][0:3]}"
                        bname = f"{nowhour}_{realname[i]}"
                        b = realbox[i]
                        x1, y1, bw, bh = b[0], b[1], b[2], b[3]
                        jsondump = [
                            {"id": bname, "x": int((x1 + bw / 2)), "y": int((y1 + bh / 2)), "time": time.time()}
                        ]

                        jsondumpbatch.extend(jsondump)

                    try:
                        await conn.send(json.dumps(jsondumpbatch))
                    except:
                        connected.remove(conn)
                        raise
                        # pass
                await asyncio.sleep(.6)

                if not len(connected):
                    break
    except:
        raise
        # pass
    finally:
        # connected.clear()
        pass

    # print(await websocket.recv())


def start_wsserver():
    asyncio.set_event_loop(asyncio.new_event_loop())
    # start_server = websockets.serve(server, HOST, 8888, ping_timeout=None)
    start_server = websockets.serve(server, HOST, 8888)

    asyncio.get_event_loop().run_until_complete(start_server)
    asyncio.get_event_loop().run_forever()


def statis_to_ws(detsq: multiprocessing.Queue):
    tWsServer = threading.Thread(target=start_wsserver, args=())
    tWsServer.start()
    global total_count, q0_count, q1_count, q2_count, q3_count, \
        q4_count, q5_count, h0, h1, h2, h3, h4, h5, \
        h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, \
        h16, h17, h18, h19, h20, h21, h22, h23, realTrack
    nowhour = datetime.datetime.now().hour
    while 1:
        '''
            (-1,
             [(236, 211, 54, 217),
              (82, 249, 62, 192),
              (463, 213, 64, 65),
              (0, 180, 31, 205)],
             array([1, 2, 3, 4]))
        (idx_frame - 1, bbox_tlwh, identities)
        '''
        frameid, bbox_tlwh, identities = detsq.get()
        realTrack['id'] = identities.tolist()
        realTrack['point'] = bbox_tlwh

        # total_count = np.max(identities).item()
        total_count = max(identities.size, total_count)

        realHeat.clear()
        realHeat.extend(bbox_tlwh)
        # if nowhour == 0:
        #     h0 += len(resultgone)
        # elif nowhour == 1:
        #     h1 += len(resultgone)
        # elif nowhour == 2:
        #     h2 += len(resultgone)
        # elif nowhour == 3:
        #     h3 += len(resultgone)
        # elif nowhour == 4:
        #     h4 += len(resultgone)
        # elif nowhour == 5:
        #     h5 += len(resultgone)
        # elif nowhour == 6:
        #     h6 += len(resultgone)
        # elif nowhour == 7:
        #     h7 += len(resultgone)
        # elif nowhour == 8:
        #     h8 += len(resultgone)
        # elif nowhour == 9:
        #     h9 += len(resultgone)
        # elif nowhour == 10:
        #     h10 += len(resultgone)
        # elif nowhour == 11:
        #     h11 += len(resultgone)
        # elif nowhour == 12:
        #     h12 += len(resultgone)
        # elif nowhour == 13:
        #     h13 += len(resultgone)
        # elif nowhour == 14:
        #     h14 += len(resultgone)
        # elif nowhour == 15:
        #     h15 += len(resultgone)
        # elif nowhour == 16:
        #     h16 += len(resultgone)
        # elif nowhour == 17:
        #     h17 += len(resultgone)
        # elif nowhour == 18:
        #     h18 += len(resultgone)
        # elif nowhour == 19:
        #     h19 += len(resultgone)
        # elif nowhour == 20:
        #     h20 += len(resultgone)
        # elif nowhour == 21:
        #     h21 += len(resultgone)
        # elif nowhour == 22:
        #     h22 += len(resultgone)
        # elif nowhour == 23:
        #     h23 += len(resultgone)

        # try:
        #     while 1:
        #         detsq.get_nowait()
        # except:
        #     pass
        # time.sleep(.3)


if __name__ == "__main__":
    args = parse_args()
    cfg = get_config()
    cfg.merge_from_file(args.config_detection)
    cfg.merge_from_file(args.config_deepsort)
    input_frame_q = multiprocessing.Queue(maxsize=200)
    output_frame_q = multiprocessing.Queue(maxsize=2000)
    q_dets_results_q = multiprocessing.Queue(maxsize=2000)
    statisprocess = multiprocessing.Process(target=statis_to_ws, args=(q_dets_results_q,))
    statisprocess.start()

    ffmpegprocess = (
        ffmpeg
            .input('pipe:', y=None
                   # , vsync=0
                   , hwaccel="cuda", hwaccel_output_format="cuda", format='rawvideo'
                   , pix_fmt='rgb24'
                   , r=16
                   , s=f'{width}x{height}')
            # .filter('scale', width='570', height='320')
            .output('rtmp://192.168.8.121/live/bbb'
                    , pix_fmt='yuv420p'
                    , vcodec="h264_nvenc", format='flv'
                    , an=None
                    # , r=17
                    # , q=10
                    , preset="p6", tune="hq", bufsize="5M", maxrate="4M", qmin=0, g=250, bf=3, b_ref_mode="middle"
                    , i_qfactor=0.75, b_qfactor=1.1
                    # ,s='570x320'
                    # -preset p6 -tune hq -b:v 5M -bufsize 5M -maxrate 10M -qmin 0 -g 250 -bf 3 -b_ref_mode middle -temporal-aq 1 -rc-lookahead 20 -i_qfactor 0.75 -b_qfactor 1.1
                    ).overwrite_output()
            .run_async(pipe_stdin=True))

    vt = multiprocessing.Process(target=vt, args=(cfg, args, input_frame_q, output_frame_q, q_dets_results_q))
    vt.start()
    p_vt_push_rtmp = multiprocessing.Process(target=vt_push_rtmp, args=(output_frame_q,))
    p_vt_push_rtmp.start()

    for video in cycle(glob.glob("./videos/*.avi")):
        cap = cv2.VideoCapture(video)
        while cap.isOpened():
            if cap.grab():
                flag, frame = cap.retrieve()
                if not flag:
                    continue
                else:
                    # print('.',end='')
                    input_frame_q.put(frame)
            else:
                print('grab error ......')
                break
        cap.release()
        # break
