# -*- coding: utf-8 -*-
"""
Class definition of YOLO_v3 style detection handlers in image or video
"""

import time
from timeit import default_timer as timer

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt


def detect_in_video(yolo, video_path, output_path=""):
    import cv2
    vid = cv2.VideoCapture(video_path)
    if not vid.isOpened():
        raise IOError("Couldn't open webcam or video")
    video_FourCC = int(vid.get(cv2.CAP_PROP_FOURCC))
    video_fps = vid.get(cv2.CAP_PROP_FPS)
    video_size = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
                  int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    total_frames = vid.get(cv2.CAP_PROP_FRAME_COUNT)
    isOutput = True if output_path != "" else False
    if isOutput:
        print("!!! TYPE:", type(output_path), type(video_FourCC), type(video_fps), type(video_size))
        out = cv2.VideoWriter(output_path, video_FourCC, video_fps, video_size)
    accum_time = 0
    curr_fps = 0
    curr_frame_number = 0
    fps = "FPS: ??"
    prev_time = timer()
    while total_frames != curr_frame_number:
        return_value, frame = vid.read()
        image = Image.fromarray(frame)
        image = yolo.detect_image(image)
        result = np.asarray(image)
        curr_time = timer()
        exec_time = curr_time - prev_time
        prev_time = curr_time
        accum_time = accum_time + exec_time
        curr_fps = curr_fps + 1
        curr_frame_number += 1
        if accum_time > 1:
            accum_time = accum_time - 1
            fps = "FPS: " + str(curr_fps)
            curr_fps = 0
        cv2.putText(result, text=fps, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.50, color=(255, 0, 0), thickness=2)
        # cv2.namedWindow("result", cv2.WINDOW_NORMAL)
        # cv2.imshow("result", result)
        if isOutput:
            out.write(result)
        # plt.imshow(result), plt.show()
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #   break
        img_name = 'test_data/imgs/img_{}.png'.format(time.time())
        plt.imsave(img_name, result)
        print('Frame {}, Image saved {}'.format(curr_frame_number, img_name))
        # cv2.imwrite('test_data/imgs/img{}'.format(time.time()), result)
    yolo.close_session()


def detect_in_image(yolo):
    while True:
        img = input('Input image filename:')
        try:
            image = Image.open(img)
        except:
            print('Open Error! Try again!')
            continue
        else:
            r_image = yolo.detect_image(image)
            r_image.show()
            img_name = 'test_data/imgs/img_{}.png'.format(time.time())
            plt.imsave(img_name, r_image)
            print('Image saved {}'.format(img_name))
    yolo.close_session()
