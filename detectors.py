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
    video_FourCC = cv2.VideoWriter_fourcc('F', 'F', 'V', '1')
    video_fps = vid.get(cv2.CAP_PROP_FPS)
    video_size = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
                  int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    total_frames = vid.get(cv2.CAP_PROP_FRAME_COUNT)
    print('Total frames in the input video are ', total_frames)
    isOutput = True if output_path != "" else False
    if isOutput:
        print("!!! TYPE:", type(output_path), type(video_FourCC), type(video_fps), type(video_size))
    else:
        print("!!! TYPE:", type(output_path), type(video_FourCC), type(video_fps), type(video_size))
        output_path = video_path.split('.')[0] + '_labeled_video_' + yolo.model_name + '.mp4'
    out = cv2.VideoWriter(output_path, video_FourCC, video_fps, video_size)
    accum_time = 0
    curr_fps = 0
    curr_frame_number = 0
    fps = "FPS: ??"
    prev_time = timer()
    fps_overtime = []
    while total_frames != curr_frame_number:
        return_value, frame = vid.read()
        image = Image.fromarray(frame)
        image, model_name = yolo.detect_image(image)
        result = np.asarray(image)
        curr_time = timer()
        exec_time = curr_time - prev_time
        prev_time = curr_time
        accum_time = accum_time + exec_time
        curr_fps = curr_fps + 1
        curr_frame_number += 1
        fps_overtime.append(curr_fps)
        avg_curr_fps = np.mean(fps_overtime)
        if accum_time > 1:
            accum_time = accum_time - 1
            fps = "Labeling FPS: " + str(curr_fps) + ' , Avg. Labeling FPS: ' + str(
                np.round(avg_curr_fps, 2))
            curr_fps = 0
        cv2.putText(result, text=fps, org=(10, 25), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.850, color=(255, 0, 0), thickness=2)
        out.write(np.uint8(result))

        # In case nothing works out, just save images from the video
        # img_name = 'test_data/imgs/img_{}.png'.format(time.time())
        # plt.imsave(img_name, result)

        print('Frame {}, with rate of {} FPS'.format(curr_frame_number, fps))
    yolo.close_session()


def detect_in_image(yolo, img_path):
    image = Image.open(img_path)
    r_image, model_name = yolo.detect_image(image)
    r_image.show()
    img_name = 'test_data/imgs/img_{}_{}.png'.format(model_name, time.time())
    plt.imsave(img_name, r_image)
    print('Image saved {} with {} model'.format(img_name, model_name))
    yolo.close_session()
