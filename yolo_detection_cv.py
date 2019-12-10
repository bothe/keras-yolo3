import time

import cv2
import numpy as np
from matplotlib import pyplot as plt

from yolo_detection_cv_utils import return_boxes, draw_on_frame, load_classes

# net = cv2.dnn.readNet("weights/yolov3.weights", "configs/yolov3.cfg")  # Original yolov3
net = cv2.dnn.readNet("weights/yolov3-tiny.weights", "configs/yolov3-tiny.cfg")  # Tiny Yolo

classes = load_classes("model_data/coco_classes.txt")

layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

np.random.seed(1234)
colors = np.random.uniform(0, 255, size=(len(classes), 3))
video = False

if video:
    # loading image
    cap = cv2.VideoCapture("test_data/cold_lake_trafic.mp4")  # 0 for 1st webcam
    starting_time = time.time()
    frame_id = 0

    while True:
        _, frame = cap.read()  #
        frame_id += 1
        height, width, channels = frame.shape
        # detecting objects
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)  # reduce 416 to 320
        net.setInput(blob)
        outs = net.forward(output_layers)
        # print(outs[1])

        class_ids, confidences, boxes = return_boxes(outs, height, width)
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.4, 0.6)

        frame, fps = draw_on_frame(frame, boxes, indexes, colors, classes, confidences,
                                   class_ids, starting_time, frame_id)

        img_name = 'test_data/imgs/img_{}.png'.format(time.time())
        plt.imsave(img_name, frame)
        print(img_name, fps)
    cap.release()

else:
    frame = plt.imread('test_data/frame66.jpg')
    height, width, channels = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)  # reduce 416 to 320
    net.setInput(blob)
    outs = net.forward(output_layers)
    starting_time = time.time()

    class_ids, confidences, boxes = return_boxes(outs, height, width)
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.4, 0.6)

    frame, fps = draw_on_frame(frame, boxes, indexes, colors, classes, confidences,
                               class_ids, starting_time, 1)

    img_name = 'test_data/imgs/img_{}.png'.format(time.time())
    plt.imsave(img_name, frame)
    print(img_name, fps)
