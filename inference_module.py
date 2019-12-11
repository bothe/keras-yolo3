from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import tensorflow as tf
from PIL.Image import fromarray
from matplotlib import pyplot as plt

from inference.img_reader import read_img_infer
from inference.utils import decode_netout, draw_boxes, postprocess_boxes_tf, draw_boxes_tf
from utils.utils import postprocess_boxes, nms, draw_bbox
from yolo3.model import yolo_boxes_and_scores
from yolo_detection_cv_utils import get_anchors, load_classes

# Load names of classes
classes = load_classes("model_data/coco_classes.txt")
nb_classes = len(classes)

tiny_yolo = False
file = 'test_data/frame66.jpg'
model_image_size = (416, 416)
infer_image, org_image = read_img_infer(file, model_image_size)
org_image_shape = (org_image.size[0], org_image.size[1])

if tiny_yolo:
    f = tf.gfile.GFile("pb_models/yolov3-tiny.pb", 'rb')
    anchors = get_anchors('model_data/tiny_yolo_anchors.txt')
else:
    f = tf.gfile.GFile("pb_models/yolo.pb", 'rb')
    anchors = get_anchors('model_data/yolo_anchors.txt')

graph_def = tf.GraphDef()
graph_def.ParseFromString(f.read())
f.close()

sess = tf.InteractiveSession()
sess.graph.as_default()
tf.import_graph_def(graph_def)

if tiny_yolo:
    output_tensors = [sess.graph.get_tensor_by_name('import/conv2d_10/BiasAdd:0'),
                      sess.graph.get_tensor_by_name('import/conv2d_13/BiasAdd:0')]
else:
    output_tensors = [sess.graph.get_tensor_by_name('import/conv2d_59/BiasAdd:0'),
                      sess.graph.get_tensor_by_name('import/conv2d_67/BiasAdd:0'),
                      sess.graph.get_tensor_by_name('import/conv2d_75/BiasAdd:0')]

predictions = sess.run(output_tensors, {'import/input_1:0': infer_image})
net_out = predictions[0]
n_shape = net_out.shape
net_out_reshaped = np.reshape(net_out, (1, n_shape[1], n_shape[2], 3, 5 + nb_classes))
features = tf.convert_to_tensor(net_out_reshaped)

num_layers = len(predictions)
anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]] if num_layers == 3 else [[3, 4, 5], [1, 2, 3]]  # default setting

all_boxes = []
for l in range(num_layers):
    net_out = predictions[l]
    n_shape = net_out.shape
    net_out_reshaped = np.reshape(net_out, (1, n_shape[1], n_shape[2], 3, 5 + nb_classes))
    features = tf.convert_to_tensor(net_out_reshaped)
    x = tf.Session().run(yolo_boxes_and_scores(features, anchors[anchor_mask[0]], nb_classes,
                                               model_image_size, org_image_shape))
    boxes = np.concatenate([x[0], np.reshape(x[2][0], (n_shape[1]*n_shape[1]*3, 1)), x[1]], axis=1)
    all_boxes.extend(boxes)
boxes_, scores_, classes_ = postprocess_boxes_tf(all_boxes, score_threshold=.3)
image = draw_boxes_tf(boxes_, scores_, classes_, classes, org_image)
image.show()

#########################################################################################################
bboxes = postprocess_boxes(all_boxes, org_image, model_image_size[0], 0.3)
bboxes = nms(bboxes, 0.45, method='nms')
image = draw_bbox(org_image, bboxes, classes)
image = fromarray(image)
image.show()

#########################################################################################################
pred_bbox = np.concatenate([np.reshape(predictions[0], (-1, 5 + nb_classes)),
                            np.reshape(predictions[0], (-1, 5 + nb_classes)),
                            np.reshape(predictions[0], (-1, 5 + nb_classes))], axis=0)

bboxes = postprocess_boxes(pred_bbox, org_image, model_image_size[0], 0.3)
bboxes = nms(bboxes, 0.45, method='nms')
image = draw_bbox(org_image, bboxes, classes)
image = fromarray(image)
image.show()

all_boxes = []
for l in range(num_layers):
    net_out = predictions[l][0]
    n_shape = net_out.shape
    net_out_reshaped = np.reshape(net_out, (n_shape[0], n_shape[1], 3, 5 + nb_classes))
    boxes = decode_netout(net_out_reshaped, np.hstack(anchors[anchor_mask[l]]), nb_classes)
    all_boxes.extend(boxes)

image = draw_boxes(org_image, all_boxes, labels=classes)
plt.imshow(image)
plt.show()
print('scores')
