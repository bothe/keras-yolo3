from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

from inference.box_predictor import post_process
from inference.img_reader import read_img_infer

# Load names of classes
classesFile = "model_data/coco_classes.txt"
classes = None
with open(classesFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

tiny_yolo = False
file = 'test_data/frame66.jpg'
model_image_size = (416, 416)
infer_image, org_image = read_img_infer(file, model_image_size)

if tiny_yolo:
    f = tf.gfile.GFile("pb_models/yolov3-tiny.pb", 'rb')
else:
    f = tf.gfile.GFile("pb_models/yolo.pb", 'rb')

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

frame = post_process(infer_image[0], predictions, classes)
