from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.python.platform import gfile

from yolo3.utils import letterbox_image

file = 'test_data/frame66.jpg'

model_image_size = (416, 416)

image = Image.open(file)
img = tf.keras.preprocessing.image.load_img(file, target_size=[416, 416])

if model_image_size != (None, None):
    assert model_image_size[0] % 32 == 0, 'Multiples of 32 required'
    assert model_image_size[1] % 32 == 0, 'Multiples of 32 required'
    boxed_image = letterbox_image(image, tuple(reversed(model_image_size)))
else:
    new_image_size = (image.width - (image.width % 32),
                      image.height - (image.height % 32))
    boxed_image = letterbox_image(image, new_image_size)
image_data = np.array(boxed_image, dtype='float32')

print(image_data.shape)
image_data /= 255.
image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

f = gfile.FastGFile("pb_models/yolov3-tiny.pb", 'rb')
graph_def = tf.GraphDef()
# Parses a serialized binary message into the current message.
graph_def.ParseFromString(f.read())
f.close()

sess = tf.InteractiveSession()

sess.graph.as_default()

# Import a serialized TensorFlow `GraphDef` protocol buffer
# and place into the current default `Graph`.
tf.import_graph_def(graph_def)

output_tensors = [sess.graph.get_tensor_by_name('import/conv2d_10/BiasAdd:0'),
                  sess.graph.get_tensor_by_name('import/conv2d_13/BiasAdd:0')]

predictions = sess.run(output_tensors, {'import/input_1:0': image_data})
