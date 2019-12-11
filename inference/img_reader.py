import cv2
import numpy as np
from PIL import Image

from yolo3.utils import letterbox_image


def read_img_infer(file, model_image_size):
    org_image = Image.open(file)
    org_image1 = cv2.imread(file)
    if model_image_size != (None, None):
        assert model_image_size[0] % 32 == 0, 'Multiples of 32 required'
        assert model_image_size[1] % 32 == 0, 'Multiples of 32 required'
        boxed_image = letterbox_image(org_image, tuple(reversed(model_image_size)))
    else:
        new_image_size = (org_image.width - (org_image.width % 32),
                          org_image.height - (org_image.height % 32))
        boxed_image = letterbox_image(org_image, new_image_size)
    infer_image = np.array(boxed_image, dtype='float32')
    infer_image /= 255.
    infer_image = np.expand_dims(infer_image, 0)  # Add batch dimension.
    print('Image shapes, org: {}, to infer: {}'.format(org_image.size, infer_image.shape))
    return infer_image, org_image
