#
# https://www.tensorflow.org/api_docs/python/tf/image/rgb_to_grayscale
# https://www.tensorflow.org/api_docs/python/tf/image/crop_to_bounding_box
# https://www.tensorflow.org/api_docs/python/tf/image/resize_image_with_crop_or_pad
# https://www.tensorflow.org/api_docs/python/tf/image/resize_images
# https://www.tensorflow.org/api_docs/python/tf/squeeze
#
# http://scikit-image.org/docs/stable/api/skimage.color.html#rgb2gray
#
# Other Resources
#    https://taesikna.github.io/resize_images.html
#    https://hackernoon.com/how-tensorflows-tf-image-resize-stole-60-days-of-my-life-aba5eb093f35
#    http://yeephycho.github.io/2016/08/15/image-data-in-tensorflow/

# From: Playing atari with deep reinforcement learning (page 5)
# Working directly with raw Atari frames, which are 210 × 160 pixel images with
# a 128 color palette, can be computationally demanding, so we apply a basic
# preprocessing step aimed at reducing the input dimensionality. The raw frames
# are preprocessed by first converting their RGB representation to gray-scale
# and down-sampling it to a 110×84 image. The final input representation is
# obtained by cropping an 84 × 84 region of the image that roughly captures the
# playing area. The final cropping stage is only required because we use the GPU
# implementation of 2D convolutions from [11], which expects square inputs.


import tensorflow as tf
import numpy as np
from skimage.color import rgb2gray

RESIZE_METHOD = tf.image.ResizeMethod.NEAREST_NEIGHBOR
INPUT_SHAPE = [210, 160, 3]
OUTPUT_SHAPE = [84, 84]

FLAG = False

def wrangle_image(data):
    global FLAG
    # TODO: wrangle image using matplotlib, skimage, or tf

    # convert to grayscale (rgb_to_grayscale)
    data = rgb2gray(data)

    if not FLAG:
        FLAG = True
        print(data)
        import pdb; pdb.set_trace()
    # crop & resize (crop_to_bounding_box or resize_image_with_crop_or_pad)

    # sqeeze data (sqeeze)

    # return
    return data
