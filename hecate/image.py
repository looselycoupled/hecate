#
# https://www.tensorflow.org/api_docs/python/tf/image/rgb_to_grayscale
# https://www.tensorflow.org/api_docs/python/tf/image/crop_to_bounding_box
# https://www.tensorflow.org/api_docs/python/tf/image/resize_images
# https://www.tensorflow.org/api_docs/python/tf/squeeze
#
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


RESIZE_METHOD = tf.image.ResizeMethod.NEAREST_NEIGHBOR
INPUT_SHAPE = [210, 160, 3]
OUTPUT_SHAPE = [84, 84]


def wrangle_image(sess, data):
    pass
