# package.module
# module description
#
# Author:   Allen Leis <allen.leis@gmail.com>
# Created:  timestamp
#
# Copyright (C) 2017 Allen Leis
# For license information, see LICENSE
#
# ID: filename.py [] allen.leis@gmail.com $

"""
module description
"""

##########################################################################
# Imports
##########################################################################

import tensorflow as tf

from hecate.utils.mixins import LoggableMixin


##########################################################################
# Classes
##########################################################################

class DeepQNetwork(LoggableMixin):

    def __init__(self, session, name, action_size):
        self.session = session
        self.name = name
        self.action_size = action_size
        self._setup()

    def _setup(self):
        with tf.variable_scope(self.name):
            # https://www.tensorflow.org/api_docs/python/tf/layers/conv2d
            # https://www.tensorflow.org/api_docs/python/tf/layers/dense

            # placeholders
            self.input = tf.placeholder(shape=[None, 84, 84, 1], dtype=tf.float32, name="input")
            self.labels = tf.placeholder(shape=[None], dtype=tf.uint8, name="labels")
            self.actions = tf.placeholder(shape=[None], dtype=tf.uint8, name="actions")

            # The first hidden layer convolves 16 8 × 8 filters with stride 4 with
            # the input image and applies a rectifier nonlinearity [10, 18].
            conv_layer_1 = tf.layers.conv2d(
                self.input,
                filters=16,
                kernel_size=(8, 8),
                strides=4,
                data_format="channels_last",
                activation=tf.nn.relu,
                name="conv_layer_1",
            )
            # The second hidden layer convolves 32 4 × 4 filters with stride 2, again
            # followed by a rectifier nonlinearity.
            conv_layer_2 = tf.layers.conv2d(
                conv_layer_1,
                filters=32,
                kernel_size=(4, 4),
                strides=2,
                data_format="channels_last",
                activation=tf.nn.relu,
                name="conv_layer_2",
            )

            # flatten before going into fully connected layers
            # https://www.tensorflow.org/api_docs/python/tf/layers/flatten
            flatten_layer = tf.layers.flatten(conv_layer_2, "flatten_layer")

            # The final hidden layer is fully-connected and consists of 256
            # rectifier units.
            dense_layer_1 = tf.layers.dense(
                flatten_layer,
                256,
                activation=tf.nn.relu,
                name="dense_layer_1",
            )

            # The output layer is a fully-connected linear layer with a single output
            # for each valid action.  Outputs action values!
            self.output_layer = tf.layers.dense(
                dense_layer_1,
                self.action_size,
                name="output_layer"
            )


            # training operations

            # From paper:
            #   In these experiments, we used the RMSProp algorithm with minibatches of size 32.
            # https://www.tensorflow.org/api_docs/python/tf/train/RMSPropOptimizer
            self._optimizer = tf.train.RMSPropOptimizer(
                0.00025, # learning_rate,
                decay=0.99,
                momentum=0.0,
                epsilon=1e-6,
                name='RMSProp'
            )

            # fix actions from output_layer for training
            # https://www.tensorflow.org/api_docs/python/tf/reduce_sum
            fixed_action_values = tf.reduce_sum(
                self.output_layer * tf.one_hot(self.actions, self.action_size),
                axis=[1]
            )

            # define loss function using labels and actions
            # https://www.tensorflow.org/api_docs/python/tf/losses/mean_squared_error
            self.loss = tf.losses.mean_squared_error(self.labels, fixed_action_values)

            # training/optimization function using above loss
            self.optimize = self._optimizer.minimize(self.loss)


    def train(self, input, labels, actions):
        train_result, loss = self.session.run(
            [self.optimize, self.loss],
            { self.input: input, self.labels: labels, self.actions: actions })

        return loss


    def predict(self, input):
        return self.session.run(self.output_layer, { self.input: input })


    def copy(self, source):
        external_vars = sorted(
            [v for v in tf.trainable_variables() if v.name.startswith(source.name)],
            key=lambda item: item.name
        )
        internal_vars = sorted(
            [v for v in tf.trainable_variables() if v.name.startswith(self.name)],
            key=lambda item: item.name
        )
        self.session.run([
            internal.assign(self.session.run(external))
            for internal, external in zip(internal_vars, external_vars)
        ])
