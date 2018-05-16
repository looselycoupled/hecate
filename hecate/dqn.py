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
# DeepQNetwork (2013)
##########################################################################

class DeepQNetwork(LoggableMixin):

    def __init__(self, session, name, action_size):
        self.session = session
        self.name = name
        self.action_size = action_size
        self._setup_network()
        self._setup_training()

    def _setup_network(self):
        with tf.variable_scope(self.name):
            # https://www.tensorflow.org/api_docs/python/tf/layers/conv2d
            # https://www.tensorflow.org/api_docs/python/tf/layers/dense

            # placeholders
            self.input = tf.placeholder(shape=[None, 84, 84, 4], dtype=tf.uint8, name="input")
            self.labels = tf.placeholder(shape=[None], dtype=tf.float32, name="labels")
            self.actions = tf.placeholder(shape=[None], dtype=tf.int32, name="actions")

            # The first hidden layer convolves 16 8 × 8 filters with stride 4 with
            # the input image and applies a rectifier nonlinearity [10, 18].
            conv_layer_1 = tf.layers.conv2d(
                tf.to_float(self.input) / 255.0,
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


    def _setup_training(self):
        with tf.variable_scope(self.name):
            # training operations
            indexes = tf.range(tf.shape(self.input)[0]) * tf.shape(self.output_layer)[1] + self.actions
            fixed_action_values = tf.gather(
                tf.reshape(self.output_layer, [-1]),
                indexes
            )

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

            # define loss function using labels and actions
            # https://www.tensorflow.org/api_docs/python/tf/losses/mean_squared_error
            # self.loss = tf.losses.mean_squared_error(self.labels, fixed_action_values)
            self.loss = tf.losses.mean_squared_error(self.labels, fixed_action_values)
            # self.losses = tf.squared_difference(self.labels, fixed_action_values)
            # self.loss = tf.reduce_mean(self.losses)

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


##########################################################################
# DeepQNetwork (2015)
##########################################################################

# From 2015 Nature Article "Human-level control through deep reinforcement learning"
# https://www.nature.com/articles/nature14236
#
# Mnih, Volodymyr, et al. "Human-level control through deep reinforcement learning."
# Nature 518.7540 (2015): 529.
#
# The input to the neural network consists of an 84x84x4 image produced by the
# preprocessing map w. The first hidden layer convolves 32 filters of 8x8 with
# stride 4 with the input image and applies a rectifier nonlinearity. The
# second hidden layer convolves 64 filters of 4x4 with stride 2, again followed
# by a rectifier nonlinearity. This is followed by a third convolutional layer that
# convolves 64 filters of 3x3 with stride 1 followed by a rectifier. The final
# hidden layer is fully-connected and con- sists of 512 rectifier units. The output
# layer is a fully-connected linear layer with a single output for each valid action.
class DeepQNetwork2015(DeepQNetwork):

    def _setup_network(self):
        with tf.variable_scope(self.name):
            # https://www.tensorflow.org/api_docs/python/tf/layers/conv2d
            # https://www.tensorflow.org/api_docs/python/tf/layers/dense

            # placeholders
            self.input = tf.placeholder(shape=[None, 84, 84, 4], dtype=tf.uint8, name="input")
            self.labels = tf.placeholder(shape=[None], dtype=tf.float32, name="labels")
            self.actions = tf.placeholder(shape=[None], dtype=tf.int32, name="actions")

            # The first hidden layer convolves 32 filters of 8x8 with
            # stride 4 with the input image and applies a rectifier nonlinearity.
            conv_layer_1 = tf.layers.conv2d(
                tf.to_float(self.input) / 255.0,
                filters=32,
                kernel_size=(8, 8),
                strides=4,
                data_format="channels_last",
                activation=tf.nn.relu,
                name="conv_layer_1",
            )

            # The second hidden layer con- volves 64 filters of 4x4 with stride 2, again followed
            # by a rectifier nonlinearity.
            conv_layer_2 = tf.layers.conv2d(
                conv_layer_1,
                filters=64,
                kernel_size=(4, 4),
                strides=2,
                data_format="channels_last",
                activation=tf.nn.relu,
                name="conv_layer_2",
            )

            # This is followed by a third convolutional layer that
            # convolves 64 filters of 3x3 with stride 1 followed by a rectifier.
            conv_layer_3 = tf.layers.conv2d(
                conv_layer_2,
                filters=64,
                kernel_size=(3, 3),
                strides=1,
                data_format="channels_last",
                activation=tf.nn.relu,
                name="conv_layer_3",
            )

            # flatten before going into fully connected layers
            # https://www.tensorflow.org/api_docs/python/tf/layers/flatten
            flatten_layer = tf.layers.flatten(conv_layer_3, "flatten_layer")

            # The final hidden layer is fully-connected and consists of 512
            # rectifier units.
            dense_layer_1 = tf.layers.dense(
                flatten_layer,
                512,
                activation=tf.nn.relu,
                name="dense_layer_1",
            )

            # The output layer is a fully-connected linear layer with a single
            # output for each valid action.
            self.output_layer = tf.layers.dense(
                dense_layer_1,
                self.action_size,
                name="output_layer"
            )
