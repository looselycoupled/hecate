# https://www.tensorflow.org/programmers_guide/low_level_intro

import numpy as np
import tensorflow as tf



sess = tf.Session()

x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)
z = x + y

print(sess.run(z, feed_dict={x: 44, y: 5}))


# my_data = [
# [0, 1,],
# [2, 3,],
# [4, 5,],
# [6, 7,],
# ]
# slices = tf.data.Dataset.from_tensor_slices(my_data)
# next_item = slices.make_one_shot_iterator().get_next()
#
# while True:
#     try:
#         print(sess.run(next_item))
#     except tf.errors.OutOfRangeError:
#         break


# r = tf.random_normal([10,3])
# dataset = tf.data.Dataset.from_tensor_slices(r)
# iterator = dataset.make_initializable_iterator()
# next_row = iterator.get_next()
# sess.run(iterator.initializer)
# while True:
#     try:
#         print(sess.run(next_row))
#     except tf.errors.OutOfRangeError:
#         break


x = tf.placeholder(tf.float32, shape=[None, 3])
y = tf.layers.dense(x, units=1)

init = tf.global_variables_initializer()
sess.run(init)

print(sess.run(y, {x: [[1, 2, 3],[4, 5, 6]]}))



x = tf.constant([[1], [2], [3], [4]], dtype=tf.float32)
y_true = tf.constant([[0], [-1], [-2], [-3]], dtype=tf.float32)
y_pred = tf.layers.dense(x, units=1)

loss = tf.losses.mean_squared_error(labels=y_true, predictions=y_pred)

optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)
for i in range(1200):
  _, loss_value = sess.run((train, loss))
  print(loss_value)

print(sess.run(y_pred))
