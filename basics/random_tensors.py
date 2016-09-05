"""
Example to generate tensors with random values using TensorFlow
"""

import tensorflow as tf

__author__ = 'Abhijit Bose'
__email__ = 'boseae@gmail.com'
__version__ = '1.0.0'
__status__ = 'Research'

# create a tensor with uniformly distributed numbers
uniform_t = tf.random_uniform([100], minval=0, maxval=1, dtype=tf.float32)
with tf.Session() as sess:
  print(sess.run(uniform_t))

# create a tensor with normal distribution with mean=0 and stddev=3
normal_t = tf.random_normal([100], mean=0, stddev=3)
with tf.Session() as sess:
  print(sess.run(normal_t))

# run both in a session
with tf.Session() as sess:
  print(sess.run(uniform_t))
  print(sess.run(normal_t))
