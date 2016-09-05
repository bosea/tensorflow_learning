"""
Example script to calculate symbolic partial derivatives in TF
"""

import tensorflow as tf

__author__ = 'Abhijit Bose'
__email__ = 'boseae@gmail.com'
__version__ = '1.0.0'
__status__ = 'Research'

x = tf.placeholder(tf.float32)
y = x * (x + 1)

# Construct symbolic partial derivative of y wrto x
dydx = tf.gradients(y, x)
dydx_actual = 2 * x + 1
with tf.Session() as sess:
  grad = sess.run(dydx, feed_dict={x:10.0})
  print(grad)
  print(sess.run(dydx_actual, feed_dict={x:10.0}))
  print("Note that tf.gradients returns a tensor, 1-dim, in this case.")

# Example of constructing partial derivatives of two variables
x1 = tf.placeholder(tf.float32)
x2 = tf.placeholder(tf.float32)
y = x1 * x2

# construct symbolic partial derivatives of y wrto x1 and x2
# tf.gradients will calculate [dydx1, dydx2] as a tensor.

dydx = tf.gradients(y, [x1, x2])

with tf.Session() as sess:
  grad = sess.run(dydx, feed_dict={x1:10.0, x2:20.0})
  print(grad)
  print("Note that tf.gradients returns a tensor, 2-dim, in this case.")
