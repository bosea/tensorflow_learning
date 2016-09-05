"""
Example of solving a basic linear regression problem
"""

import numpy as np
import tensorflow as tf

__author__ = 'Abhijit Bose'
__email__ = 'boseae@gmail.com'
__version__ = '1.0.0'
__status__ = 'Research'


# generate input points (npoints) and define the linear equation.
npoints = 100
a = tf.constant(0.5)
b = tf.constant(0.8)

# Note: The value of a feed_dict cannot be a tf.Tensor object. So if
# we create x points as a tf.random_normal as in:
#   xp = tf.random_normal([npoints], mean=0.0, stddev=0.5)
# we will not be able to feed it later in the graph.
# Acceptable feed values include Python scalars, strings, lists, or
# numpy ndarrays. Therefore, we will define x points as np arrays.
xp = np.random.normal(loc=0.0, scale=0.5, size=npoints)
yp = a * xp + b

# define cost function and parameters to calculate
a1 = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b1 = tf.Variable(tf.zeros([1]))
xs = tf.placeholder(tf.float32, shape=(npoints))
ys = a1 * xs + b1
cost = tf.reduce_mean(tf.square(yp - ys))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(cost)

init = tf.initialize_all_variables()

# execute the graph in a session and perform niter gradient descent
#   iterations
niter = 25
with tf.Session() as sess:
  sess.run(init)
  for step in range(0, niter):
    [stepCost, _] = sess.run([cost, train], feed_dict={xs:xp})
    if (step % 5 == 0):
      print('step = ', step)
      print('  cost = ', stepCost)
      print('  a1 = ', sess.run(a1))
      print('  b1 = ', sess.run(b1))
