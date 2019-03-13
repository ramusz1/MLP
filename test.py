import tensorflow as tf
import numpy as np

import layer as lr
import functions as fn

W = np.arange(12).reshape(3,4).astype(np.float64)
x = np.arange(15).reshape(5,3).astype(np.float64)
b = np.arange(4).astype(np.float64)

print('W', W)
print('x', x)
print('b', b)

fc = lr.FullyConnected(3,4)
fc.weight = W
fc.bias = b

output = fc.forwardWithSave(x)
print('output:', output)

with tf.Session() as sess:
    W_t = tf.constant(W)
    x_t = tf.constant(x)
    b_t = tf.constant(b)

    grad_in = np.ones(output.shape)
    grad = fc.backprop(grad_in, 2, 0)
    print('grad:', grad)

    output_t = tf.linalg.matmul(x_t, W_t) + b_t
    grad_t = tf.gradients([output_t], [x_t])
    print('grad_t:', sess.run(grad_t))


    # sigmoid
    # 100% works
    '''
    x_t = tf.constant(x)
    sigm_t = tf.math.sigmoid(x_t)

    sigm = fn.sigmoid().call(x)
    print('sigm:', sigm)
    print('sigm_t', sess.run(sigm_t))

    grad = fn.sigmoid().derivative(x)
    grad_t = tf.gradients([sigm_t], [x_t])

    print('grad:', grad)
    print('grad_t:', sess.run(grad_t))
    '''