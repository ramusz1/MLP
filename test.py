import tensorflow as tf
import numpy as np

import layer as lr
import functions as fn

W = np.arange(12).reshape(3,4).astype(np.float64)
x = np.arange(15).reshape(5,3).astype(np.float64)
b = np.arange(4).astype(np.float64)

y = np.arange(15).reshape(5,3).astype(np.float64)
y *= y


print('W', W)
print('x', x)
print('b', b)

fc = lr.FullyConnected(3,4)
fc.weight = W
fc.bias = b


with tf.Session() as sess:
    
    output = fc.forwardWithSave(x)

    W_t = tf.constant(W)
    x_t = tf.constant(x)
    b_t = tf.constant(b)
    y_t = tf.constant(y)

    # basic matmul backprop - works
    grad_in = np.ones(output.shape)
    grad = fc.backprop(grad_in, 0, 0)

    output_t = tf.linalg.matmul(x_t, W_t) + b_t
    grad_t = tf.gradients([output_t], [x_t])
    print('xGrad_t:', sess.run(grad_t))
    grad_t = tf.gradients([output_t], [W_t])
    print('wGrad_t:', sess.run(grad_t))
    grad_t = tf.gradients([output_t], [b_t])
    print('bGrad_t:', sess.run(grad_t))

    # sigmoid
    # 100% works

    x_t = tf.constant(x)
    sigm_t = tf.math.sigmoid(x_t)

    sigm = fn.sigmoid().call(x)
    print('sigm:', sigm)
    print('sigm_t', sess.run(sigm_t))

    grad = fn.sigmoid().derivative(x)
    grad_t = tf.gradients([sigm_t], [x_t])

    print('grad:', grad)
    print('grad_t:', sess.run(grad_t))

    # sigmoid + matmul combo - combination test
    sigmoid = lr.Activation(fn.sigmoid())

    output = fc.forwardWithSave(x)
    outputS = sigmoid.forwardWithSave(output)

    print('out', outputS)
    grad_in = np.ones(output.shape)
    grad_in = sigmoid.backprop(grad_in, 0, 0)
    grad = fc.backprop(grad_in, 0, 0)

    output_t = tf.math.sigmoid(tf.linalg.matmul(x_t, W_t) + b_t)
    grad_t = tf.gradients([output_t], [x_t])
    print('xGrad_t:', sess.run(grad_t))
    grad_t = tf.gradients([output_t], [W_t])
    print('wGrad_t:', sess.run(grad_t))
    grad_t = tf.gradients([output_t], [b_t])
    print('bGrad_t:', sess.run(grad_t))

    # mse gradients
    # does not work :|
    print('MMMMSSSSEEE')
    mse = lr.Loss(fn.MSE())
    loss = mse.forwardWithSave(x, y)

    x_t = tf.constant(x)
    y_t = tf.constant(y)

    loss_t = tf.losses.mean_squared_error(y_t, x_t)
    print('loss:', np.sum(loss))
    print('loss_t:', sess.run(loss_t))

    grad = mse.backprop(1, 0, 0)
    grad_t = tf.gradients([loss_t], [x_t])
    print('grad:', grad)
    print('grad_t:', sess.run(grad_t))

    # cross entropy with softmax gradients
    ce = lr.Loss(fn.crossEntropyWithLoss())
    ce.forwardWithSave(x, y)
