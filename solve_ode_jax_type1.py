'''
f'' + 2*x*f = 0
'''

import tensorflow as tf
import jax.numpy as np
from jax import random
from jax import grad
from jax import vmap
from jax import jit
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1. / (1. + np.exp(-x))


def f(params, x):
    w0 = params[:10]
    b0 = params[10:20]
    w1 = params[20:30]
    b1 = params[30]
    x = sigmoid(x * w0 + b0)
    x = sigmoid(np.sum(x * w1) + b1)
    return x


@jit
def loss(params, inputs):
    eq = dfdx_vec(params, inputs) + 2. * inputs * f_vec(params, inputs)
    ic = f(params, 0.) - 1.  # initial condition
    return np.mean(eq ** 2) + ic ** 2


if __name__ == '__main__':
    key = random.PRNGKey(0)
    params = random.normal(key, shape=(31,))

    dfdx = grad(f, 1)  # １はf()の第2引数で微分することを要請している。
    inputs = np.linspace(-2., 2., 401)  # -2 <= x <= 2
    f_vec = vmap(f, (None, 0))
    dfdx_vec = vmap(dfdx, (None, 0))
    grad_loss = jit(grad(loss, 0))  # fの第一引数に対してlossを取りたいから0

    epochs = 1000
    learning_rate = 0.1
    momentum = 0.99
    velocity = 0.

    for epoch in range(epochs):
        if epoch % 100 == 0:
            print('epoch: {} loss: {}' .format(epoch, loss(params, inputs)))
        gradient = grad_loss(params + momentum * velocity, inputs)
        velocity = momentum * velocity - learning_rate * gradient  # NAG
        params += velocity

    plt.plot(inputs, np.exp(-inputs ** 2), label='exact')
    plt.plot(inputs, f_vec(params, inputs), label='approx')
    plt.legend()
    plt.show()
