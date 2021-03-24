'''
f'' + 2*x*f = 0
'''

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import numpy as np
import argparse
import tqdm
os.environ['KMP_DUPLICATE_LIB_OK']='True'


class SolnModel(tf.keras.models.Model):
    def __init__(self):
        super(SolnModel, self).__init__()
        self.d1 = tf.keras.layers.Dense(10, activation='sigmoid',\
                            kernel_initializer=tf.keras.initializers.RandomNormal(mean=0., stddev=1.))
        self.d2 = tf.keras.layers.Dense(1, activation = 'sigmoid', \
                            kernel_initializer=tf.keras.initializers.RandomNormal(mean=0., stddev=1.))

    def call(self, x):
        x = self.d1(x)
        x = self.d2(x)
        return x


def loss_obj(input):
    input = tf.Variable(input)
    with tf.GradientTape() as loss_tape:
        out = model(input)
    eq = loss_tape.gradient(out, input) + 2. * tf.matmul(out, tf.transpose(input))
    ic = tf.subtract(model(tf.zeros(shape=(1, 1))), tf.ones(shape=(1, 1)))
    loss = tf.reduce_mean(eq ** 2) + tf.reduce_mean(ic ** 2 )
    return loss


def train_step(input):
    with tf.GradientTape() as tape:
        output = model(input)
        loss = loss_obj(input)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    train_loss(loss)
    return loss


if __name__ == '__main__':
    hparams = argparse.Namespace()
    hparams.batch_size = 400
    hparams.epochs = 1000
    hparams.learning_rate = 1e-3
    hparams.momentum = 0.99
    hparams.velocity = 0.

    inputs = tf.reshape(tf.linspace(-2., 2., 401), shape=(-1, 1))
    train_ds = tf.data.Dataset.from_tensor_slices((inputs)).batch(hparams.batch_size).shuffle(400)
    test = tf.reshape(tf.linspace(-2., 2., 401), shape=(-1, 1))
    test_ds = tf.data.Dataset.from_tensor_slices((test))

    model = SolnModel()

    train_loss = tf.keras.metrics.Mean()
    lrs = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=hparams.learning_rate, decay_steps=500, decay_rate=0.98)
    optimizer = tf.keras.optimizers.SGD(learning_rate=lrs, momentum=hparams.momentum, decay=hparams.velocity, nesterov=True)
    # optimizer = tf.keras.optimizers.Nadam(learning_rate=hparams.learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-07)

    ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, net=model)
    manager = tf.train.CheckpointManager(ckpt, './tf_ckpts', max_to_keep=3)

    train_loss.reset_states()
    # train loop
    loss = 10**10
    for epoch in tqdm.tqdm(range(hparams.epochs)):
        if epoch % 100 == 0 and epoch != 0:
            print(epoch)
        for x in train_ds:
            ckpt.step.assign_add(1)
            cl = train_step(x)
            if cl < loss:
                loss = cl
                print(loss)
                manager.save()

    # test loop
    preds = []
    for x in test_ds:
        pred = model.predict(x)
        preds.append(pred)
    # train_loss.reset_states()
    preds = tf.squeeze(preds)
    print(preds[:10])

    plt.plot(test, np.exp(-test ** 2), label='exact')
    plt.plot(test, preds, label='approx')
    plt.legend()
    plt.show()