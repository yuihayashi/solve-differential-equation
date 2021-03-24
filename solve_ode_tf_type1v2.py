'''
f'' + 2*x*f = 0
'''
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import numpy as np
import argparse
import tqdm


os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class SolnModel(tf.keras.models.Model):
    def __init__(self):
        super(SolnModel, self).__init__()
        self.d1 = tf.keras.layers.Dense(10, activation='sigmoid',
                                        kernel_initializer=tf.keras.initializers.RandomNormal(mean=0., stddev=1.))
        self.d2 = tf.keras.layers.Dense(1, activation='sigmoid',
                                        kernel_initializer=tf.keras.initializers.RandomNormal(mean=0., stddev=1.))

    def call(self, x):
        x = self.d1(x)
        x = self.d2(x)
        return x


class TrainSolnModel():
    def __init__(self, train_gen, model, optimizer, train_loss):
        self.train_gen = train_gen
        self.model = model
        self.optimizer = optimizer
        self.train_loss = train_loss

    def train_step(self, input):
        with tf.GradientTape() as tape:
            output = self.model(input)
            loss = self.loss_obj(output)
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        self.train_loss(loss)
        return loss

    def loss_obj(self, input):
        input = tf.Variable(input)
        with tf.GradientTape() as loss_tape:
            out = self.model(input)
        eq = loss_tape.gradient(out, input) + 2. * tf.matmul(out, tf.transpose(input))
        ic = tf.subtract(self.model(tf.zeros(shape=(1, 1))), tf.ones(shape=(1, 1)))
        loss = tf.reduce_mean(eq**2) + tf.reduce_mean(ic**2)
        return loss

    def fit(self):
        ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=self.optimizer, net=self.model)
        manager = tf.train.CheckpointManager(ckpt, hparams.save_model_path, max_to_keep=3)
        self.train_loss.reset_states()
        # train loop
        loss = 10**10
        for epoch in tqdm.tqdm(range(hparams.epochs)):
            if epoch % 100 == 0 and epoch != 0:
                print(epoch)
            for x in self.train_gen:
                ckpt.step.assign_add(1)
                cl = self.train_step(x)
                if cl < loss:
                    loss = cl
                    print(loss)
                    manager.save()


if __name__ == '__main__':
    hparams = argparse.Namespace()
    hparams.batch_size = 4
    hparams.epochs = 10000
    hparams.learning_rate = 1e-3
    hparams.momentum = 0.9
    hparams.velocity = 0.
    hparams.save_model_path = './tf_ckpts'

    inputs = tf.reshape(tf.linspace(-2., 2., 401), shape=(-1, 1))
    train_ds = tf.data.Dataset.from_tensor_slices((inputs)).batch(hparams.batch_size).shuffle(400)
    test = tf.reshape(tf.linspace(-2., 2., 401), shape=(-1, 1))
    test_ds = tf.data.Dataset.from_tensor_slices((test))

    model = SolnModel()

    train_loss = tf.keras.metrics.Mean()
    lrs = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=hparams.learning_rate, decay_steps=500, decay_rate=0.98)
    optimizer = tf.keras.optimizers.SGD(learning_rate=lrs, momentum=hparams.momentum, decay=hparams.velocity, nesterov=True)

    # train loop
    train_model = TrainSolnModel(train_gen=train_ds, model=model, optimizer=optimizer, train_loss=train_loss)
    train_model.fit()

    # test loop
    preds = []
    for x in test_ds:
        pred = model.predict(x)
        preds.append(pred)

    # visualize
    preds = tf.squeeze(preds)
    plt.plot(test, np.exp(-test**2), label='exact')
    plt.plot(test, preds, label='approx')
    plt.legend()
    plt.show()
