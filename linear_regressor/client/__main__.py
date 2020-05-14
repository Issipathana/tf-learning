import os

import numpy as np
import pylab
import tensorflow as tf


tf.reset_default_graph()
session = tf.Session()


class LinearRegressionModel():
    def __init__(self):
        with tf.name_scope("inputs"):
            self.x = tf.placeholder(tf.float32, (1,), name="x")
            self.y_known = tf.placeholder(tf.float32, (1,), name="y_known")
        with tf.name_scope("model"):
            self.m_hat = tf.Variable(tf.random_normal((1,)), name="m", dtype=tf.float32)
            self.b_hat = tf.Variable(tf.random_normal((1,)), name="b", dtype=tf.float32)
            self.y_hat = tf.multiply(self.m_hat, self.x) + self.b_hat


class Trainer():
    def __init__(self, model: LinearRegressionModel):
        self.model = model
        with tf.name_scope('training'):
            with tf.name_scope('loss'):
                self.loss = tf.reduce_mean(tf.square(model.y_hat - model.y_placeholder))
            with tf.name_scope('optimizer'):
                self.optimizer = tf.train.GradientDescentOptimizer()
                self.train_op = self.optimizer.minimize(self.loss)
        self.summary_op = None  #TODO
        self.train_op = None  #TODO

    def train_on_data(self, session, x, y, n_steps=1):
        for step in range(n_steps):
            _x = x[step: step + 1]
            _y = y[step: step + 1]
            session.run(
                [self.summary_op, self.train_op],
                feed_dict={
                    self.model.x_placeholder: _x,
                    self.model.y_placeholder: _y,
                }
            )


model = LinearRegressionModel()
model_trainer = Trainer(model)
TRAIN_STEPS = 100


def make_noisy_data(
        m: float = 0.1,
        b: float = 0.3,
        n_samples: int = 5,
        e_std: float = 0.01,
        random_state: np.random.RandomState = np.random.RandomState()):
    x = random_state.uniform(size=n_samples)
    e = random_state.normal(scale=e_std, size=len(x)).astype(np.float32)
    y = m * x + b + e
    return x, y


print(make_noisy_data())

x_train, y_train = make_noisy_data()
x_test, y_test = make_noisy_data()

pylab.plot(x_train, y_train, 'b.')
pylab.plot(x_test, y_test, 'g.')

x_data, y_data = make_noisy_data()
model_trainer.train_on_data(session, x_data, y_data, n_steps=TRAIN_STEPS)

