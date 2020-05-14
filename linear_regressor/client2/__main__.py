import numpy as np
import tensorflow as tf


N_TRAIN_STEPS = 11


with tf.name_scope("input"):
    y = tf.placeholder(dtype=tf.float32, shape=(1,))
    x = tf.placeholder(dtype=tf.float32, shape=(1,))
with tf.name_scope("model"):
    b_hat = tf.Variable(dtype=tf.float32, initial_value=0)
    b_0_hat = tf.Variable(dtype=tf.float32, initial_value=0)
    y_hat = tf.multiply(b_hat, x) + b_0_hat
with tf.name_scope("train"):
    loss = (y - y_hat) ** 2
optimizer = tf.train.AdamOptimizer()
train_op = optimizer.minimize(loss)


class TrueModel(object):
    def __init__(self, b_hat=4.0, b_0_hat=7.0, noise_sd=1.0):
        self.b_hat = np.float32(b_hat)
        self.b_0_hat = np.float32(b_0_hat)
        self.noise_sd = np.float32(noise_sd)

    def sample(self, n_samples=1, random_state=None):
        if random_state is None:
            random_state = np.random.RandomState()
        x = random_state.uniform(size=(n_samples,)).astype(np.float32)
        assert x.shape == (n_samples,), x.shape
        noise = random_state.normal(scale=self.noise_sd, size=(n_samples,)).astype(np.float32)
        y = self.b_hat * x + self.b_0_hat + noise
        assert y.shape == (n_samples,), y.shape
        assert x.dtype == np.float32, x.dtype
        assert y.dtype == np.float32, y.dtype
        return x, y


true_model = TrueModel(b_hat=4.0, b_0_hat=7.0, noise_sd=1.0)

operations = [train_op]
random_state = np.random.RandomState(seed=0)

with tf.Session() as session:
    tf.global_variables_initializer()
    for train_step in range(N_TRAIN_STEPS):
        x_feed, y_feed = true_model.sample(n_samples=1)
        session.run(operations, feed_dict={x: x_feed, y: y_feed})
        print(b_hat)
