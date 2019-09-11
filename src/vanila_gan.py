import collections
import numpy as np
import tensorflow as tf
# noinspection PyPep8Naming
import TensorFlow_utils as tf_utils
from tensorflow.contrib.layers import flatten

import utils as utils


class GAN(object):
    def __init__(self, sess, flags, image_size):
        self.sess = sess
        self.flags = flags
        self.image_size = image_size
        self.num_hiddens = 128
        self.out_func = tf.nn.sigmoid if self.flags.dataset == 'mnist' else tf.nn.tanh

        self._build_net()
        self._tensorboard()
        print("Initialized Vanilla GAN SUCCESS!")

    def _build_net(self):
        self.Y = tf.placeholder(tf.float32, shape=[None, *self.image_size], name='output')  # output data
        self.z = tf.placeholder(tf.float32, shape=[None, self.flags.z_dim], name='latent_vector')

        self.g_samples = self.generator(self.z)
        d_real, d_logit_real = self.discriminator(self.Y)
        d_fake, d_logit_fake = self.discriminator(self.g_samples, is_reuse=True)

        # discriminator loss
        d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=d_logit_real, labels=tf.ones_like(d_logit_real)))
        d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=d_logit_fake, labels=tf.zeros_like(d_logit_fake)))
        self.d_loss = d_loss_real + d_loss_fake

        # generator loss
        self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=d_logit_fake, labels=tf.ones_like(d_logit_fake)))

        d_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='d_')
        g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='g_')

        self.dis_optim = tf.train.AdamOptimizer(
            learning_rate=self.flags.learning_rate,
            beta1=self.flags.beta1).minimize(self.d_loss, var_list=d_vars)
        self.gen_optim = tf.train.AdamOptimizer(
            learning_rate=self.flags.learning_rate,
            beta1=self.flags.beta1).minimize(self.g_loss, var_list=g_vars)

    def _tensorboard(self):
        tf.summary.scalar('loss/d_loss', self.d_loss)
        tf.summary.scalar('loss/g_loss', self.g_loss)

        self.summary_op = tf.summary.merge_all()

    def generator(self, x_data, name='g_'):
        with tf.variable_scope(name):
            x_data = flatten(x_data)
            g0 = tf.nn.relu(tf_utils.linear(x_data, self.num_hiddens, name='fc1'), name='relu1')
            g1 = tf_utils.linear(g0, self.image_size[0] * self.image_size[1], name='fc2')

        return self.out_func(g1)

    def discriminator(self, y_data, name='d_', is_reuse=False):
        with tf.variable_scope(name, reuse=is_reuse):
            y_data = flatten(y_data)
            d0 = tf.nn.relu(tf_utils.linear(y_data, self.num_hiddens, name='fc1'))
            d1 = tf_utils.linear(d0, 1, name='fc2')

        return tf.nn.sigmoid(d1), d1

    def train_step(self, x_data, y_data):
        feed = {self.z: self.sample_z(num=self.flags.batch_size), self.Y: y_data}

        _, d_loss = self.sess.run([self.dis_optim, self.d_loss], feed_dict=feed)
        _, g_loss = self.sess.run([self.gen_optim, self.g_loss], feed_dict=feed)

        # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
        _, g_loss, summary = self.sess.run([self.gen_optim, self.g_loss, self.summary_op], feed_dict=feed)

        return [d_loss, g_loss], summary

    def test_step(self, x_data, y_data):
        return self.sample_imgs(x_data, y_data)


    def sample_imgs(self, x_data, y_data):
        g_feed = {self.z: self.sample_z(num=x_data.shape[0])}
        return [x_data, self.sess.run(self.g_samples, feed_dict=g_feed), y_data]

    def sample_z(self, num):
        return np.random.uniform(-1., 1., size=[num, self.flags.z_dim])

    def print_info(self, loss, iter_time):
        if np.mod(iter_time, self.flags.print_freq) == 0:
            ord_output = collections.OrderedDict([('cur_iter', iter_time), ('tar_iters', self.flags.iters),
                                                  ('batch_size', self.flags.batch_size),
                                                  ('d_loss', loss[0]), ('g_loss', loss[1]),
                                                  ('dataset', self.flags.dataset),
                                                  ('gpu_index', self.flags.gpu_index)])

            utils.print_metrics(iter_time, ord_output)
