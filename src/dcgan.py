import collections
import numpy as np
import tensorflow as tf
# noinspection PyPep8Naming
import TensorFlow_utils as tf_utils
from tensorflow.contrib.layers import flatten

import utils as utils


class DCGAN(object):
    def __init__(self, sess, flags, image_size):
        self.sess = sess
        self.flags = flags
        self.image_size = image_size

        self._gen_train_ops, self._dis_train_ops = [], []
        self.gen_c = [1024, 512, 256, 128, 64, 64, self.image_size[2]]
        self.dis_c = [64, 128, 256, 512, 512, 512, 1]
        # self.out_fun = tf.nn.sigmoid if self.flags.dataset == 'mnist' else tf.nn.tanh

        self._build_net()
        self._tensorboard()
        print("Initialized DCGAN SUCCESS!")

    def _build_net(self):
        self.Y = tf.placeholder(tf.float32, shape=[None, *self.image_size], name='output')
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

        # Optimizer
        dis_op = tf.train.AdamOptimizer(learning_rate=self.flags.learning_rate, beta1=self.flags.beta1)\
            .minimize(self.d_loss, var_list=d_vars)
        dis_ops = [dis_op] + self._dis_train_ops
        self.dis_optim = tf.group(*dis_ops)

        gen_op = tf.train.AdamOptimizer(learning_rate=self.flags.learning_rate, beta1=self.flags.beta1)\
            .minimize(self.g_loss, var_list=g_vars)
        gen_ops = [gen_op] + self._gen_train_ops
        self.gen_optim = tf.group(*gen_ops)

    def _tensorboard(self):
        tf.summary.scalar('loss/d_loss', self.d_loss)
        tf.summary.scalar('loss/g_loss', self.g_loss)

        self.summary_op = tf.summary.merge_all()

    def generator(self, data, name='g_'):
        with tf.variable_scope(name):
            data_flatten = flatten(data)

            # 4 x 4
            h0_linear = tf_utils.linear(data_flatten, 4*4*self.gen_c[0], name='h0_linear')
            h0_reshape = tf.reshape(h0_linear, [tf.shape(h0_linear)[0], 4, 4, self.gen_c[0]])
            h0_batchnorm = tf_utils.batch_norm(h0_reshape, name='h0_batchnorm', _ops=self._gen_train_ops)
            h0_relu = tf.nn.relu(h0_batchnorm, name='h0_relu')

            # 8 x 8
            h1_deconv = tf_utils.deconv2d(h0_relu, self.gen_c[1], name='h1_deconv2d')
            h1_batchnorm = tf_utils.batch_norm(h1_deconv, name='h1_batchnorm', _ops=self._gen_train_ops)
            h1_relu = tf.nn.relu(h1_batchnorm, name='h1_relu')

            # 16 x 16
            h2_deconv = tf_utils.deconv2d(h1_relu, self.gen_c[2], name='h2_deconv2d')
            h2_batchnorm = tf_utils.batch_norm(h2_deconv, name='h2_batchnorm', _ops=self._gen_train_ops)
            h2_relu = tf.nn.relu(h2_batchnorm, name='h2_relu')

            # 32 x 32
            h3_deconv = tf_utils.deconv2d(h2_relu, self.gen_c[3], name='h3_deconv2d')
            h3_batchnorm = tf_utils.batch_norm(h3_deconv, name='h3_batchnorm', _ops=self._gen_train_ops)
            h3_relu = tf.nn.relu(h3_batchnorm, name='h3_relu')

            # 64 x 64
            h4_deconv = tf_utils.deconv2d(h3_relu, self.gen_c[4], name='h4_deconv2d')
            h4_batchnorm = tf_utils.batch_norm(h4_deconv, name='h4_batchnorm', _ops=self._gen_train_ops)
            h4_relu = tf.nn.relu(h4_batchnorm, name='h4_relu')

            # 128 x 128
            h5_deconv = tf_utils.deconv2d(h4_relu, self.gen_c[5], name='h5_deconv2d')
            h5_batchnorm = tf_utils.batch_norm(h5_deconv, name='h5_batchnorm', _ops=self._gen_train_ops)
            h5_relu = tf.nn.relu(h5_batchnorm, name='h5_relu')

            # 256 x 256
            h6_deconv = tf_utils.deconv2d(h5_relu, self.gen_c[6], name='h6_deconv2d')

            return tf.nn.tanh(h6_deconv)

    def discriminator(self, data, name='d_', is_reuse=False):
        with tf.variable_scope(name) as scope:
            if is_reuse is True:
                scope.reuse_variables()

            # 256 -> 128
            h0_conv = tf_utils.conv2d(data, self.dis_c[0], name='h0_conv2d')
            h0_lrelu = tf_utils.lrelu(h0_conv, name='h0_lrelu')

            # 128 -> 64
            h1_conv = tf_utils.conv2d(h0_lrelu, self.dis_c[1], name='h1_conv2d')
            h1_batchnorm = tf_utils.batch_norm(h1_conv, name='h1_batchnorm', _ops=self._dis_train_ops)
            h1_lrelu = tf_utils.lrelu(h1_batchnorm, name='h1_lrelu')

            # 64 -> 32
            h2_conv = tf_utils.conv2d(h1_lrelu, self.dis_c[2], name='h2_conv2d')
            h2_batchnorm = tf_utils.batch_norm(h2_conv, name='h2_batchnorm', _ops=self._dis_train_ops)
            h2_lrelu = tf_utils.lrelu(h2_batchnorm, name='h2_lrelu')

            # 32 -> 16
            h3_conv = tf_utils.conv2d(h2_lrelu, self.dis_c[3], name='h3_conv2d')
            h3_batchnorm = tf_utils.batch_norm(h3_conv, name='h3_batchnorm', _ops=self._dis_train_ops)
            h3_lrelu = tf_utils.lrelu(h3_batchnorm, name='h3_lrelu')

            # 16 -> 8
            h4_conv = tf_utils.conv2d(h3_lrelu, self.dis_c[4], name='h4_conv2d')
            h4_batchnorm = tf_utils.batch_norm(h4_conv, name='h4_batchnorm', _ops=self._dis_train_ops)
            h4_lrelu = tf_utils.lrelu(h4_batchnorm, name='h4_lrelu')

            # 8 -> 4
            h5_conv = tf_utils.conv2d(h4_lrelu, self.dis_c[5], name='h5_conv2d')
            h5_batchnorm = tf_utils.batch_norm(h5_conv, name='h5_batchnorm', _ops=self._dis_train_ops)
            h5_lrelu = tf_utils.lrelu(h5_batchnorm, name='h5_lrelu')

            h5_flatten = flatten(h5_lrelu)
            h6_linear = tf_utils.linear(h5_flatten, self.dis_c[6], name='h6_linear')

            return tf.nn.sigmoid(h6_linear), h6_linear

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
