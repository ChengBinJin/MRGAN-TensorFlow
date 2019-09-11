# import os
import collections
# import scipy.misc
import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.gridspec as gridspec
import tensorflow as tf
from tensorflow.contrib.layers import flatten

# noinspection PyPep8Naming
import TensorFlow_utils as tf_utils
import utils as utils


class Pix2Pix(object):
    def __init__(self, sess, flags, image_size):
        self.sess = sess
        self.flags = flags
        self.image_size = image_size

        self.L1_lamba = 100.0
        self._gen_train_ops, self._dis_train_ops = [], []
        self.gen_c = [64, 128, 256, 512, 512, 512, 512, 512,
                      512, 512, 512, 512, 256, 128, 64, self.image_size[2]]
        self.dis_c = [64, 128, 256, 512, 1]

        self._build_net()
        self._tensorboard()
        print("Initialized pix2pix SUCCESS!")

    def _build_net(self):
        self.X = tf.placeholder(tf.float32, shape=[None, *self.image_size], name='input')
        self.Y = tf.placeholder(tf.float32, shape=[None, *self.image_size], name='output')

        self.g_samples = self.generator(self.X)
        self.real_pair = tf.concat([self.X, self.Y], axis=3)
        self.fake_pair = tf.concat([self.X, self.g_samples], axis=3)

        d_real, d_logit_real = self.discriminator(self.real_pair)
        d_fake, d_logit_fake = self.discriminator(self.fake_pair, is_reuse=True)

        # discriminator loss
        self.d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=d_logit_real, labels=tf.ones_like(d_real)))
        self.d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=d_logit_fake, labels=tf.zeros_like(d_logit_fake)))
        self.d_loss = self.d_loss_real + self.d_loss_fake

        # generator loss
        gan_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=d_logit_fake, labels=tf.ones_like(d_logit_fake)))
        cond_loss = self.L1_lamba * tf.reduce_mean(tf.abs(self.Y - self.g_samples))
        self.g_loss = gan_loss + cond_loss

        d_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='d_')
        g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='g_')

        dis_op = tf.train.AdamOptimizer(learning_rate=self.flags.learning_rate, beta1=self.flags.beta1)\
            .minimize(self.d_loss, var_list=d_vars)
        dis_ops = [dis_op] + self._dis_train_ops
        self.dis_optim = tf.group(*dis_ops)

        gen_op = tf.train.AdamOptimizer(learning_rate=self.flags.learning_rate, beta1=self.flags.beta1)\
            .minimize(self.g_loss, var_list=g_vars)
        gen_ops = [gen_op] + self._gen_train_ops
        self.gen_optim = tf.group(*gen_ops)

    def _tensorboard(self):
        tf.summary.scalar("loss/D_dis", self.d_loss)
        tf.summary.scalar("loss/G_gen", self.g_loss)

        self.summary_op = tf.summary.merge_all()

    def generator(self, data, name='g_'):
        with tf.variable_scope(name):
            # 256 -> 128
            e0_conv2d = tf_utils.conv2d(data, self.gen_c[0], name='e0_conv2d')
            e0_lrelu = tf_utils.lrelu(e0_conv2d, name='e0_lrelu')

            # 128 -> 64
            e1_conv2d = tf_utils.conv2d(e0_lrelu, self.gen_c[1], name='e1_conv2d')
            e1_batchnorm = tf_utils.batch_norm(e1_conv2d, name='e1_batchnorm', _ops=self._gen_train_ops)
            e1_lrelu = tf_utils.lrelu(e1_batchnorm, name='e1_lrelu')

            # 64 -> 32
            e2_conv2d = tf_utils.conv2d(e1_lrelu, self.gen_c[2], name='e2_conv2d')
            e2_batchnorm = tf_utils.batch_norm(e2_conv2d, name='e2_batchnorm', _ops=self._gen_train_ops)
            e2_lrelu = tf_utils.lrelu(e2_batchnorm, name='e2_lrelu')

            # 32 -> 16
            e3_conv2d = tf_utils.conv2d(e2_lrelu, self.gen_c[3], name='e3_conv2d')
            e3_batchnorm = tf_utils.batch_norm(e3_conv2d, name='e3_batchnorm', _ops=self._gen_train_ops)
            e3_lrelu = tf_utils.lrelu(e3_batchnorm, name='e3_lrelu')

            # 16 -> 8
            e4_conv2d = tf_utils.conv2d(e3_lrelu, self.gen_c[4], name='e4_conv2d')
            e4_batchnorm = tf_utils.batch_norm(e4_conv2d, name='e4_batchnorm', _ops=self._gen_train_ops)
            e4_lrelu = tf_utils.lrelu(e4_batchnorm, name='e4_lrelu')

            # 8 -> 4
            e5_conv2d = tf_utils.conv2d(e4_lrelu, self.gen_c[5], name='e5_conv2d')
            e5_batchnorm = tf_utils.batch_norm(e5_conv2d, name='e5_batchnorm', _ops=self._gen_train_ops)
            e5_lrelu = tf_utils.lrelu(e5_batchnorm, name='e5_lrelu')

            # 4 -> 2
            e6_conv2d = tf_utils.conv2d(e5_lrelu, self.gen_c[6], name='e6_conv2d')
            e6_batchnorm = tf_utils.batch_norm(e6_conv2d, name='e6_batchnorm', _ops=self._gen_train_ops)
            e6_lrelu = tf_utils.lrelu(e6_batchnorm, name='e6_lrelu')

            # 2 -> 1
            e7_conv2d = tf_utils.conv2d(e6_lrelu, self.gen_c[7], name='e7_conv2d')
            e7_batchnorm = tf_utils.batch_norm(e7_conv2d, name='e7_batchnorm', _ops=self._gen_train_ops)
            e7_relu = tf.nn.relu(e7_batchnorm, name='e7_relu')

            # 1 -> 2
            d0_deconv = tf_utils.deconv2d(e7_relu, self.gen_c[8], name='d0_deconv2d')
            d0_batchnorm = tf_utils.batch_norm(d0_deconv, name='d0_batchnorm', _ops=self._gen_train_ops)
            d0_drop = tf.nn.dropout(d0_batchnorm, keep_prob=0.5, name='d0_dropout')
            d0_concat = tf.concat([d0_drop, e6_batchnorm], axis=3, name='d0_concat')
            d0_relu = tf.nn.relu(d0_concat, name='d0_relu')

            # 2 -> 4
            d1_deconv = tf_utils.deconv2d(d0_relu, self.gen_c[9], name='d1_deconv2d')
            d1_batchnorm = tf_utils.batch_norm(d1_deconv, name='d1_batchnorm', _ops=self._gen_train_ops)
            d1_drop = tf.nn.dropout(d1_batchnorm, keep_prob=0.5, name='d1_dropout')
            d1_concat = tf.concat([d1_drop, e5_batchnorm], axis=3, name='d1_concat')
            d1_relu = tf.nn.relu(d1_concat, name='d1_relu')

            # 4 -> 8
            d2_deconv = tf_utils.deconv2d(d1_relu, self.gen_c[10], name='d2_deconv2d')
            d2_batchnorm = tf_utils.batch_norm(d2_deconv, name='d2_batchnorm', _ops=self._gen_train_ops)
            d2_drop = tf.nn.dropout(d2_batchnorm, keep_prob=0.5, name='d2_dropout')
            d2_concat = tf.concat([d2_drop, e4_batchnorm], axis=3, name='d2_concat')
            d2_relu = tf.nn.relu(d2_concat, name='d2_relu')

            # 8 -> 16
            d3_deconv = tf_utils.deconv2d(d2_relu, self.gen_c[11], name='d3_deconv2d')
            d3_batchnorm = tf_utils.batch_norm(d3_deconv, name='d3_batchnorm', _ops=self._gen_train_ops)
            d3_concat = tf.concat([d3_batchnorm, e3_batchnorm], axis=3, name='d3_concat')
            d3_relu = tf.nn.relu(d3_concat, name='d3_relu')

            # 16 -> 32
            d4_deconv = tf_utils.deconv2d(d3_relu, self.gen_c[12], name='d4_deconv2d')
            d4_batchnorm = tf_utils.batch_norm(d4_deconv, name='d4_batchnorm', _ops=self._gen_train_ops)
            d4_concat = tf.concat([d4_batchnorm, e2_batchnorm], axis=3, name='d4_concat')
            d4_relu = tf.nn.relu(d4_concat, name='d4_relu')

            # 32 -> 64
            d5_deconv = tf_utils.deconv2d(d4_relu, self.gen_c[13], name='d5_deconv2d')
            d5_batchnorm = tf_utils.batch_norm(d5_deconv, name='d5_batchnorm', _ops=self._gen_train_ops)
            d5_concat = tf.concat([d5_batchnorm, e1_batchnorm], axis=3, name='d5_concat')
            d5_relu = tf.nn.relu(d5_concat, name='d5_relu')

            # 64 -> 128
            d6_deconv = tf_utils.deconv2d(d5_relu, self.gen_c[14], name='d6_deconv2d')
            d6_batchnorm = tf_utils.batch_norm(d6_deconv, name='d6_batchnorm', _ops=self._gen_train_ops)
            d6_concat = tf.concat([d6_batchnorm, e0_conv2d], axis=3, name='d6_concat')
            d6_relu = tf.nn.relu(d6_concat, name='d6_relu')

            # 128 -> 256
            d7_deconv = tf_utils.deconv2d(d6_relu, self.gen_c[15], name='d7_deconv2d')

            return tf.nn.tanh(d7_deconv)

    def discriminator(self, data, name='d_', is_reuse=False):
        with tf.variable_scope(name, reuse=is_reuse):
            # 256 -> 128
            h0_conv2d = tf_utils.conv2d(data, self.dis_c[0], name='h0_conv2d')
            h0_lrelu = tf_utils.lrelu(h0_conv2d, name='h0_lrelu')

            # 128 -> 64
            h1_conv2d = tf_utils.conv2d(h0_lrelu, self.dis_c[1], name='h1_conv2d')
            h1_batchnorm = tf_utils.batch_norm(h1_conv2d, name='h1_batchnorm', _ops=self._dis_train_ops)
            h1_lrelu = tf_utils.lrelu(h1_batchnorm, name='h1_lrelu')

            # 64 -> 32
            h2_conv2d = tf_utils.conv2d(h1_lrelu, self.dis_c[2], name='h2_conv2d')
            h2_batchnorm = tf_utils.batch_norm(h2_conv2d, name='h2_batchnorm', _ops=self._dis_train_ops)
            h2_lrelu = tf_utils.lrelu(h2_batchnorm, name='h2_lrelu')

            # 32 -> 32
            h3_conv2d = tf_utils.conv2d(h2_lrelu, self.dis_c[3], d_h=1, d_w=1, name='h3_conv2d')
            h3_batchnorm = tf_utils.batch_norm(h3_conv2d, name='h3_batchnorm', _ops=self._dis_train_ops)
            h3_lrelu = tf_utils.lrelu(h3_batchnorm, name='h3_lrelu')

            # linear
            h3_flatten = flatten(h3_lrelu)
            h4_linear = tf_utils.linear(h3_flatten, self.dis_c[4], name='h4_linear')

            return tf.nn.sigmoid(h4_linear), h4_linear

    def train_step(self, x_data, y_data):
        feed_dict = {self.X: x_data, self.Y: y_data}
        _, d_loss = self.sess.run([self.dis_optim, self.d_loss], feed_dict=feed_dict)
        _, g_loss = self.sess.run([self.gen_optim, self.g_loss], feed_dict=feed_dict)

        # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
        _, g_loss, summary = self.sess.run([self.gen_optim, self.g_loss, self.summary_op], feed_dict=feed_dict)

        return [d_loss, g_loss], summary

    def test_step(self, x_data, y_data):
        y_fakes = self.sess.run(self.g_samples, feed_dict={self.X: x_data})
        return [x_data, y_fakes, y_data]

    def test_only(self, x_data):
        y_fakes = self.sess.run(self.g_samples, feed_dict={self.X: x_data})
        return y_fakes

    def sample_imgs(self, x_data, y_data):
        y_fakes = self.sess.run(self.g_samples, feed_dict={self.X: x_data})
        return [x_data, y_fakes, y_data]

    def print_info(self, loss, iter_time):
        if np.mod(iter_time, self.flags.print_freq) == 0:
            ord_output = collections.OrderedDict([('cur_iter', iter_time), ('tar_iters', self.flags.iters),
                                                  ('batch_size', self.flags.batch_size),
                                                  ('D_loss', loss[0]), ('G_loss', loss[1]),
                                                  ('dataset', self.flags.dataset),
                                                  ('gpu_index', self.flags.gpu_index)])

            utils.print_metrics(iter_time, ord_output)

    # @staticmethod
    # def plots(imgs, iter_time, image_size, save_file):
    #     # parameters for plot size
    #     scale, margin = 0.015, 0.02
    #     n_cols, n_rows = len(imgs), imgs[0].shape[0]
    #     cell_size_h, cell_size_w = imgs[0].shape[1] * scale, imgs[0].shape[2] * scale
    #
    #     fig = plt.figure(figsize=(cell_size_w * n_cols, cell_size_h * n_rows))  # (column, row)
    #     gs = gridspec.GridSpec(n_rows, n_cols)  # (row, column)
    #     gs.update(wspace=margin, hspace=margin)
    #
    #     imgs = [utils.inverse_transform(imgs[idx]) for idx in range(len(imgs))]
    #
    #     # save more bigger image
    #     for col_index in range(n_cols):
    #         for row_index in range(n_rows):
    #             ax = plt.subplot(gs[row_index * n_cols + col_index])
    #             plt.axis('off')
    #             ax.set_xticklabels([])
    #             ax.set_yticklabels([])
    #             ax.set_aspect('equal')
    #             if image_size[2] == 3:
    #                 plt.imshow((imgs[col_index][row_index]).reshape(
    #                     image_size[0], image_size[1], image_size[2]), cmap='Greys_r')
    #             elif image_size[2] == 1:
    #                 plt.imshow((imgs[col_index][row_index]).reshape(
    #                     image_size[0], image_size[1]), cmap='Greys_r')
    #
    #     plt.savefig(os.path.join(save_file, '{}.png').format(str(iter_time).zfill(7)), bbox_inches='tight')
    #     plt.close(fig)
    #
    #     # just save prediction image
    #     pre_img = imgs[1][0, :, :, 0]
    #     scipy.misc.imsave(os.path.join(save_file, 'p_{}.png').format(str(iter_time).zfill(7)), pre_img)
    #
    # @staticmethod
    # def plots2(imgs, iter_time, image_size, save_file):
    #     new_img = np.zeros((image_size[0], 3*image_size[1]))
    #     imgs = [utils.inverse_transform(imgs[idx]) for idx in range(len(imgs))]
    #
    #     new_img[:, 0:image_size[1]] = imgs[0][0, :, :, 0]  # ct img
    #     new_img[:, 1 * image_size[1]:2 * image_size[1]] = imgs[1][0, :, :, 0]  # pred img
    #     new_img[:, 2 * image_size[1]:3 * image_size[1]] = imgs[2][0, :, :, 0]  # mri img
    #
    #     scipy.misc.imsave(os.path.join(save_file, '{}.png').format(str(iter_time).zfill(7)), new_img)
