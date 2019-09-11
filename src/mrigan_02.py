# import os
import collections
import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.gridspec as gridspec
import tensorflow as tf

# import cv2
# import scipy.misc

# noinspection PyPep8Naming
import TensorFlow_utils as tf_utils
import utils as utils
from reader import Reader


# noinspection PyPep8Naming
class MRIGAN_02(object):
    def __init__(self, sess, flags, image_size, data_path):
        self.sess = sess
        self.flags = flags
        self.image_size = image_size
        self.x_path, self.y_path = data_path[0], data_path[1]

        # True: use lsgan (mean squared error)
        # False: use cross entropy loss
        self.use_lsgan = True
        self.use_sigmoid = not self.use_lsgan
        # [instance|batch] use instance norm or batch norm, default: instance
        self.norm = 'instane'
        self.lambda1, self.lambda2 = 10.0, 10.0  # unpaired learning
        self.l1_lamba = 100.0  # paired learning
        self.ngf, self.ndf = 64, 64
        self.real_label = 0.9
        self.start_dcay_step = 100000
        self.decay_steps = 200000
        self.eps = 1e-12

        self._G_gen_train_ops, self._F_gen_train_ops = [], []
        self._Dy_dis_train_ops, self._Dx_dis_train_ops = [], []

        self._build_net()
        self._tensorboard()

    def _build_net(self):
        self.mae_record_placeholder = tf.placeholder(tf.float32, name='mae_record_placeholder')
        self.mae_record = tf.Variable(256., trainable=False, dtype=tf.float32, name='mae_record')
        self.mae_record_assign_op = self.mae_record.assign(self.mae_record_placeholder)

        # tfph: TensorFlow PlaceHolder
        self.x_test_tfph = tf.placeholder(tf.float32, shape=[None, *self.image_size], name='x_test_tfph')
        self.y_test_tfph = tf.placeholder(tf.float32, shape=[None, *self.image_size], name='y_test_tfph')
        self.fake_x_tfph = tf.placeholder(tf.float32, shape=[None, *self.image_size], name='fake_x_tfph')
        self.fake_y_tfph = tf.placeholder(tf.float32, shape=[None, *self.image_size], name='fake_y_tfph')

        self.G_gen = Generator(name='G', ngf=self.ngf, norm=self.norm, image_size=self.image_size,
                               _ops=self._G_gen_train_ops)
        self.Dy_dis = Discriminator(name='Dy', ndf=self.ndf, norm=self.norm, _ops=self._Dy_dis_train_ops,
                                    use_sigmoid=self.use_sigmoid)
        self.F_gen = Generator(name='F', ngf=self.ngf, norm=self.norm, image_size=self.image_size,
                               _ops=self._F_gen_train_ops)
        self.Dx_dis = Discriminator(name='Dx', ndf=self.ndf, norm=self.norm, _ops=self._Dx_dis_train_ops,
                                    use_sigmoid=self.use_sigmoid)

        x_reader = Reader(self.x_path, name='X', image_size=self.image_size, batch_size=self.flags.batch_size)
        y_reader = Reader(self.y_path, name='Y', image_size=self.image_size, batch_size=self.flags.batch_size)
        self.x_imgs = x_reader.feed()
        self.y_imgs = y_reader.feed()

        self.fake_x_pool_obj = utils.ImagePool(pool_size=50)
        self.fake_y_pool_obj = utils.ImagePool(pool_size=50)

        self._unpair_net()    # idea from cyclegan
        self._pair_net()      # ide from pix2pix

        # Optimizers
        # G generator for unpaired data
        G_op_unpair = self.optimizer(loss=self.G_loss_unpair, variables=self.G_gen.variables, name='Adam_G_unpair')
        G_ops_unpair = [G_op_unpair] + self._G_gen_train_ops
        G_optim_unpair = tf.group(*G_ops_unpair)

        # G generator for paired data
        G_op_pair = self.optimizer(loss=self.G_loss_pair, variables=self.G_gen.variables, name='Adam_G_pair')
        G_ops_pair = [G_op_pair] + self._G_gen_train_ops
        self.G_optim_pair = tf.group(*G_ops_pair)

        # Dy discriminator for unpaired data
        Dy_op_unpair = self.optimizer(loss=self.Dy_dis_loss_unpair,
                                      variables=[self.Dy_dis.share_variables, self.Dy_dis.unpair_variables],
                                      name='Adam_Dy_unpair')
        Dy_ops_unpair = [Dy_op_unpair] + self._Dy_dis_train_ops
        Dy_optim_unpair = tf.group(*Dy_ops_unpair)

        # Dy discriminator for paired data
        Dy_op_pair = self.optimizer(loss=self.Dy_dis_loss_pair,
                                    variables=[self.Dy_dis.share_variables, self.Dy_dis.pair_variables],
                                    name='Adam_Dy_pair')
        Dy_ops_pair = [Dy_op_pair] + self._Dy_dis_train_ops
        self.Dy_optim_pair = tf.group(*Dy_ops_pair)

        # F generator for unpaired data
        F_op_unpair = self.optimizer(loss=self.F_loss_unpair, variables=self.F_gen.variables, name='Adam_F_unpair')
        F_ops_unpair = [F_op_unpair] + self._F_gen_train_ops
        F_optim_unpair = tf.group(*F_ops_unpair)

        # F generator for paired data
        F_op_pair = self.optimizer(loss=self.F_loss_pair, variables=self.F_gen.variables, name='Adam_F_pair')
        F_ops_pair = [F_op_pair] + self._F_gen_train_ops
        self.F_optim_pair = tf.group(*F_ops_pair)

        # Dx discriminator for unpaired data
        Dx_op_unpair = self.optimizer(loss=self.Dx_dis_loss_unpair,
                                      variables=[self.Dx_dis.share_variables, self.Dx_dis.unpair_variables],
                                      name='Adam_Dx_unpair')
        Dx_ops_unpair = [Dx_op_unpair] + self._Dx_dis_train_ops
        Dx_optim_unpair = tf.group(*Dx_ops_unpair)

        # Dx discriminator for paired data
        Dx_op_pair = self.optimizer(loss=self.Dx_dis_loss_pair,
                                    variables=[self.Dx_dis.share_variables, self.Dx_dis.pair_variables],
                                    name='Adam_Dx_pair')
        Dx_ops_pair = [Dx_op_pair] + self._Dx_dis_train_ops
        self.Dx_optim_pair = tf.group(*Dx_ops_pair)

        self.optims_unpair = tf.group([G_optim_unpair, Dy_optim_unpair, F_optim_unpair, Dx_optim_unpair])
        self.optims_pair = tf.group([self.G_optim_pair, self.Dy_optim_pair, self.F_optim_pair, self.Dx_optim_pair])
        self.loss_collections = [self.G_loss_unpair, self.Dy_dis_loss_unpair,
                                 self.F_loss_unpair, self.Dx_dis_loss_unpair,
                                 self.G_loss_pair, self.Dy_dis_loss_pair,
                                 self.F_loss_pair, self.Dx_dis_loss_pair]

        # with tf.control_dependencies([G_optim_pair, Dy_optim_pair, F_optim_pair, Dx_optim_pair]):
        #     self.optims_pair = tf.no_op(name='optimizers')

    def _unpair_net(self):
        # Cycle consistency loss
        cycle_loss = self.cycle_consistency_loss(self.x_imgs, self.y_imgs)

        # X -> Y
        self.fake_y_imgs = self.G_gen(self.x_imgs)
        self.G_gen_loss_unpair = self.generator_loss_unpair(self.Dy_dis, self.fake_y_imgs, use_lsgan=self.use_lsgan)
        self.G_loss_unpair = self.G_gen_loss_unpair + cycle_loss
        self.Dy_dis_loss_unpair = self.discriminator_loss_unpair(self.Dy_dis, self.y_imgs, self.fake_y_tfph,
                                                                 use_lsgan=self.use_lsgan)

        # Y -> X
        self.fake_x_imgs = self.F_gen(self.y_imgs)
        self.F_gen_loss_unpair = self.generator_loss_unpair(self.Dx_dis, self.fake_x_imgs, use_lsgan=self.use_lsgan)
        self.F_loss_unpair = self.F_gen_loss_unpair + cycle_loss
        self.Dx_dis_loss_unpair = self.discriminator_loss_unpair(self.Dx_dis, self.x_imgs, self.fake_x_tfph,
                                                                 use_lsgan=self.use_lsgan)

    def _pair_net(self):
        self.fake_y_sample = self.G_gen(self.x_test_tfph)
        self.recon_x_sample = self.F_gen(self.G_gen(self.x_test_tfph))
        self.real_xy_pair = tf.concat([self.x_test_tfph, self.y_test_tfph], axis=3)
        self.fake_xy_pair = tf.concat([self.x_test_tfph, self.fake_y_sample], axis=3)

        self.fake_x_sample = self.F_gen(self.y_test_tfph)
        self.real_yx_pair = tf.concat([self.y_test_tfph, self.x_test_tfph], axis=3)
        self.fake_yx_pair = tf.concat([self.y_test_tfph, self.fake_x_sample], axis=3)

        # discriminator loss
        self.Dy_dis_loss_pair, dy_logit_fake = self.discriminator_loss_pair(
            self.Dy_dis, self.real_xy_pair, self.fake_xy_pair)
        self.Dx_dis_loss_pair, dx_logit_fake = self.discriminator_loss_pair(
            self.Dx_dis, self.real_yx_pair, self.fake_yx_pair)

        # generator loss
        self.G_gen_loss_pair = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=dy_logit_fake, labels=tf.ones_like(dy_logit_fake)))
        self.F_gen_loss_pair = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=dx_logit_fake, labels=tf.ones_like(dx_logit_fake)))

        self.G_cond_loss = self.l1_lamba * tf.reduce_mean(tf.abs(self.y_test_tfph - self.fake_y_sample))
        self.F_cond_loss = self.l1_lamba * tf.reduce_mean(tf.abs(self.x_test_tfph - self.fake_x_sample))

        self.G_loss_pair = self.G_gen_loss_pair + self.G_cond_loss
        self.F_loss_pair = self.F_gen_loss_pair + self.F_cond_loss

    def optimizer(self, loss, variables, name='Adam'):
        global_step = tf.Variable(0, trainable=False)
        starter_learning_rate = self.flags.learning_rate
        end_learning_rate = 0.
        start_decay_step = self.start_dcay_step
        decay_steps = self.decay_steps

        learning_rate = (tf.where(tf.greater_equal(global_step, start_decay_step),
                                  tf.train.polynomial_decay(starter_learning_rate,
                                                            global_step - start_decay_step,
                                                            decay_steps, end_learning_rate, power=1.0),
                                  starter_learning_rate))
        tf.summary.scalar('learning_rate/{}'.format(name), learning_rate)

        learn_step = tf.train.AdamOptimizer(learning_rate, beta1=self.flags.beta1, name=name).\
            minimize(loss, global_step=global_step, var_list=variables)

        return learn_step

    def cycle_consistency_loss(self, x_imgs, y_imgs):
        forward_loss = tf.reduce_mean(tf.abs(self.F_gen(self.G_gen(x_imgs)) - x_imgs))
        backward_loss = tf.reduce_mean(tf.abs(self.G_gen(self.F_gen(y_imgs)) - y_imgs))
        loss = self.lambda1 * forward_loss + self.lambda2 * backward_loss
        return loss

    def generator_loss_unpair(self, dis_obj, fake_img, use_lsgan=True):
        if use_lsgan:
            # use mean squared error
            # loss = 0.5 * tf.reduce_mean(tf.squared_difference(dis_obj(fake_img), self.real_label))
            loss = tf.reduce_mean(tf.squared_difference(dis_obj(fake_img), self.real_label))
        else:
            # heuristic, non-saturating loss (I don't understand here!)
            # loss = -tf.reduce_mean(tf.log(dis_obj(fake_img) + self.eps)) / 2.  (???)
            loss = -tf.reduce_mean(tf.log(dis_obj(fake_img) + self.eps))
        return loss

    def discriminator_loss_unpair(self, dis_obj, real_img, fake_img, use_lsgan=True):
        if use_lsgan:
            # use mean squared error
            error_real = tf.reduce_mean(tf.squared_difference(dis_obj(real_img), self.real_label))
            error_fake = tf.reduce_mean(tf.square(dis_obj(fake_img)))
        else:
            # use cross entropy
            error_real = -tf.reduce_mean(tf.log(dis_obj(real_img) + self.eps))
            error_fake = -tf.reduce_mean(tf.log(1. - dis_obj(fake_img) + self.eps))

        # loss = 0.5 * (error_real + error_fake)
        loss = error_real + error_fake
        return loss

    @staticmethod
    def discriminator_loss_pair(dis_obj, real_img, fake_img):
        d_logit_real, d_logit_fake = dis_obj(real_img), dis_obj(fake_img)

        d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=d_logit_real, labels=tf.ones_like(d_logit_real)))
        d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=d_logit_fake, labels=tf.zeros_like(d_logit_fake)))

        d_loss = d_loss_real + d_loss_fake

        return d_loss, d_logit_fake

    def _tensorboard(self):
        tf.summary.histogram('Dy/real_unpaired', self.Dy_dis(self.y_imgs))
        tf.summary.histogram('Dy/fake_unpaired', self.Dy_dis(self.fake_y_imgs))
        tf.summary.histogram('Dx/real_unpaired', self.Dx_dis(self.x_imgs))
        tf.summary.histogram('Dx/fake_unpaired', self.Dx_dis(self.fake_x_imgs))

        tf.summary.histogram('Dy/real_paired', self.Dy_dis(self.real_xy_pair))
        tf.summary.histogram('Dy/fake_paired', self.Dy_dis(self.fake_xy_pair))
        tf.summary.histogram('Dx/real_paired', self.Dx_dis(self.real_yx_pair))
        tf.summary.histogram('Dx/fake_paired', self.Dx_dis(self.fake_yx_pair))

        tf.summary.scalar('loss/G_gen_unpaired', self.G_gen_loss_unpair)
        tf.summary.scalar('loss/Dy_dis_unpaired', self.Dy_dis_loss_unpair)
        tf.summary.scalar('loss/F_gen_unpaired', self.F_gen_loss_unpair)
        tf.summary.scalar('loss/Dx_dis_unpaired', self.Dx_dis_loss_unpair)

        tf.summary.scalar('loss/G_gen_paired', self.G_gen_loss_pair)
        tf.summary.scalar('loss/Dy_dis_paired', self.Dy_dis_loss_pair)
        tf.summary.scalar('loss/F_gen_paired', self.F_gen_loss_pair)
        tf.summary.scalar('loss/Dx_dis_paired', self.Dx_dis_loss_pair)

        # tf.summary.image('X/input', tf_utils.batch_convert2int(self.x_imgs))
        # tf.summary.image('X/generated_Y', tf_utils.batch_convert2int(self.G_gen(self.x_imgs)))
        # tf.summary.image('X/reconstruction', tf_utils.batch_convert2int(self.F_gen(self.G_gen(self.x_imgs))))
        # tf.summary.image('Y/input', tf_utils.batch_convert2int(self.y_imgs))
        # tf.summary.image('Y/generated_X', tf_utils.batch_convert2int(self.F_gen(self.y_imgs)))
        # tf.summary.image('Y/reconstruction', tf_utils.batch_convert2int(self.G_gen(self.F_gen(self.y_imgs))))

        self.summary_op = tf.summary.merge_all()

    def train_step(self, x_imgs=None, y_imgs=None, iter_time=0):
        # Unsupervised loss
        fake_y_val, fake_x_val, x_val, y_val = self.sess.run([self.fake_y_imgs, self.fake_x_imgs,
                                                              self.x_imgs, self.y_imgs])

        # Unsuperivsed learning
        _ = self.sess.run([self.optims_unpair], feed_dict={self.fake_x_tfph: self.fake_x_pool_obj.query(fake_x_val),
                                                           self.fake_y_tfph: self.fake_y_pool_obj.query(fake_y_val)})

        # Supervised learning: discriminator
        _ = self.sess.run([self.optims_pair], feed_dict={self.x_test_tfph: x_imgs, self.y_test_tfph: y_imgs})

        # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
        _, _ = self.sess.run([self.G_optim_pair, self.F_optim_pair],
                             feed_dict={self.x_test_tfph: x_imgs, self.y_test_tfph: y_imgs})

        feed_dict = {self.fake_x_tfph: self.fake_x_pool_obj.query(fake_x_val),
                     self.fake_y_tfph: self.fake_y_pool_obj.query(fake_y_val),
                     self.x_test_tfph: x_imgs, self.y_test_tfph: y_imgs}
        loss, summary = self.sess.run([self.loss_collections, self.summary_op], feed_dict=feed_dict)

        return loss, summary

    def sample_imgs(self, x_imgs=None, y_imgs=None):
        x_val, y_val = self.sess.run([self.x_imgs, self.y_imgs])
        fake_y, fake_x = self.sess.run([self.fake_y_sample, self.fake_x_sample],
                                       feed_dict={self.x_test_tfph: x_val, self.y_test_tfph: y_val})

        return [x_val, fake_y, y_val, fake_x]

    def test_step(self, img, mode='XtoY', is_recon=True):
        if mode == 'XtoY':
            if is_recon:
                fake_y, recon_x = self.sess.run([self.fake_y_sample, self.recon_x_sample],
                                                feed_dict={self.x_test_tfph: img})
                return [img, fake_y, recon_x]
            elif not is_recon:
                fake_y = self.sess.run(self.fake_y_sample, feed_dict={self.x_test_tfph: img})
                return [img, fake_y]

        elif mode == 'YtoX':
            fake_x = self.sess.run(self.fake_x_sample, feed_dict={self.y_test_tfph: img})
            return [img, fake_x]
        else:
            raise NotImplementedError

    def print_info(self, loss, iter_time, epoch_time):
        if np.mod(iter_time, self.flags.print_freq) == 0:
            ord_output = collections.OrderedDict([('cur_epoch', epoch_time), ('tar_Epoch', self.flags.epoch),
                                                  ('batch_size', self.flags.batch_size),
                                                  ('G_loss_unpair', loss[0]), ('Dy_loss_unpair', loss[1]),
                                                  ('F_loss_unpair', loss[2]), ('Dx_loss_unpair', loss[3]),
                                                  ('G_loss_pair', loss[4]), ('Dy_loss_pair', loss[5]),
                                                  ('F_loss_pair', loss[6]), ('Dx_loss_pair', loss[7]),
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

    def mae_record_assign(self, mae):
        self.sess.run(self.mae_record_assign_op, feed_dict={self.mae_record_placeholder: mae})


class Generator(object):
    def __init__(self, name=None, ngf=64, norm='instance', image_size=(128, 256, 3), _ops=None):
        self.name = name
        self.ngf = ngf
        self.norm = norm
        self.image_size = image_size
        self._ops = _ops
        self.reuse = False

    def __call__(self, x):
        with tf.variable_scope(self.name, reuse=self.reuse):
            tf_utils.print_activations(x)

            # (N, H, W, C) -> (N, H, W, 64)
            conv1 = tf_utils.padding2d(x, p_h=3, p_w=3, pad_type='REFLECT', name='conv1_padding')
            conv1 = tf_utils.conv2d(conv1, self.ngf, k_h=7, k_w=7, d_h=1, d_w=1, padding='VALID',
                                    name='conv1_conv')
            conv1 = tf_utils.norm(conv1, _type='instance', _ops=self._ops, name='conv1_norm')
            conv1 = tf_utils.relu(conv1, name='conv1_relu', is_print=True)

            # (N, H, W, 64)  -> (N, H/2, W/2, 128)
            conv2 = tf_utils.conv2d(conv1, 2*self.ngf, k_h=3, k_w=3, d_h=2, d_w=2, padding='SAME',
                                    name='conv2_conv')
            conv2 = tf_utils.norm(conv2, _type='instance', _ops=self._ops, name='conv2_norm',)
            conv2 = tf_utils.relu(conv2, name='conv2_relu', is_print=True)

            # (N, H/2, W/2, 128) -> (N, H/4, W/4, 256)
            conv3 = tf_utils.conv2d(conv2, 4*self.ngf, k_h=3, k_w=3, d_h=2, d_w=2, padding='SAME',
                                    name='conv3_conv')
            conv3 = tf_utils.norm(conv3, _type='instance', _ops=self._ops, name='conv3_norm',)
            conv3 = tf_utils.relu(conv3, name='conv3_relu', is_print=True)

            # (N, H/4, W/4, 256) -> (N, H/4, W/4, 256)
            if (self.image_size[0] <= 128) and (self.image_size[1] <= 128):
                # use 6 residual blocks for 128x128 images
                res_out = tf_utils.n_res_blocks(conv3, num_blocks=6, is_print=True)
            else:
                # use 9 blocks for higher resolution
                res_out = tf_utils.n_res_blocks(conv3, num_blocks=9, is_print=True)

            # (N, H/4, W/4, 256) -> (N, H/2, W/2, 128)
            conv4 = tf_utils.deconv2d(res_out, 2*self.ngf, name='conv4_deconv2d')
            conv4 = tf_utils.norm(conv4, _type='instance', _ops=self._ops, name='conv4_norm')
            conv4 = tf_utils.relu(conv4, name='conv4_relu', is_print=True)

            # (N, H/2, W/2, 128) -> (N, H, W, 64)
            conv5 = tf_utils.deconv2d(conv4, self.ngf, name='conv5_deconv2d')
            conv5 = tf_utils.norm(conv5, _type='instance', _ops=self._ops, name='conv5_norm')
            conv5 = tf_utils.relu(conv5, name='conv5_relu', is_print=True)

            # (N, H, W, 64) -> (N, H, W, 3)
            conv6 = tf_utils.padding2d(conv5, p_h=3, p_w=3, pad_type='REFLECT', name='output_padding')
            conv6 = tf_utils.conv2d(conv6, self.image_size[2], k_h=7, k_w=7, d_h=1, d_w=1,
                                    padding='VALID', name='output_conv')
            output = tf_utils.tanh(conv6, name='output_tanh', is_print=True)

            # set reuse=True for next call
            self.reuse = True
            self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

            return output


class Discriminator(object):
    def __init__(self, name='', ndf=64, norm='instance', _ops=None, use_sigmoid=False):
        self.name = name
        self.ndf = ndf
        self.norm = norm
        self._ops = _ops
        self.reuse, self.reuse_extra_1, self.reuse_extra_2 = False, False, False
        self.use_sigmoid = use_sigmoid

    def __call__(self, x):
        output = None
        input_shape = x.get_shape().as_list()
        print('input_shape: {}'.format(input_shape))

        if input_shape[3] == 1:  # unpaired image
            with tf.variable_scope(self.name+'_unpair', reuse=self.reuse_extra_1):
                # Extra-head network
                # (N, H, W, 1) -> (N, H/2, W/2, 64)
                conv01 = tf_utils.conv2d(x, self.ndf, k_h=4, k_w=4, d_h=2, d_w=2, padding='SAME', name='conv01_conv_c1')
                conv01 = tf_utils.lrelu(conv01, name='conv01_lrelu_c1', is_print=True)

            # Share network
            share_out = self.share_net(conv01, name=self.name)

            with tf.variable_scope(self.name + '_unpair', reuse=self.reuse_extra_1):
                # Extra-tail network
                # # (N, H/16, W/16, 512) -> (N, H/16, W/16, 512)
                # conv02 = tf_utils.conv2d(share_out, 8 * self.ndf, k_h=4, k_w=4, d_h=1, d_w=1, padding='SAME',
                #                          name='conv02_conv_c1')
                # conv02 = tf_utils.norm(conv02, _type='instance', _ops=self._ops, name='conv02_norm_c1')
                # conv02 = tf_utils.lrelu(conv02, name='conv02_lrelu_c1', is_print=True)

                # (N, H/16, W/16, 512) -> (N, H/16, W/16, 1)
                conv02 = tf_utils.conv2d(share_out, 1, k_h=4, k_w=4, d_h=1, d_w=1, padding='SAME',
                                         name='conv02_conv_c1', is_print=True)

                if self.use_sigmoid:
                    output = tf_utils.sigmoid(conv02, name='output_sigmoid_c1', is_print=True)
                else:
                    output = tf_utils.identity(conv02, name='output_without_sigmoid_c1', is_print=True)

                # set reuse=True for next call
                self.reuse_extra_1 = True

        elif input_shape[3] == 2:  # paired image
            with tf.variable_scope(self.name+'_pair', reuse=self.reuse_extra_2):
                # Extra-head network
                # (N, H, W, 1) -> (N, H/2, W/2, 64)
                conv01 = tf_utils.conv2d(x, self.ndf, k_h=4, k_w=4, d_h=2, d_w=2, padding='SAME', name='conv01_conv_c2')
                conv01 = tf_utils.lrelu(conv01, name='conv01_lrelu_c2', is_print=True)

            # Share network
            share_out = self.share_net(conv01, name=self.name)

            with tf.variable_scope(self.name + '_pair', reuse=self.reuse_extra_2):
                # Extra-tail network
                # # (N, H/16, W/16, 512) -> (N, H/16, W/16, 512)
                # conv02 = tf_utils.conv2d(share_out, 8 * self.ndf, k_h=4, k_w=4, d_h=1, d_w=1, padding='SAME',
                #                          name='conv02_conv_c2')
                # conv02 = tf_utils.norm(conv02, _type='instance', _ops=self._ops, name='conv02_norm_c2')
                # conv02 = tf_utils.lrelu(conv02, name='conv02_lrelu_c2', is_print=True)

                # (N, H/16, W/16, 512) -> (N, H/16, W/16, 1)
                conv02 = tf_utils.conv2d(share_out, 1, k_h=4, k_w=4, d_h=1, d_w=1, padding='SAME',
                                         name='conv02_conv_c2', is_print=True)

                if self.use_sigmoid:
                    output = tf_utils.sigmoid(conv02, name='output_sigmoid_c2', is_print=True)
                else:
                    output = tf_utils.identity(conv02, name='output_without_sigmoid_c2', is_print=True)

                # set reuse=True for next call
                self.reuse_extra_2 = True

        self.unpair_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name+'_unpair')
        self.pair_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name+'_pair')
        self.share_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

        return output

    def share_net(self, x, name=''):
        with tf.variable_scope(name, reuse=self.reuse):
            tf_utils.print_activations(x)

            # (N, H/2, W/2, 64) -> (N, H/4, W/4, 128)
            conv1 = tf_utils.conv2d(x, 2 * self.ndf, k_h=4, k_w=4, d_h=2, d_w=2, padding='SAME',
                                    name='conv1_conv')
            conv1 = tf_utils.norm(conv1, _type='instance', _ops=self._ops, name='conv1_norm')
            conv1 = tf_utils.lrelu(conv1, name='conv1_lrelu', is_print=True)

            # (N, H/4, W/4, 128) -> (N, H/8, W/8, 256)
            conv2 = tf_utils.conv2d(conv1, 4 * self.ndf, k_h=4, k_w=4, d_h=2, d_w=2, padding='SAME',
                                    name='conv2_conv')
            conv2 = tf_utils.norm(conv2, _type='instance', _ops=self._ops, name='conv2_norm')
            conv2 = tf_utils.lrelu(conv2, name='conv2_lrelu', is_print=True)

            # (N, H/8, W/8, 256) -> (N, H/16, W/16, 512)
            conv3 = tf_utils.conv2d(conv2, 8 * self.ndf, k_h=4, k_w=4, d_h=2, d_w=2, padding='SAME',
                                    name='conv3_conv')
            conv3 = tf_utils.norm(conv3, _type='instance', _ops=self._ops, name='conv3_norm')
            conv3 = tf_utils.lrelu(conv3, name='conv3_lrelu', is_print=True)

            # # (N, H/8, W/8, 256) -> (N, H/16, W/16, 512)
            # conv4 = tf_utils.conv2d(conv3, 8 * self.ndf, k_h=4, k_w=4, d_h=2, d_w=2, padding='SAME',
            #                         name='conv4_conv')
            # conv4 = tf_utils.norm(conv4, _type='instance', _ops=self._ops, name='conv4_norm')
            # conv4 = tf_utils.lrelu(conv4, name='conv4_lrelu', is_print=True)

            # (N, H/16, W/16, 512) -> (N, H/16, W/16, 1)
            # conv5 = tf_utils.conv2d(conv4, 1, k_h=4, k_w=4, d_h=1, d_w=1, padding='SAME',
            #                         name='conv5_conv', is_print=True)
            #
            # if self.use_sigmoid:
            #     output = tf_utils.sigmoid(conv5, name='output_sigmoid', is_print=True)
            # else:
            #     output = tf_utils.identity(conv5, name='output_without_sigmoid', is_print=True)

            # set reuse=True for next call
            self.reuse = True

            return conv3
