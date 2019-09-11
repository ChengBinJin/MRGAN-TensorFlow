import tensorflow as tf
# noinspection PyPep8Naming
import TensorFlow_utils as tf_utils
from pix2pix import Pix2Pix


class Pix2PixPatch(Pix2Pix):
    def __init__(self, sess, flags, image_size):
        super(Pix2PixPatch, self).__init__(sess, flags, image_size)
        
        print("Initialized pix2pix-patch SUCCESS!")

    def discriminator(self, data, name='d_', is_reuse=False):
        with tf.variable_scope(name) as scope:
            if is_reuse is True:
                scope.reuse_variables()

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

            # 32 -> 16
            h3_conv2d = tf_utils.conv2d(h2_lrelu, self.dis_c[3], name='h3_conv2d')
            h3_batchnorm = tf_utils.batch_norm(h3_conv2d, name='h3_batchnorm', _ops=self._dis_train_ops)
            h3_lrelu = tf_utils.lrelu(h3_batchnorm, name='h3_lrelu')

            # Patch GAN: 16 -> 16
            h4_conv2d = tf_utils.conv2d(h3_lrelu, self.dis_c[4], k_h=3, k_w=3, d_h=1, d_w=1, name='h4_conv2d')

            return tf.nn.sigmoid(h4_conv2d), h4_conv2d
