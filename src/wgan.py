import tensorflow as tf
from dcgan import DCGAN


class WGAN(DCGAN):
    def __init__(self, sess, flags, dataset):
        self.dataset = dataset
        self.clip_val = 0.01
        self.num_iter_dis = 5
        self.lr = 5e-5

        super(WGAN, self).__init__(sess, flags, dataset.image_size)
        print("Initialized WGAN SUCCESS!")

    def _build_net(self):
        self.Y = tf.placeholder(tf.float32, shape=[None, *self.image_size], name='output')
        self.z = tf.placeholder(tf.float32, shape=[None, self.flags.z_dim], name='latent_vector')

        self.g_samples = self.generator(self.z)
        _, d_logit_real = self.discriminator(self.Y)
        _, d_logit_fake = self.discriminator(self.g_samples, is_reuse=True)

        # discriminator loss
        self.d_loss = tf.reduce_mean(d_logit_real) - tf.reduce_mean(d_logit_fake)
        # generator loss
        self.g_loss = -tf.reduce_mean(d_logit_fake)

        d_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='d_')
        g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='g_')

        # Optimizers for generator and discriminator
        dis_op = tf.train.RMSPropOptimizer(learning_rate=self.lr).minimize(-self.d_loss, var_list=d_vars)
        dis_ops = [dis_op] + self._dis_train_ops
        self.dis_optim = tf.group(*dis_ops)
        self.clip_dis = [var.assign(tf.clip_by_value(var, -self.clip_val, self.clip_val)) for var in d_vars]

        gen_op = tf.train.RMSPropOptimizer(learning_rate=self.lr).minimize(self.g_loss, var_list=g_vars)
        gen_ops = [gen_op] + self._gen_train_ops
        self.gen_optim = tf.group(*gen_ops)

    def train_step(self, x_data, y_data):
        d_loss, summary_d, summary_g = None, None, None

        # train discriminator
        for idx in range(self.num_iter_dis):
            _, y_data = self.dataset.train_next_batch(batch_size=self.flags.batch_size,
                                                      which_direction=self.flags.which_direction)
            dis_feed = {self.z: self.sample_z(num=self.flags.batch_size), self.Y: y_data}
            _, d_loss, _ = self.sess.run([self.dis_optim, self.d_loss, self.clip_dis], feed_dict=dis_feed)

        # train generator
        _, y_data = self.dataset.train_next_batch(batch_size=self.flags.batch_size,
                                                  which_direction=self.flags.which_direction)
        gen_feed = {self.z: self.sample_z(num=self.flags.batch_size), self.Y: y_data}
        _, g_loss, summary = self.sess.run([self.gen_optim, self.g_loss, self.summary_op], feed_dict=gen_feed)

        return [d_loss, g_loss], summary
