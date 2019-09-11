import os
import tensorflow as tf
from solver import Solver

FLAGS = tf.flags.FLAGS

# vanilla_gan, dcgan, pix2pix, pix2pix-patch, wgan, cyclegan, and mrgan
tf.flags.DEFINE_string('gan_model', 'pix2pix', 'default: wgan')
tf.flags.DEFINE_bool('is_train', False, 'default: False (test mode)')
tf.flags.DEFINE_bool('is_continue', False, 'default: False')
tf.flags.DEFINE_string('gpu_index', '0', 'gpu index, default: 0')

tf.flags.DEFINE_string('dataset', 'brain01', 'brain01, brain02, spine04, and mnist dataset will be used, '
                                             'default: brain01')
tf.flags.DEFINE_integer('which_direction', 0, 'AtoB (0) or BtoA (1), default: AtoB (ct2mri), 0')
tf.flags.DEFINE_integer('sample_batch', 4, 'sample batch size, default: 4')
tf.flags.DEFINE_integer('batch_size', 4, 'batch size, default: 4')
tf.flags.DEFINE_float('learning_rate', 2e-4, 'initial learning rate for Adam, default: 0.0002')
tf.flags.DEFINE_integer('z_dim', 100, 'dimension of z vector, default: 100')
tf.flags.DEFINE_float('beta1', 0.5, 'momentum term of adam')

tf.flags.DEFINE_integer('iters', 20, 'number of epochs, default: 200000')
tf.flags.DEFINE_integer('save_freq', 20, 'number of epochs, default: 2000')
tf.flags.DEFINE_integer('print_freq', 2, 'print frequency, default: 100')
tf.flags.DEFINE_integer('sample_freq', 5, 'sample frequency, default: 500')
tf.flags.DEFINE_string('load_model', '20180508-0535',
                       'folder of saved model that you wish to test, (e.g. 20180704-1736), default: None')


def main(_):
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu_index

    solver = Solver(FLAGS)
    if FLAGS.is_train:
        print('==' * 10)
        print('Trainig Mode!')
        print('==' * 10)
        solver.train()
    elif not FLAGS.is_train:
        print('=='*10)
        print('Test Mode!')
        print('==' * 10)
        solver.test()


if __name__ == '__main__':
    tf.app.run()
