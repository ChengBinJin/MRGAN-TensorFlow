import os
import tensorflow as tf
from test_solver import Solver

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('gan_model', 'pix2pix-patch', 'default: pix2pix-patch')
tf.flags.DEFINE_string('dataset', 'spine04', 'default: spine04')
tf.flags.DEFINE_string('gpu_index', '0', 'gpu index, default: 0')

tf.flags.DEFINE_integer('which_direction', 0, 'AtoB (0) or BtoA (1), default: AtoB (ct2mri), 0')
tf.flags.DEFINE_string('load_model', None,
                       'folder of saved model that you with to test, (e.g. 20180815-2019), default: None')

# used in _build_net() only
tf.flags.DEFINE_float('learning_rate', 2e-4, 'initial learning rate for Adam, default: 0.0002')
tf.flags.DEFINE_float('beta1', 0.5, 'momentum term of adam')


def main(_):
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu_index

    solver = Solver(FLAGS)
    solver.test()


if __name__ == '__main__':
    tf.app.run()
