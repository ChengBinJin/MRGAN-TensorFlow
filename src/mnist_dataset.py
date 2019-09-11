import os
import cv2
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data


class MnistDataset(object):
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        self.num_trains, self.num_vals = 0, 0

        self.mnist_path = os.path.join('../Data', self.dataset_name)
        self._load_mnist()

    def _load_mnist(self):
        print('Load {} dataset...'.format(self.dataset_name))
        self.train_data = input_data.read_data_sets(self.mnist_path, one_hot=True)
        self.num_trains = self.train_data.train.num_examples

        self.num_vals = self.train_data.train.num_examples
        self.val_data = self.train_data
        self.image_size = (32, 32, 1)
        print('Load {} dataset SUCCESS!'.format(self.dataset_name))

    def train_next_batch(self, batch_size, which_direction=0):
        batch_imgs, _ = self.train_data.train.next_batch(batch_size)

        # reshape 784 vector to 28 x 28
        imgs = [np.reshape(batch_imgs[idx], (28, 28)) for idx in range(batch_imgs.shape[0])]
        # resize to 32 x 32
        resize_imgs = [cv2.resize(imgs[idx], (32, 32)) for idx in range(len(imgs))]
        # reshape 32 x 32 to 32 x 32 x 1
        reshape_imgs = [np.reshape(resize_imgs[idx], self.image_size) for idx in range(len(resize_imgs))]
        # list to array
        arr_imgs = np.asarray(reshape_imgs).astype(np.float32)  # list to array

        return arr_imgs, arr_imgs

    def val_next_batch(self, iter_time=0, which_direction=0):
        arr_img, _ = self.train_next_batch(batch_size=1)
        return arr_img, arr_img
