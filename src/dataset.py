import os
import numpy as np
import random
import cv2
from datetime import datetime
from tensorflow.examples.tutorials.mnist import input_data

import utils as utils

# brain01: paired brain dataset from the Havard
# brain02: unpaired brain dataset from the Radiopaedia
# brain03: unpaired brain dataset from the INTERNET STROKE CENTER (not use now)
# spine04: paired spine dataset from the Pusan National University Hospital
# brain05: paired brain dataset from the Pusan National University Hospital
# mnist:   mnist dataset for varifying implementation


def dataset(dataset_name):
    if dataset_name == 'brain01' or dataset_name == 'spine04':
        return CtMRIDataset(dataset_name)
    elif dataset_name == 'spine_val':
        return SpineVal(dataset_name)
    elif dataset_name == 'brain02':
        return Brain02(dataset_name)
    elif dataset_name == 'brain05':
        return Brain05(dataset_name)
    elif dataset_name == 'mnist':
        return MnistDataset(dataset_name)
    else:
        raise NotImplementedError


class CtMRIDataset(object):
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        self.sub_name = "raw"
        if self.dataset_name == 'spine04':
            self.sub_name = "train/"
        self.num_trains = 0

        self.ct_mri_path = '../../Data/{}'.format(os.path.join(self.dataset_name, self.sub_name))
        self.is_gray = True
        self._load_ct_mri()

    def _load_ct_mri(self):
        print('Load {} dataset...'.format(self.dataset_name))

        self.image_size = (256, 256, 1)
        self.train_data = utils.all_files_under(self.ct_mri_path, extension='.png')
        self.num_trains = len(self.train_data)

        print('Load {} dataset SUCCESS!'.format(self.dataset_name))

    def train_next_batch(self, batch_size=1, which_direction=0):
        random.seed(datetime.now())  # set random seed
        batch_files = np.random.choice(self.train_data, batch_size, replace=False)

        data_x, data_y = [], []
        for batch_file in batch_files:
            batch_x, batch_y = utils.load_data(image_path=batch_file, which_direction=which_direction)
            data_x.append(batch_x)
            data_y.append(batch_y)

        batch_ximgs = np.asarray(data_x).astype(np.float32)  # list to array
        batch_yimgs = np.asarray(data_y).astype(np.float32)  # list to array

        return batch_ximgs, batch_yimgs


class Brain02(object):
    def __init__(self, dataset_name):
        # Brain02 is unpaired brain dataset that use tfrecord
        self.dataset_name = dataset_name
        self.image_size = (256, 256, 1)
        self.num_trains = 2568  # 2568

        # tfrecord path
        self.ct_tfpath = '../../Data/brain02/CT.tfrecords'
        self.mri_tfpath = '../../Data/brain02/MRI.tfrecords'

    def __call__(self):
        print('Load {} dataset...'.format(self.dataset_name))
        print('Load {} dataset SUCCESS!'.format(self.dataset_name))
        return [self.ct_tfpath, self.mri_tfpath]


class Brain05(object):
    def __init__(self, dataset_name):
        # Brain05 is paired brain dataset from the Pusan National University Hospital
        self.dataset_name = dataset_name
        self.image_size = (256, 256, 1)
        self.is_gray = True
        self.path_file = '../../Data/brain05'

        self.person_id_list = ['p004', 'p005', 'p009', 'p012', 'p014',
                               'p015', 'p016', 'p017', 'p019', 'p020']
        self.num_persons = len(self.person_id_list)

        self.num_vals = np.zeros(self.num_persons, dtype=np.uint16)
        self.data_path = []

        self._load_ct_mri()

    def _load_ct_mri(self):
        print('Load {} dataset...'.format(self.dataset_name))

        for idx, p_id in enumerate(self.person_id_list):
            data_path = utils.all_files_under(os.path.join(self.path_file, p_id))
            self.num_vals[idx] = len(data_path)
            self.data_path.append(data_path)

        print('Load {} dataset SUCCESS!'.format(self.dataset_name))

    def val_next_batch(self, p_id, iter_time, which_direction=0):
        data_x, data_y = utils.load_data(self.data_path[p_id][iter_time], flip=False, is_test=True,
                                         which_direction=which_direction, is_gray_scale=self.is_gray)
        batch_ximg = np.asarray(data_x).astype(np.float32)
        batch_yimg = np.asarray(data_y).astype(np.float32)

        if np.ndim(batch_ximg) == 3:  # shape from (256, 256, 1) to (1, 256, 256, 1)
            batch_ximg = np.expand_dims(batch_ximg, axis=0)
            batch_yimg = np.expand_dims(batch_yimg, axis=0)

        return batch_ximg, batch_yimg


class SpineVal(object):
    def __init__(self, dataset_name):
        # Spine04 is paired spine dataset from the Pusan National University Hospital
        self.dataset_name = dataset_name
        self.image_size = (256, 256, 1)
        self.real_size = (300, 200, 1)
        self.is_gray = True
        self.path_file = '../../Data/spine04/val'

        self.person_id_list = ['p000', 'p001', 'p002', 'p003', 'p004', 'p005', 'p006', 'p007', 'p008', 'p009',
                               'p010', 'p011', 'p012', 'p013', 'p014', 'p015', 'p016', 'p017', 'p018', 'p019',
                               'p020', 'p021', 'p022', 'p023', 'p024', 'p025', 'p026', 'p027', 'p028', 'p029',
                               'p030', 'p031', 'p032', 'p033']
        self.num_persons = len(self.person_id_list)

        self.num_vals = np.zeros(self.num_persons, dtype=np.uint16)
        self.data_path = []

        self._load_ct_mri()

    def _load_ct_mri(self):
        print('Load {} dataset...'.format(self.dataset_name))

        for idx, p_id in enumerate(self.person_id_list):
            data_path = utils.all_files_under(os.path.join(self.path_file, p_id))
            self.num_vals[idx] = len(data_path)
            self.data_path.append(data_path)

        print('Load {} dataset SUCCESS!'.format(self.dataset_name))

    def val_next_batch(self, p_id, iter_time, which_direction=0):
        data_x, data_y = utils.load_data(self.data_path[p_id][iter_time], flip=False, is_test=True,
                                         which_direction=which_direction, is_gray_scale=self.is_gray)
        batch_ximg = np.asarray(data_x).astype(np.float32)
        batch_yimg = np.asarray(data_y).astype(np.float32)

        if np.ndim(batch_ximg) == 3:  # shape from (256, 256, 1) to (1, 256, 256, 1)
            batch_ximg = np.expand_dims(batch_ximg, axis=0)
            batch_yimg = np.expand_dims(batch_yimg, axis=0)

        return batch_ximg, batch_yimg


class MnistDataset(object):
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        self.num_trains, self.num_vals = 0, 0

        self.mnist_path = os.path.join('../../Data', self.dataset_name)
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
