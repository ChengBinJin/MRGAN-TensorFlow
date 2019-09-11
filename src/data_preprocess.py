import numpy as np
import scipy.misc
import os
import cv2

import utils as utils


def main(source_a, source_b, dataset, size=256, read_img_type='.png', write_img_type='.png'):
    # target_train_file = '../Data/{}/train/'.format(dataset)
    target_val_file = '../Data/{}/val/'.format(dataset)

    # if not os.path.isdir(target_train_file):
    #     os.makedirs(target_train_file)
    if not os.path.isdir(target_val_file):
        os.makedirs(target_val_file)

    data_a = utils.all_files_under(source_a, extension=read_img_type, sort=True)
    data_b = utils.all_files_under(source_b, extension=read_img_type, sort=True)

    print('Number of data A: {}'.format(len(data_a)))
    print('Number of data B: {}'.format(len(data_b)))

    # make training images
    for idx in range(len(data_a)):
        a_img = imread(data_a[idx], is_grayscale=True)
        b_img = imread(data_b[idx], is_grayscale=True)
        print(data_a[idx])
        print(data_b[idx])
        imsave(a_img, b_img, os.path.join(target_val_file, str(idx) + write_img_type), size=size)


def imread(path, is_grayscale=False):
    if is_grayscale is True:
        return scipy.misc.imread(path, flatten=True).astype(np.float)
    else:
        return scipy.misc.imread(path).astype(np.float)


def imsave(a_img, b_img, save_path, size=256):
    image = np.zeros((size, 2*size), dtype=np.uint8)

    h, w = a_img.shape
    ratio = size / np.maximum(h, w)
    w_ = np.ceil(w * ratio).astype(np.uint16)
    h_ = np.ceil(h * ratio).astype(np.uint16)

    start_position_w = np.ceil((size - w_) / 2).astype(np.uint16)
    end_position_w = np.ceil((size - w_) / 2).astype(np.uint16) + w_

    start_position_h = np.ceil((size - h_) / 2).astype(np.uint16)
    end_position_h = np.ceil((size - h_) / 2).astype(np.uint16) + h_

    left_img = cv2.resize(a_img, (w_, h_), interpolation=cv2.INTER_LINEAR)
    right_img = cv2.resize(b_img, (w_, h_), interpolation=cv2.INTER_LINEAR)

    image[start_position_h:end_position_h, start_position_w:end_position_w] = left_img
    image[start_position_h:end_position_h, size + start_position_w:size + end_position_w] = right_img
    scipy.misc.imsave(save_path, image)


if __name__ == '__main__':
    sourceA = '../C2M/datasets/DB_04_Spine/val/CT'
    sourcebB = '../C2M/datasets/DB_04_Spine/val/MRI/'
    dataset_name = 'spine04'

    main(sourceA, sourcebB, dataset_name, read_img_type='.jpg')
