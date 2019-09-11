import numpy as np
import scipy.misc
import os
import cv2

import utils as utils


def main(source, person_id, dataset, size=256, img_type='.png'):
    target_file = '../Data/{}/{}/'.format(dataset, person_id)

    if not os.path.isdir(target_file):
        os.makedirs(target_file)

    data = utils.all_files_under(os.path.join(source, person_id), extension=img_type, sort=True)

    print('Number of data A: {}'.format(len(data)))

    # make training images
    for idx in range(0, len(data), 2):
        a_img = imread(data[idx], is_grayscale=True)
        b_img = imread(data[idx+1], is_grayscale=True)
        print(data[idx])
        print(data[idx+1])
        imsave(a_img, b_img, os.path.join(target_file, person_id + '_' + str(int(idx/2)) + img_type), size=size)


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
    file_lists = ['p004', 'p005', 'p009', 'p012', 'p014', 'p015', 'p016', 'p017', 'p019', 'p020']

    source = '../C2M/datasets/DB_05_Brain_T2/t2_ct_with_4080setting'
    dataset_name = 'brain05_new'

    for idx in range(len(file_lists)):
        main(source, file_lists[idx], dataset_name, img_type='.png')
