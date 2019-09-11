import os
import sys
import random
import scipy.misc
import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')  # or whatever other backend that you want to solve Segmentation fault (core dumped)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


class ImagePool(object):
    def __init__(self, pool_size=50):
        self.pool_size = pool_size
        self.imgs = []

    def query(self, imgs):
        if self.pool_size == 0:
            return imgs

        if len(self.imgs) < self.pool_size:
            self.imgs.append(imgs)
            return imgs
        else:
            if random.random() > 0.5:
                # use old image
                random_id = random.randrange(0, self.pool_size)
                tmp_imgs = self.imgs[random_id].copy()
                self.imgs[random_id] = imgs.copy()
                return tmp_imgs
            else:
                return imgs


def all_files_under(path, extension=None, append_path=True, sort=True):
    if append_path:
        if extension is None:
            filenames = [os.path.join(path, fname) for fname in os.listdir(path)]
        else:
            filenames = [os.path.join(path, fname)
                         for fname in os.listdir(path) if fname.endswith(extension)]
    else:
        if extension is None:
            filenames = [os.path.basename(fname) for fname in os.listdir(path)]
        else:
            filenames = [os.path.basename(fname)
                         for fname in os.listdir(path) if fname.endswith(extension)]

    if sort:
        filenames = sorted(filenames)

    return filenames


def print_metrics(itr, kargs):
    print("*** Iteration {}  ====> ".format(itr))
    for name, value in kargs.items():
        print("{} : {}, ".format(name, value))
    print("")
    sys.stdout.flush()


def transform(imgs):
    return imgs / 127.5 - 1.0


def inverse_transform(imgs):
    return (imgs + 1.) / 2.


def preprocess_pair(img_a, img_b, load_size=286, fine_size=256, flip=True, is_test=False):
    if is_test:
        img_a = scipy.misc.imresize(img_a, [fine_size, fine_size])
        img_b = scipy.misc.imresize(img_b, [fine_size, fine_size])
    else:
        img_a = scipy.misc.imresize(img_a, [load_size, load_size])
        img_b = scipy.misc.imresize(img_b, [load_size, load_size])

        h1 = int(np.ceil(np.random.uniform(1e-2, load_size - fine_size)))
        w1 = int(np.ceil(np.random.uniform(1e-2, load_size - fine_size)))
        img_a = img_a[h1:h1 + fine_size, w1:w1 + fine_size]
        img_b = img_b[h1:h1 + fine_size, w1:w1 + fine_size]

        if flip and np.random.random() > 0.5:
            img_a = np.fliplr(img_a)
            img_b = np.fliplr(img_b)

    return img_a, img_b


def imread(path, is_gray_scale=False, img_size=None):
    if is_gray_scale:
        img = scipy.misc.imread(path, flatten=True).astype(np.float)
    else:
        img = scipy.misc.imread(path, mode='RGB').astype(np.float)

        if not (img.ndim == 3 and img.shape[2] == 3):
            img = np.dstack((img, img, img))
        
    if img_size is not None:
        img = scipy.misc.imresize(img, img_size)

    return img


def load_image(image_path, which_direction=0, is_gray_scale=True):
    input_img = imread(image_path, is_gray_scale=is_gray_scale)
    w_pair = int(input_img.shape[1])
    w_single = int(w_pair / 2)

    if which_direction == 0:    # ct to mr
        img_a = input_img[:, 0:w_single]
        img_b = input_img[:, w_single:w_pair]
    else:                       # mr to ct
        img_a = input_img[:, w_single:w_pair]
        img_b = input_img[:, 0:w_single]

    return img_a, img_b


def load_data(image_path, flip=True, is_test=False, which_direction=0, is_gray_scale=True):
    img_a, img_b = load_image(image_path=image_path, which_direction=which_direction,
                              is_gray_scale=is_gray_scale)

    img_a, img_b = preprocess_pair(img_a, img_b, flip=flip, is_test=is_test)
    img_a = transform(img_a)
    img_b = transform(img_b)

    if (img_a.ndim == 2) and (img_b.ndim == 2):
        # img_a = np.reshape(img_a, (img_a.shape[0], img_a.shape[1], 1))
        # img_b = np.reshape(img_b, (img_b.shape[0], img_b.shape[1], 1))
        img_a = np.expand_dims(img_a, axis=2)
        img_b = np.expand_dims(img_b, axis=2)

    return img_a, img_b


def plots(imgs, iter_time, image_size, save_file):
    # parameters for plot size
    scale, margin = 0.015, 0.02
    n_cols, n_rows = len(imgs), imgs[0].shape[0]
    cell_size_h, cell_size_w = imgs[0].shape[1] * scale, imgs[0].shape[2] * scale

    fig = plt.figure(figsize=(cell_size_w * n_cols, cell_size_h * n_rows))  # (column, row)
    gs = gridspec.GridSpec(n_rows, n_cols)  # (row, column)
    gs.update(wspace=margin, hspace=margin)

    imgs = [inverse_transform(imgs[idx]) for idx in range(len(imgs))]

    # save more bigger image
    for col_index in range(n_cols):
        for row_index in range(n_rows):
            ax = plt.subplot(gs[row_index * n_cols + col_index])
            plt.axis('off')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect('equal')
            if image_size[2] == 3:
                plt.imshow((imgs[col_index][row_index]).reshape(
                    image_size[0], image_size[1], image_size[2]), cmap='Greys_r')
            elif image_size[2] == 1:
                plt.imshow((imgs[col_index][row_index]).reshape(image_size[0], image_size[1]), cmap='Greys_r')

    plt.savefig(os.path.join(save_file, '{}.png').format(str(iter_time).zfill(7)), bbox_inches='tight')
    plt.close(fig)


def save_cycle_consistent_imgs(imgs, iter_time, image_size, save_file):
    # save imgs for cyclegan
    w = image_size[1]
    new_img = np.zeros((image_size[0], len(imgs)*image_size[1]))
    imgs = [inverse_transform(imgs[idx]) for idx in range(len(imgs))]

    # new_img[:, 0:image_size[1]] = imgs[0][0, :, :, 0]  # ct img
    # new_img[:, image_size[1]:2*image_size[1]] = imgs[1][0, :, :, 0]  # pred img
    # new_img[:, 2*image_size[1]:3*image_size[1]] = imgs[2][0, :, :, 0]  # mri img
    # new_img[:, 3 * image_size[1]:4 * image_size[1]] = imgs[3][0, :, :, 0]  # mri img

    new_img[:, 0:w] = np.squeeze(imgs[0])           # ct img
    new_img[:, w:2 * w] = np.squeeze(imgs[1])       # pred img
    new_img[:, 2 * w:3 * w] = np.squeeze(imgs[2])   # mri img
    new_img[:, 3 * w:4 * w] = np.squeeze(imgs[3])   # mri img

    # save imgs
    scipy.misc.imsave(os.path.join(save_file, '{}.png').format(str(iter_time).zfill(7)), new_img)


def exists(p, msg):
    assert os.path.exists(p), msg
