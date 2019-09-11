import os
import cv2
import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')  # or whatever other backend that you want to solve Segmentation fault (core dumped)
import matplotlib.pyplot as plt
# noinspection PyPep8Naming
import utils as Utils

img_size = 256
img_paths = Utils.all_files_under('./exp_0/brain02/mrigan_1', extension='.png')

max_diff = float("inf")
max_id = 300

for idx in range(len(img_paths)):
    print(idx)
    img = cv2.imread(img_paths[idx], 0)

    ct = img[:, 0:img_size]
    pre = img[:, img_size:2 * img_size]
    mri = img[:, 2 * img_size:3 * img_size]
    recon_ct = img[:, 3 * img_size:4 * img_size]

    # diff_mri = np.abs(pre - mri)
    # diff_mri[diff_mri < 100] = 0

    # if np.sum(diff_mri) < 2000000:
    #     print('max_id: {}, diff_sum: {}'.format(idx, np.sum(diff_mri)))

    # plt.figure()
    # plt.imshow(diff_mri, vmin=0, vmax=600, cmap='afmhot')
    # cb = plt.colorbar(ticks=[0, 300, 600])
    # cb.ax.set_yticklabels(cb.ax.get_yticklabels(), fontsize=18)
    # plt.axis('off')
    #
    # plt.savefig(os.path.join('./exp_0/brain02/mrigan_1', 'dif_{}.png').format(str(idx).zfill(3)),
    #             bbox_inches='tight')
    # plt.close()

    diff_rela = np.abs(ct - recon_ct) / (ct.astype(np.float) + 1e-5)
    plt.figure()
    plt.imshow(diff_rela, vmin=0, vmax=100, cmap='afmhot')
    cb = plt.colorbar(ticks=[0, 50, 100])
    cb.ax.set_yticklabels(cb.ax.get_yticklabels(), fontsize=18)
    plt.axis('off')

    plt.savefig(os.path.join('./exp_0/brain02/mrigan_1', 'rela_{}.png').format(str(idx).zfill(3)),
                bbox_inches='tight')
    plt.close()



