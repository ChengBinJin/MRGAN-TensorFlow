import numpy as np


import utils as utils


class Eval(object):
    def __init__(self, image_size, num_vals):
        self.image_size = image_size
        self.num_vals = num_vals
        self.MAX = 65025  # 255 * 255
        self.eps = 1e-3

    def calculate(self, preds, gts):
        # from list to array
        arr_preds = np.asarray(preds).astype(np.float32)
        arr_gts = np.asarray(gts).astype(np.float32)

        # reshape to (N, image_size, image_size)
        arr_preds = np.reshape(arr_preds, (arr_preds.shape[0], self.image_size[0], self.image_size[1]))
        arr_gts = np.reshape(arr_gts, (arr_gts.shape[0], self.image_size[0], self.image_size[1]))

        # conver from [-1. to 1.]  to [0. to 255.]
        arr_preds_ = utils.inverse_transform(arr_preds) * 255.
        arr_gts_ = utils.inverse_transform(arr_gts) * 255.

        mae = self.mean_absoulute_error(arr_preds_, arr_gts_)
        psnr = self.peak_signal_to_noise_ratio(arr_preds_, arr_gts_)

        return mae, psnr

    def mean_absoulute_error(self, preds, gts):
        mae = np.sum(np.abs(preds - gts)) / (self.num_vals * self.image_size[0] * self.image_size[1])
        print('MAE: {:.3f}'.format(mae))

        return mae

    def peak_signal_to_noise_ratio(self, preds, gts):
        upper_bound = 20 * np.log10(self.MAX)
        psnr = upper_bound - 10 * np.log10(np.sum(np.square(preds - gts)) / (
                self.num_vals * self.image_size[0] * self.image_size[1]))
        print('PSNR: {:.3f}\n'.format(psnr))

        return psnr
