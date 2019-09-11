import os
import time
import csv
import scipy.misc
import numpy as np
import tensorflow as tf
import matplotlib as mpl
mpl.use('TkAgg')  # or whatever other backend that you want to solve Segmentation fault (core dumped)
import matplotlib.pyplot as plt

# import utils as utils
from dataset import dataset
from eval import Eval
from gan_repository import gan_repository


class Solver(object):
    def __init__(self, flags):
        run_config = tf.ConfigProto()
        run_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=run_config)

        self.flags = flags

        if self.flags.dataset == 'spine04':
            self.val_dataset = dataset('spine_val')
            self.crop_cord = self.calculate_w_start()
            print('train dataset name: {}'.format(self.val_dataset.dataset_name))

        self.model = gan_repository(self.sess, self.flags, self.val_dataset)
        self._make_folders()
        self.evaluator = None

        self.saver = tf.train.Saver()
        self.sess.run(tf.global_variables_initializer())

        # threads for tfrecrod
        self.coord = tf.train.Coordinator()
        self.threads = tf.train.start_queue_runners(sess=self.sess, coord=self.coord)

        # Utils.show_all_variables()

    def _make_folders(self):
        self.model_out_dir = "checkpoints_{}/{}/{}/{}".format(self.flags.which_direction, self.flags.dataset,
                                                              self.flags.gan_model, self.flags.load_model)
        self.test_out_dir = "experiments_{}/{}/{}/{}".format(self.flags.which_direction, self.flags.dataset,
                                                             self.flags.gan_model, self.flags.load_model)

        if not os.path.isdir(self.test_out_dir):
            os.makedirs(self.test_out_dir)

    def test(self):
        if self.load_model():
            print(' [*] Load success')
        else:
            print(' [!] Load failed...')

        try:
            # self.eval_test()  # evaluation for test dataset
            self.simple_test()
        except KeyboardInterrupt:
            self.coord.request_stop()
        except Exception as e:
            self.coord.request_stop(e)
        finally:
            # when done, ask the threads to stop
            self.coord.request_stop()
            self.coord.join(self.threads)

    def simple_test(self):
        total_time = 0.


    def eval_test(self):
        total_time = 0.
        mae_record, psnr_record = np.zeros(self.val_dataset.num_persons), np.zeros(self.val_dataset.num_persons)

        # create csv file
        csvfile = open(os.path.join(self.test_out_dir, 'stat.csv'), 'w', newline='')
        csvwriter = csv.writer(csvfile, delimiter=',')
        csvwriter.writerow(['p_id', 'MAE', 'PSNR', 'MAE std', 'PSNR std'])

        global_iter = 0
        for p_id in range(self.val_dataset.num_persons):
            self.evaluator = Eval(self.val_dataset.image_size, self.val_dataset.num_vals[p_id])
            samples, y_imgs = [], []

            for iter_ in range(self.val_dataset.num_vals[p_id]):
                print('p_id: {}, iter: {}'.format(p_id, iter_))

                x_img, y_img = self.val_dataset.val_next_batch(p_id, iter_, which_direction=self.flags.which_direction)
                start_time = time.time()
                imgs = self.model.test_step(x_img, y_img)
                total_time += time.time() - start_time

                # utils.plots(imgs, global_iter, self.val_dataset.image_size, save_file=self.test_out_dir)
                self.plots(imgs, global_iter)

                samples.append(imgs[1])  # imgs[1] == fake_y
                y_imgs.append(y_img)
                global_iter += 1

            # calcualte MAE and PSNR
            mae_record[p_id], psnr_record[p_id] = self.evaluator.calculate(samples, y_imgs)
            # write to csv file
            csvwriter.writerow([p_id + 1, mae_record[p_id], psnr_record[p_id]])

        for p_id in range(self.val_dataset.num_persons):
            print('p_id: {}, MAE: {:.2f}, PSNR: {:.2f}'.format(p_id, mae_record[p_id], psnr_record[p_id]))

        print('MAE Avg. {:.2f} and SD. {:.2f}'.format(np.mean(mae_record), np.std(mae_record)))
        print('PSRN Avg. {:.2f} and SD. {:.2f}'.format(np.mean(psnr_record), np.std(psnr_record)))
        print('Average PT: {:.2f} msec.'.format((total_time / np.sum(self.val_dataset.num_vals)) * 1000))

        # write to csv file for mean and std of MAE and PSNR
        csvwriter.writerow(['MEAN', np.mean(mae_record), np.mean(psnr_record), np.std(mae_record), np.std(psnr_record)])

    def load_model(self):
        print(' [*] Reading checkpoint...')

        ckpt = tf.train.get_checkpoint_state(self.model_out_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(self.model_out_dir, ckpt_name))
            return True
        else:
            return False

    def plots(self, imgs, global_iter):
        ct_img, gen_img, mr_img = imgs

        ct_img = ct_img[0, self.crop_cord[0]:self.crop_cord[1], self.crop_cord[2]:self.crop_cord[3], 0]
        gen_img = gen_img[0, self.crop_cord[0]:self.crop_cord[1], self.crop_cord[2]:self.crop_cord[3], 0]
        mr_img = mr_img[0, self.crop_cord[0]:self.crop_cord[1], self.crop_cord[2]:self.crop_cord[3], 0]
        # inverse transform from [-1., 1.0] to [0, 1]
        ct_img, gen_img, mr_img = (ct_img + 1.) / 2., (gen_img + 1.) / 2., (mr_img + 1.) / 2.
        # calculate difference between gen_img and mr_img
        self.difference(gen_img, mr_img, iter_time=global_iter, save_file=self.test_out_dir)

        scipy.misc.imsave(os.path.join(self.test_out_dir, 'ct_{}.png').format(str(global_iter).zfill(6)), ct_img)
        scipy.misc.imsave(os.path.join(self.test_out_dir, 'gen_{}.png').format(str(global_iter).zfill(6)), gen_img)
        scipy.misc.imsave(os.path.join(self.test_out_dir, 'mr_{}.png').format(str(global_iter).zfill(6)), mr_img)

    def calculate_w_start(self, size=256):
        # this function for spine_val to crop original input
        h, w = self.val_dataset.real_size[:2]  # real image size
        ratio = size / np.maximum(h, w)
        w_ = np.ceil(w * ratio).astype(np.uint16)
        h_ = np.ceil(h * ratio).astype(np.uint16)

        start_position_w = np.ceil((size - w_) / 2).astype(np.uint16)
        end_position_w = np.ceil((size - w_) / 2).astype(np.uint16) + w_

        start_position_h = np.ceil((size - h_) / 2).astype(np.uint16)
        end_position_h = np.ceil((size - h_) / 2).astype(np.uint16) + h_

        return [start_position_h, end_position_h, start_position_w, end_position_w]

    @staticmethod
    def difference(gen_img, mr_img, iter_time, save_file):
        diff_rela = np.abs((gen_img - mr_img) * 255)

        plt.figure()
        plt.imshow(diff_rela, vmin=0, vmax=255, cmap='afmhot')
        cb = plt.colorbar(ticks=[0, 128, 255])
        cb.ax.set_yticklabels(cb.ax.get_yticklabels(), fontsize=18)
        plt.axis('off')

        plt.savefig(os.path.join(save_file, 'dif_{}.png').format(str(iter_time).zfill(6)), bbox_inches='tight')
        plt.close()

