import os
import cv2
import time
import csv
import numpy as np
import tensorflow as tf
from datetime import datetime

import utils as utils
from dataset import dataset
from eval import Eval
from gan_repository import gan_repository


class Solver(object):
    def __init__(self, flags):
        run_config = tf.ConfigProto()
        run_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=run_config)

        self.flags = flags
        self.best_mae = float("inf")
        self.iter_time = 0

        self.train_dataset = dataset(self.flags.dataset)
        print('train dataset name: {}'.format(self.train_dataset.dataset_name))
        if self.flags.dataset == 'brain01':
            self.val_dataset = dataset('brain05')
        elif self.flags.dataset == 'spine04':
            self.val_dataset = dataset('spine_val')
        print('val datset name: {}'.format(self.val_dataset.dataset_name))

        self.model = gan_repository(self.sess, self.flags, self.train_dataset)
        self._make_folders()
        self.evaluator = None

        self.saver = tf.train.Saver()
        self.sess.run(tf.global_variables_initializer())

        # threads for tfrecrod
        self.coord = tf.train.Coordinator()
        self.threads = tf.train.start_queue_runners(sess=self.sess, coord=self.coord)

        # Utils.show_all_variables()

    def _make_folders(self):
        if self.flags.is_train:
            cur_time = datetime.now().strftime("%Y%m%d-%H%M")
            self.model_out_dir = "checkpoints_{}/{}/{}/{}".format(self.flags.which_direction,
                                                                  self.flags.dataset, self.flags.gan_model, cur_time)
            self.sample_out_dir = "sample_{}/{}/{}/{}".format(self.flags.which_direction,
                                                              self.flags.dataset, self.flags.gan_model, cur_time)

            self.train_writer = tf.summary.FileWriter('logs_{}/{}/{}/{}'.format(self.flags.which_direction,
                                                                                self.flags.dataset,
                                                                                self.flags.gan_model, cur_time))

            if not os.path.isdir(self.model_out_dir):
                os.makedirs(self.model_out_dir)
            if not os.path.isdir(self.sample_out_dir):
                os.makedirs(self.sample_out_dir)

        elif not self.flags.is_train:
            # self.model_out_dir = "checkpoints_{}/{}/{}/{}".format(self.flags.which_direction,
            #                                                       self.flags.dataset, self.flags.gan_model,
            #                                                       self.flags.load_model)
            self.model_out_dir = "checkpoints_{}/{}/{}".format(self.flags.which_direction,
                                                                  self.flags.dataset, self.flags.gan_model)

            self.test_out_dir = "test_{}/{}/{}/{}".format(self.flags.which_direction,
                                                          self.flags.dataset, self.flags.gan_model,
                                                          self.flags.load_model)

            # self.exp_out_dir = "experiments_{}/{}/{}_{}".format(self.flags.which_direction,
            #                                                     self.flags.dataset, self.flags.gan_model,
            #                                                     self.flags.batch_size)

            if not os.path.isdir(self.test_out_dir):
                os.makedirs(self.test_out_dir)

            # if not os.path.isdir(self.exp_out_dir):
            #     os.makedirs(self.exp_out_dir)

    def train(self):
        if self.flags.is_continue:
            if self.load_model():
                print(' [*] Load success!\n')
            else:
                print(' [!] Load faild...\n')

        try:
            for self.iter_time in range(self.flags.iters+1):
                # sampling images and save them
                self.sample(self.iter_time)

                # read batch data
                x_imgs, y_imgs = self.train_dataset.train_next_batch(batch_size=self.flags.batch_size,
                                                                     which_direction=self.flags.which_direction)

                loss, summary = self.model.train_step(x_imgs, y_imgs)

                # print loss information
                self.model.print_info(loss, self.iter_time)

                # write log to tensorboard
                self.train_writer.add_summary(summary, self.iter_time)
                self.train_writer.flush()

                self.save_model(self.iter_time)  # save model at the end

            self.save_model(self.flags.iters)  # save model at the end

        except KeyboardInterrupt:
            self.coord.request_stop()
        except Exception as e:
            self.coord.request_stop(e)
        finally:
            # when done, ask the threads to stop
            self.coord.request_stop()
            self.coord.join(self.threads)

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
        size = 256
        folder = './test01'
        imgPaths = utils.all_files_under(folder, extension='.png')

        saveFolder = os.path.join(folder, 'results')
        if not os.path.exists(saveFolder):
            os.makedirs(saveFolder)

        for idx, imgPath in enumerate(imgPaths):
            img = cv2.imread(imgPath, cv2.IMREAD_GRAYSCALE)
            img = img / 127.5 - 1.
            img = img[:, :, np.newaxis]
            # print('img shape: {}'.format(img.shape))

            y_imgs = self.model.test_only([img])
            y_img = (y_imgs[0] + 1.) / 2.

            cv2.imshow('test', y_img)
            cv2.waitKey(0)

            img = img[:, :, 0]
            y_img = y_img[:, :, 0]

            canvas = np.zeros((size, 2*size), dtype=np.uint8)
            canvas[:, :size] = (255. * ((img + 1.) / 2.)).astype(np.uint8)
            canvas[:, -size:] = (255. * y_img).astype(np.uint8)

            cv2.imwrite(os.path.join(saveFolder, str(idx) + '.png'), canvas)

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

                utils.plots(imgs, global_iter, self.val_dataset.image_size, save_file=self.test_out_dir)
                # utils.save_cycle_consistent_imgs([x_img, y_fake, y_img, recon_x], global_iter,
                #                                  self.val_dataset.image_size, save_file=self.exp_out_dir)

                samples.append(imgs[1])  # imgs[1] == fake_y
                y_imgs.append(y_img)

                global_iter += 1

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

    def sample(self, iter_time):
        if np.mod(iter_time, self.flags.sample_freq) == 0:
            x_imgs, y_imgs = self.train_dataset.train_next_batch(batch_size=self.flags.sample_batch,
                                                                 which_direction=self.flags.which_direction)
            samples = self.model.sample_imgs(x_imgs, y_imgs)
            utils.plots(samples, iter_time, self.train_dataset.image_size, self.sample_out_dir)

    def save_model(self, iter_time):
        if np.mod(iter_time, self.flags.save_freq) == 0:
            model_name = 'model'
            self.saver.save(self.sess, os.path.join(self.model_out_dir, model_name), global_step=iter_time)

            print('=====================================')
            print('              Model saved!           ')
            print('=====================================\n')

    def load_model(self):
        print(' [*] Reading checkpoint...')
        print('checkpoints: {}'.format(self.model_out_dir))

        ckpt = tf.train.get_checkpoint_state(self.model_out_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(self.model_out_dir, ckpt_name))

            meta_graph_path = ckpt.model_checkpoint_path + '.meta'
            self.iter_time = int(meta_graph_path.split('-')[-1].split('.')[0])

            return True
        else:
            return False
