########################################
#     import requirement libraries     #
########################################
import os
import re
import numpy as np
import random
import cv2

from keras.utils import to_categorical
li = []

class Spatial():
    def __init__(self, root_dir, batch_size=32):
        self._data_list = []
        self._root_dir = root_dir   # data root
        self._batch_size = batch_size
        self._front_idx = 0
        self._end_idx = batch_size

    def set_data_list(self, _data_txt, train_test_type='train', _shuffle=True):
        cv_lists = self.data_dir_reader(_data_txt)

        if train_test_type == 'train':
            final_list = []
            for cv_list in cv_lists:
                file_name = cv_list[0]
                label = cv_list[1]
                final_list.append([file_name + "-original", label])
                for i in range(5):
                    final_list.append([file_name + "-cropped-%d" % i, label])
                    final_list.append([file_name + "-flipped-%d" % i, label])

            self._data_list = final_list

        elif train_test_type == 'test':
            final_list = []
            for cv_list in cv_lists:
                file_name = cv_list[0]
                label = cv_list[1]
                final_list.append([file_name + "-original", label])

            self._data_list = final_list

        else:
            raise ValueError

        if _shuffle:
            random.shuffle(self._data_list)

    @staticmethod
    def data_dir_reader(_txt_path):

        tmp = []
        f = open(_txt_path, 'r')
        for line in f.readlines():
            line = line.replace('\n', '')
            split_line = re.split(r"[\s+,\/]*", line)
            # split_info = dir_name.split(' ')
            # split_info[0] = split_info[0].replace('/', '-')
            file_info = split_line[0] + "-%05d" % int(split_line[1])
            tmp.append([file_info, int(split_line[-1])])

        return tmp

    def next_batch(self):

        data = []
        label = []
        end_of_file = False

        for data_list in self._data_list[self._front_idx: self._end_idx+1]:
            file_name = data_list[0]
            cur_label = data_list[1]

            load_file_name = file_name + '.npy'
            file_root = os.path.join(self._root_dir, load_file_name)

            try:
                dat = np.load(file_root)
                data.append(dat)
                # label_name = data_list.split('-')[0]

                # data.append(self.make_input_shape(file_list, file_root))
                # li.append(label_name)
                onehot_label = to_categorical(cur_label, num_classes=51)
                label.append(onehot_label)
                # TODO: make label using video_name

            except FileNotFoundError:
                continue

        self._front_idx += self._batch_size
        self._end_idx += self._batch_size

        if self._end_idx > len(self._data_list):
            self._end_idx = len(self._data_list)

        if self._front_idx > len(self._data_list):
            end_of_file = True
            self._front_idx = 0
            self._end_idx = self._batch_size

        if not data:
            print("Data is empty!!")

        if not label:
            print('Label is empty!!')

        return np.asarray(data), np.asarray(label), end_of_file

    def shuffle(self):

        if not self._data_list:
            print('Data list is None')
        random.shuffle(self._data_list)
        self._front_idx = 0
        self._end_idx = self._batch_size

    def get_data_list(self):

        return self._data_list
