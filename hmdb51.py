########################################
#     import requirement libraries     #
########################################
import os
import numpy as np
import random
import cv2

from keras.utils import to_categorical
li = []


class Spatial():
    def __init__(self, root_dir, batch_size=None):
        self._data_list = None
        self._root_dir = root_dir
        self._batch_size = batch_size
        self._front_idx = 0
        self._end_idx = batch_size

    def set_data_list(self, _data_txt, _shuffle=True):
        self._data_list = self.data_dir_reader(_data_txt)

        if _shuffle:
            random.shuffle(self._data_list)

    @staticmethod
    def data_dir_reader(_txt_path):

        tmp = []
        f = open(_txt_path, 'r')
        for dir_name in f.readlines():
            dir_name = dir_name.replace('\n', '')
            tmp.append(dir_name)

        return tmp

    def load_all_data(self):
        data = []
        label = []

        for data_list in self._data_list:
            video_tag = data_list.split(' ')[0]
            video_name = video_tag.split('/')[0]
            video_number = int(video_tag.split('/')[1])
            """
            print(data_list)
            print(data_list.split(' ')[-1])
            print('--')
            """
            load_file_name = "%s_%05d_frame.npy" % (video_name, video_number)
            file_root = self._root_dir + '/' + load_file_name

            try:
                dat = np.load(file_root)

            except FileNotFoundError:
                print('pass')
                pass

            else:
                data.append(dat)
                label_name = int(data_list.split(' ')[-1])

                # data.append(self.make_input_shape(file_list, file_root))
                # li.append(label_name)
                onehot_label = to_categorical(label_name, num_classes=51)
                label.append(onehot_label)
                # TODO: make label using video_name

        if not data:
            print("Data is empty!!")

        if not label:
            print('Label is empty!!')

        return np.asarray(data), np.asarray(label)

    def next_batch(self):

        data = []
        label = []
        end_of_file = False

        #print(self._data_list[self._front_idx: self._end_idx])
        for data_list in self._data_list[self._front_idx: self._end_idx]:
            video_tag = data_list.split(' ')[0]
            video_name = video_tag.split('/')[0]
            video_number = int(video_tag.split('/')[1])

            print(data_list)
            print(data_list.split(' ')[-1])
            print('--')

            load_file_name = "%s_%05d_frame.npy" %(video_name, video_number)
            file_root = self._root_dir + '/' + load_file_name

            try:
                dat = np.load(file_root)
                data.append(dat)
                label_name = int(data_list.split(' ')[-1])

                # data.append(self.make_input_shape(file_list, file_root))
                # li.append(label_name)
                onehot_label = to_categorical(label_name, num_classes=51)
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


class Temporal(Spatial):

    def load_all_data(self):
        data = []
        label = []

        for data_list in self._data_list:
            video_tag = data_list.split(' ')[0]
            video_name = video_tag.split('/')[0]
            video_number = int(video_tag.split('/')[1])
            """
            print(data_list)
            print(data_list.split(' ')[-1])
            print('--')
            """
            load_file_name = "%s_%05d_flow.npy" % (video_name, video_number)
            file_root = self._root_dir + '/' + load_file_name

            try:
                dat = np.load(file_root)

            except FileNotFoundError:
                print('pass')
                pass

            else:
                data.append(dat)
                label_name = int(data_list.split(' ')[-1])

                # data.append(self.make_input_shape(file_list, file_root))
                # li.append(label_name)
                onehot_label = to_categorical(label_name, num_classes=51)
                label.append(onehot_label)
                # TODO: make label using video_name

        if not data:
            print("Data is empty!!")

        if not label:
            print('Label is empty!!')

        return np.asarray(data), np.asarray(label)


if __name__=='__main__':

    # HMDB-51 data loader
    root = '/home/jm/Two-stream_data/HMDB51/npy/frame'
    txt_root = '/home/jm/Two-stream_data/HMDB51/train_split1'

    loader = Spatial(root)
    loader.set_data_list(txt_root)
    x, y = loader.load_all_data()
    print(x[0])
    print(x.shape)
    print(y.shape)
    """
    n = 0
    for epoch in range(5):
        while 1:
            x, y, eof = loader.next_batch()

            print(x.shape)
            n += 1

            if eof:
                break
    """







