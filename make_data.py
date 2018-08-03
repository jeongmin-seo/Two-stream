########################################
#     import requirement libraries     #
########################################
import os
import numpy as np
import random
import cv2
import re
import progressbar
# one hot encoding relate
from keras.utils import to_categorical

saction = ['brush_hair','cartwheel','catch','chew','clap','climb','climb_stairs',
      'dive','draw_sword','dribble','drink','eat','fall_floor','fencing',
      'flic_flac','golf','handstand','hit','hug','jump','kick_ball',
      'kick','kiss','laugh','pick','pour','pullup','punch',
      'push','pushup','ride_bike','ride_horse','run','shake_hands','shoot_ball',
      'shoot_bow','shoot_gun','sit','situp','smile','smoke','somersault',
      'stand','swing_baseball','sword_exercise','sword','talk','throw','turn','walk','wave']

image_size = (224, 224)

def random_cropping(_image, _size):
    row, col, _ = _image.shape
    left, top = random.choice(range(0, row - _size)), random.choice(range(0, col - _size))

    return _image[left:left+_size, top:top+_size]

def horizontal_flip(_image):
    return cv2.flip(_image,1)

class MakePreprocessData():

    def __init__(self, _L, _Nframe):
        self._frames_root = '/home/jm/Two-stream_data/HMDB51/original/frames'
        self._flow_root = '/home/jm/Two-stream_data/HMDB51/original/flow'
        self._save_path = '/home/jm/hdd/preprocess' # '/home/jm/Two-stream_data/HMDB51/npy'
        self._L = _L
        self._Nframe = _Nframe

    @staticmethod
    def split_x_y_disparity(_flow_path):
        result_dict = {'x':[],
                       'y':[]}
        for name in os.listdir(_flow_path):
            if re.split('[_]+', name)[1] == 'x':
                result_dict['x'].append(name)
            else:
                result_dict['y'].append(name)

        result_dict['x'].sort()
        result_dict['y'].sort()
        return result_dict

    def make_temporal_data(self, _flow_path, _action_name, _video_number):
        temporal_save_path = os.path.join(self._save_path, 'flow')
        flow_list = self.split_x_y_disparity(_flow_path)
        numerator = len(flow_list['x']) - self._L + 1
        interval = int(numerator/self._Nframe)

        for n in range(self._Nframe):
            start_frame_num = n * interval
            for i in range(self._L):
                flow_x_name = flow_list['x'][start_frame_num + i]
                flow_y_name = flow_list['y'][start_frame_num + i]

                x = re.split('[_.]+', flow_x_name)[-2]
                y = re.split('[_.]+', flow_y_name)[-2]
                if not(re.split('[_.]+', flow_x_name)[-2] == re.split('[_.]+', flow_y_name)[-2]):
                    print('Not Matched x and y disparity')
                    raise IndexError

                flow_x = cv2.imread(os.path.join(_flow_path, flow_x_name), cv2.IMREAD_GRAYSCALE)
                flow_y = cv2.imread(os.path.join(_flow_path, flow_y_name), cv2.IMREAD_GRAYSCALE)
                resize_flow_x = cv2.resize(flow_x, image_size)
                resize_flow_y = cv2.resize(flow_y, image_size)

                flow = np.dstack([flow_x, flow_y])
                resize_flow = np.dstack([resize_flow_x, resize_flow_y])

                if i == 0:
                    stacked_flow = flow
                    resize_stacked_flow = resize_flow
                    continue

                stacked_flow = np.dstack([stacked_flow, flow])
                resize_stacked_flow = np.dstack([resize_stacked_flow, resize_flow])

            stacked_flow.astype(np.uint8)
            resize_stacked_flow.astype(np.uint8)

            save_name = '%s-%05d-%04d' % (_action_name, int(_video_number), int(start_frame_num+1))
            resized_save_name = save_name + '-original.npy'
            save_path = os.path.join(temporal_save_path, resized_save_name)

            np.save(save_path, resize_stacked_flow)

            if stacked_flow.shape[0] <= image_size[0] or stacked_flow.shape[1] <= image_size[1]:
                return 0

            for i in range(5):
                cropped_img = random_cropping(stacked_flow, 224)
                # flipped_img = horizontal_flip(cropped_img)

                cropped_save_name = save_name + '-cropped-%d.npy' % i
                cropped_save_path = os.path.join(temporal_save_path, cropped_save_name)
                # flipped_save_path = save_name + '-flipped-%d.npy' % i

                np.save(cropped_save_path, cropped_img)

        pass

    def make_spatial_data(self, _frame_path, _action_name, _video_number):
        spatial_save_path = os.path.join(self._save_path, 'frames')
        frame_list = os.listdir(_frame_path)
        interval = int(len(frame_list) / self._Nframe)

        for n in range(self._Nframe):
            img_name = os.path.join(_frame_path, frame_list[n*interval])
            img = cv2.imread(img_name)

            save_name = "%s-%05d-%04d" %(_action_name, int(_video_number), int(n*interval+1))
            save_name = os.path.join(spatial_save_path, save_name)

            # save original image after resize
            resized_img = cv2.resize(img, image_size)
            resized_save_path = save_name + '-original.npy'
            resized_img.astype(np.uint8)
            np.save(resized_save_path, resized_img)

            if img.shape[0] < image_size[0] or img.shape[1] < image_size[1]:
                continue

            # save augmentation image after cropping and flipping
            for i in range(5):
                cropped_img = random_cropping(img, 224)
                # flipped_img = horizontal_flip(cropped_img)

                cropped_save_path = save_name + '-cropped-%d.npy' % i
                # flipped_save_path = save_name + '-flipped-%d.npy' % i

                cropped_img.astype(np.uint8)
                # flipped_img.astype(np.uint8)

                np.save(cropped_save_path, cropped_img)
                # np.save(flipped_save_path, flipped_img)


    def run(self):
        frame_save_root = os.path.join(self._save_path,'frames')
        flow_save_root = os.path.join(self._save_path, 'flow')
        os.makedirs(frame_save_root)
        os.makedirs(flow_save_root)
        action_list = os.listdir(self._flow_root)
        for i in progressbar.progressbar(range(len(action_list))):
            action = action_list[i]
            action_frame_path = os.path.join(self._frames_root, action)
            action_flow_path = os.path.join(self._flow_root, action)

            for video_number in os.listdir(action_flow_path):
                frame_path = os.path.join(action_frame_path, video_number)
                # flow_path = os.path.join(action_flow_path, video_number)

                # run all preprocess procedure
                self.make_spatial_data(frame_path, action, video_number)
                # self.make_temporal_data(flow_path, action, video_number)


if __name__ == '__main__':

    preprocess = MakePreprocessData(10,15)
    preprocess.run()

