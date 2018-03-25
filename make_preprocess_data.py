########################################
#     import requirement libraries     #
########################################
import os
import numpy as np
import random
import cv2
import glob
import re

# one hot encoding relate
from keras.utils import to_categorical

saction = ['brush_hair','cartwheel','catch','chew','clap','climb','climb_stairs',
      'dive','draw_sword','dribble','drink','eat','fall_floor','fencing',
      'flic_flac','golf','handstand','hit','hug','jump','kick_ball',
      'kick','kiss','laugh','pick','pour','pullup','punch',
      'push','pushup','ride_bike','ride_horse','run','shake_hands','shoot_ball',
      'shoot_bow','shoot_gun','sit','situp','smile','smoke','somersault',
      'stand','swing_baseball','sword_exercise','sword','talk','throw','turn','walk','wave']

class MakePreprocessData():

    def __init__(self, _L, _bgr_min_max, _flow_min_max):
        self._frames_root = '/home/jm/Two-stream_data/HMDB51/frames'
        self._flow_root = '/home/jm/Two-stream_data/HMDB51/flow'
        self._save_path = '/home/jm/Two-stream_data/HMDB51/npy'
        self._L = _L
        self._bgr_min_max = _bgr_min_max
        self._flow_min_max = _flow_min_max


    @staticmethod
    def extract_start_frame(_frame_path):

        # extract start frame number using number of still image frame
        frame_list = sorted(os.listdir(_frame_path))[0:-11]
        frame_name = random.choice(frame_list)
        frame_number = int(re.split('[_.]', frame_name)[1])

        return frame_name, frame_number

    def select_flow_frame(self,_flow_path, _start_num):

        # To make flow data format, select start frame to L'th flow frame
        flow_all_list = os.listdir(_flow_path)
        flow_list = {'x':[], 'y':[]}

        # separate x and y disparity images
        for flow in flow_all_list:
            if re.split('[_.]', flow)[1] == 'x':
                flow_list['x'].append(flow)

            elif re.split('[_.]', flow)[1] == 'y':
                flow_list['y'].append(flow)

            _flow_x_list = sorted(flow_list['x'])
            _flow_y_list = sorted(flow_list['y'])

        return _flow_x_list[_start_num: _start_num+self._L], _flow_y_list[_start_num: _start_num+self._L]

    @staticmethod
    def make_temporal_data(_flow_path, _flow_x_list, _flow_y_list):

        assert len(_flow_x_list) == len(_flow_y_list)

        for idx in range(len(_flow_x_list)):
            x_name = _flow_path + '/' + _flow_x_list[idx]
            y_name = _flow_path + '/' + _flow_y_list[idx]

            opt_x = cv2.imread(x_name, cv2.IMREAD_GRAYSCALE)
            opt_y = cv2.imread(y_name, cv2.IMREAD_GRAYSCALE)

            resized_opt_x = cv2.resize(opt_x, (224, 224))
            resized_opt_y = cv2.resize(opt_y, (224, 224))

            normalized_opt_x = resized_opt_x/255        # 255 is max pixel value relate normalize
            normalized_opt_y = resized_opt_y/255

            if idx == 0:
                stacked_opt = np.dstack([normalized_opt_x, normalized_opt_y])
                continue

            stacked_opt = np.dstack([stacked_opt, normalized_opt_x, normalized_opt_y])

        return stacked_opt.astype(np.uint8)

    @staticmethod
    def make_label(_action):

        assert _action in saction

        label = saction.index(_action)
        categorical_label = to_categorical(label, num_classes=51)

        return categorical_label.astype(np.uint8)

    @staticmethod
    def make_spatial_data(_frame_path, _frame_name):
        img_name = _frame_path + '/' + _frame_name

        img = cv2.imread(img_name)
        resized_img = cv2.resize(img, (224, 224))   # (224, 224, 3) is image size in paper
        normalized_img = resized_img/255    # 255 is max pixel value relate normalize

        return normalized_img.astype(np.uint8)

    def run(self):

        for action in os.listdir(self._flow_root):
            action_frame_path = self._frames_root + '/' + action
            action_flow_path = self._flow_root + '/' + action

            for video_number in os.listdir(action_flow_path):
                frame_path = action_frame_path + '/' + video_number
                flow_path = action_flow_path + '/' + video_number

                # TODO: add extract_start_frame function
                frame_name, start_frame_num = self.extract_start_frame(frame_path)
                flow_x_list, flow_y_list = self.select_flow_frame(flow_path, start_frame_num)

                #run all preprocess procedure
                result_frame = self.make_spatial_data(frame_path, frame_name)
                result_flow = self.make_temporal_data(flow_path, flow_x_list, flow_y_list)
                result_label = self.make_label(action)

                #save_action = action.replace('_', ' ')

                save_name = "%s_%05d" %(action, int(video_number))

                if result_flow.shape[2] != 20:
                    print(save_name)
                    continue

                save_flow = self._save_path + '/flow/' + save_name + '_unidirctional_flow.npy'  #how to fix!!
                save_frame = self._save_path + '/frame/' + save_name + '_frame.npy'
                save_label = self._save_path + '/label/' + save_name + '_label.npy'

                np.save(save_flow, result_flow)
                np.save(save_frame, result_frame)
                np.save(save_label, result_label)



if __name__ == '__main__':

    bgr_min_max = {'b': [0, 255],
                   'g': [0, 255],
                   'r': [0, 255]}
    flow_min_max = {'x': [0, 255],
                    'y': [0, 255]}


    preprocess = MakePreprocessData(10, bgr_min_max, flow_min_max)

    preprocess.run()

