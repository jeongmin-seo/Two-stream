########################################
#     import requirement libraries     #
########################################
import os
import numpy as np
import random
import cv2
import glob

# one hot encoding relate
from keras.utils import to_categorical


class MakePreprocessData():

    def __init__(self, L):
        self._frames_root = '/home/jm/Two-stream_data/HMDB51/frames'
        self._flow_root = '/home/jm/Two-stream_data/HMDB51/flow'
        self._L = L

    def extract_start_frame(self):

        #TODO: extract start frame number using number of still image frame
        pass

    def run(self):

        for action in os.listdir(self._frames_root):
            action_frame_path = self._frames_root + '/' + action
            action_flow_path = self._flow_root + '/' + action

            for video_number in os.listdir(action_frame_path):
                frame_path = action_frame_path + '/' + video_number

                # TODO: add extract_start_frame function

                for frame_list in os.listdir(frame_path):
                    image_name = frame_path + '/' + frame_list
                    img = cv2.imread(image_name)
        #run all preprocess procedure
        pass


"""
def make_flow_data():
    data_path = data_root + '/' + 'flow'

    for action in os.listdir(data_path):
        action_path = data_path + '/' + action

        for video_number in os.listdir(action_path):
            frame_path = action_path + '/' + video_number

            for frame_list in os.listdir(frame_path):
"""




if __name__ == '__main__':

    MakePreprocessData(10)


