########################################
#     import requirement libraries     #
########################################
import os
import numpy as np
import cv2
import re

bgr_min_max = {'b':[255, 0],
               'g':[255, 0],
               'r':[255, 0]}
flow_min_max = {'x':[],
                'y':[]}

frames_root = '/home/jeongmin/workspace/data/HMDB51/frames'
flow_root = '/home/jeongmin/workspace/data/HMDB51/flow'
"""
action1 = {}
for action in os.listdir(frames_root):
    video_root = frames_root + '/' + action
    action1[action] = len(os.listdir(video_root))

action2 = {}
for action in os.listdir(flow_root):
    video_root = flow_root + '/' + action
    action2[action] = len(os.listdir(video_root))

num = 0
for key in action1.keys():

    if action1[key] != action2[key]:
        num += 1
        print(key)
        print('frame: %d  flow: %d' % (action1[key] ,action2[key]))

print(num)
"""
num = 0
for action in os.listdir(flow_root):
    action_frame_path = frames_root + '/' + action
    action_flow_path = flow_root + '/' + action

    for video_number in os.listdir(action_flow_path):
        frame_path = action_frame_path + '/' + video_number
        flow_path = action_flow_path + '/' + video_number

        for frame in os.listdir(frame_path):
            frame_name = frame_path + '/' + frame
            img = cv2.imread(frame_name)

            img_b, img_g, img_r = cv2.split(img)

            b_min = img_b.min()
            b_max = img_b.max()

            g_min = img_g.min()
            g_max = img_g.max()

            r_min = img_r.min()
            r_max = img_r.max()

            if bgr_min_max['b'][0] > b_min:
                bgr_min_max['b'][0] = b_min

            if bgr_min_max['b'][1] < b_max:
                bgr_min_max['b'][1] = b_max

            if bgr_min_max['g'][0] > b_min:
                bgr_min_max['g'][0] = b_min

            if bgr_min_max['g'][1] < b_max:
                bgr_min_max['g'][1] = b_max

            if bgr_min_max['r'][0] > b_min:
                bgr_min_max['r'][0] = b_min

            if bgr_min_max['r'][1] < b_max:
                bgr_min_max['r'][1] = b_max




        for flow in os.listdir(flow_path):
            flow_name = flow_path + '/' + flow
            img = cv2.imread(flow_name)

            flow_min = img.min()
            flow_max = img.max()

            if re.split('[_.]', flow)[1] == 'x':
                if not flow_min_max['x']:
                    flow_min_max['x'].append(flow_min)
                    flow_min_max['x'].append(flow_max)
                    continue

                else:
                    if flow_min_max['x'][0] > flow_min:
                        flow_min_max['x'][0] = flow_min

                    if flow_min_max['x'][1] < flow_max:
                        flow_min_max['x'][1] = flow_max


            elif re.split('[_.]', flow)[1] == 'y':
                if not flow_min_max['y']:
                    flow_min_max['y'].append(flow_min)
                    flow_min_max['y'].append(flow_max)

                else:
                    if flow_min_max['y'][0] > flow_min:
                        flow_min_max['y'][0] = flow_min

                    if flow_min_max['y'][1] < flow_max:
                        flow_min_max['y'][1] = flow_max

        num += 1
        print("%d / %d" %(num, len(os.listdir(action_flow_path))))


print("#"*100)
print(bgr_min_max)
print('-'*100)
print(flow_min_max)

