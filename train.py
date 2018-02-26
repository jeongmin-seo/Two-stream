########################################
#     import requirement libraries     #
########################################
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers import Merge, Input
from keras.layers.core import Dropout
from keras.backend import set_session
from keras.models import Model
from keras.activations import softmax

from keras.models import model_from_json
from keras.utils import np_utils
import scipy.io as sio
import numpy as np
import keras.layers
from keras.utils.np_utils import to_categorical

from keras.utils.vis_utils import model_to_dot
import random
import matplotlib.pyplot as plt

# custom module
import data_loader

# time stamp
import timeit

# set the quantity of GPU memory consumed
import tensorflow as tf
config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.85
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
set_session(sess)



########################################
#           Set stream model           #
########################################
def stream_conv():

    # img_shape = Input(shape=(224, 224, 57))  # TODO: modify data size (ref Two-stream conv paper)
    model = Sequential()

    # conv1 layer
    model.add(Conv2D(96, (7, 7), padding='same', strides=2, input_shape=(224, 224, 57)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2,2), padding='same'))
    """
    x = Conv2D(96, (7, 7), padding='same', strides=2)(img_shape)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    """

    # conv2 layer
    model.add(Conv2D(256, (5, 5), padding='same', strides=2))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2), padding='same'))
    """
    x = Conv2D(256, (5, 5), padding='same', strides=2)(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    """

    # conv3 layer
    model.add(Conv2D(512, (3, 3), padding='same'))
    # x = Conv2D(512, (3, 3), padding='same')(x)

    # conv4 layer
    model.add(Conv2D(512, (3, 3), padding='same'))
    # x = Conv2D(512, (3, 3), padding='same')(x)

    # conv5 layer
    model.add(Conv2D(512, (3,3), padding='same'))
    model.add(MaxPooling2D((2, 2), padding='same'))
    """
    x = Conv2D(512, (3,3), padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    """

    # full6 layer
    model.add(Flatten())
    model.add(Dense(4096))
    model.add(Dropout(0.5))  # TODO: modify dropout ratio
    """
    x = Flatten()(x)
    x = Dense(4096)(x)
    x = Dropout(0.5)(x)  # TODO: modify dropout ratio
    """

    # full7 layer
    model.add(Dense(2048))
    model.add(Dropout(0.5))  # TODO: modify dropout ratio
    """
    x = Dense(2048)(x)
    x = Dropout(0.5)(x)  # TODO: modify dropout ratio
    """

    # softamx layer
    model.add(Dense(101, activation='softmax'))
    # model.add(Activation('softmax'))
    # x = softmax()(x)

    model.summary()
    """
    network = Model(input_img, x)
    network.summary()
    """
    return model


if __name__=='__main__':

    #####################################################
    #     import requirement data using data loader     #
    #####################################################
    root = '/home/jm/Two-stream_data/jpegs_256/'
    txt_root = '/home/jm/Two-stream_data/trainlist01.txt'
    loader = data_loader.DataLoader(root, batch_size=640)
    loader.set_data_list(txt_root)
    print('complete setting data list')

    #####################################################
    #     set convolution neural network structure      #
    #####################################################
    print('set network')
    spatial_stream = stream_conv()
    # temporal_stream = stream_conv()
    print('complete')

    spatial_opti = keras.optimizers.Adam(lr=1e-2, beta_1=0.99,
                                     beta_2=0.99, epsilon=1e-08, decay=1e-4)

    spatial_stream.compile(optimizer=spatial_opti,  # 'Adam',
                        loss='categorical_crossentropy',
                        metrics=['accuracy'])
    print('complete network setting')


    for epoch in range(1,101):
        start = timeit.default_timer()

        print('%d epoch train start' % epoch)
        loader.shuffle()    # shuffle data set
        while 1:
            x, y, eof = loader.next_batch()
            print(x.shape)
            spatial_stream.fit(x, y, verbose=1)

            del x, y
            if eof:
                break

        stop = timeit.default_timer()
        print(stop-start)

        print('=' * 50)
        print('%d epoch end' % epoch)

    print('*'*50)
    print('complete train')

    model_json = spatial_stream.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    spatial_stream.save_weights("model.h5")
    print("Saved model to disk")


