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

    img_shape = Input(shape=(448, 448, 1))  # TODO: modify data size (ref Two-stream conv paper)
    model = Sequential()

    # conv1 layer
    model.add(Conv2D(96, (7, 7), padding='same', strides=2, input_shape=img_shape))
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
    model.add(Activation('softmax'))
    # x = softmax()(x)

    model.summary()
    """
    network = Model(input_img, x)
    network.summary()
    """
    return model
