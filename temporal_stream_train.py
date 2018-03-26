########################################
#     import requirement libraries     #
########################################
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.layers import Merge, Input
from keras.layers.core import Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.backend import set_session
from keras.models import Model
from keras.activations import softmax
from keras.applications.vgg16 import VGG16

from keras.models import model_from_json
from keras.models import load_model
from keras.models import save_model
from keras.utils import np_utils
import scipy.io as sio
import numpy as np
import keras.layers
from keras.utils.np_utils import to_categorical

from keras.utils.vis_utils import model_to_dot
import random
import matplotlib.pyplot as plt

# custom module
import hmdb51

# time stamp
import timeit

# set the quantity of GPU memory consumed
import tensorflow as tf
config = tf.ConfigProto()
# use GPU memory in the available GPU memory capacity
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
set_session(sess)

# If this variable is true, you can use pre-trained model.
using_pretrained_model = False


########################################
#          Set temporal model          #
########################################
def temporal_conv():

    # img_shape = Input(shape=(224, 224, 57))  # TODO: modify data size (ref Two-stream conv paper)
    model = Sequential()

    # conv1 layer
    model.add(Conv2D(96, (7, 7), padding='same', strides=2, input_shape=(224, 224, 20)))
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
    # model.add(Dense(101, activation='softmax'))  # ucf- 101
    model.add(Dense(51, activation='softmax'))
    # model.add(Activation('softmax'))
    # x = softmax()(x)

    model.summary()
    """
    network = Model(input_img, x)
    network.summary()
    """
    return model


if __name__ == '__main__':

    #####################################################
    #     import requirement data using data loader     #
    #####################################################

    # HMDB-51 data loader
    root = '/home/jm/Two-stream_data/HMDB51/npy/flow'
    txt_root = '/home/jm/Two-stream_data/HMDB51/train_split1'

    loader = hmdb51.Temporal(root)
    loader.set_data_list(txt_root)

    print('complete setting data list')

    #####################################################
    #     set convolution neural network structure      #
    #####################################################
    print('set network')
    temporal_stream = temporal_conv()

    print('complete')
    sgd = keras.optimizers.SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
    temporal_stream.compile(optimizer=sgd,  # 'Adam',
                        loss='categorical_crossentropy',
                        metrics=['accuracy'])
    print('complete network setting')

    x,y = loader.load_all_data()
    print(x.shape)

    #TODO:modify parameters
    datagen = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True)

    datagen.fit(x)
    gen = datagen.flow(x, y, batch_size=128)

    for e in range(50000):
        print('Epoch', e)
        batches = 0
        for x_batch, y_batch in gen:
            temporal_stream.fit(x_batch, y_batch)
            batches += 1
            if batches >= len(x) / 128:
                # we need to break the loop by hand because
                # the generator loops indefinitely
                break

        # save model when train 100 epoch
        if e%100 == 0:

            if e == 0:
                continue

            model_json = temporal_stream.to_json()
            json_model_name = "%d_epoch_temporal_model.json" %e
            with open(json_model_name, "w") as json_file:
                json_file.write(model_json)

            weight_name = "%d_epoch_temporal_weight.h5" %e
            model_name = "%d_epoch_temporal_model.h5" %e
            temporal_stream.save_weights(weight_name)
            temporal_stream.save(model_name)
            print("Saved model to disk")



