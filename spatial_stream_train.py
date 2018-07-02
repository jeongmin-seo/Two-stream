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
import os
import progressbar

# custom module
import ucf101
import hmdb51


# set the quantity of GPU memory consumed
import tensorflow as tf
config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.85
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
set_session(sess)

# using pretrained model
pretrained_model_name = '20_epoch_temporal_model.h5'
using_pretrained_model = True
save_model_path = '/home/jm/workspace/Two-stream/frame_model'
num_epoch = 100
batch_size = 64

#########################################################
#                   tensorboard setup                   #
#########################################################
from keras.callbacks import TensorBoard, ModelCheckpoint
tbCallBack = TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)
# mcCallBack = ModelCheckpoint('./flow_result/{epoch:0}', monitor='val_loss',
#                              verbose=1, save_best_only=True)
# mcCallBack.

def write_log(callback, names, logs, batch_no):
    for name, value in zip(names, logs):
        summary = tf.Summary()
        summary_value = summary.value.add()
        summary_value.simple_value = value
        summary_value.tag = name
        callback.writer.add_summary(summary, batch_no)
        callback.writer.flush()

########################################
#           Set stream model           #
########################################
def stream_conv():

    # img_shape = Input(shape=(224, 224, 57))  # TODO: modify data size (ref. Two-stream conv paper)
    model = Sequential()

    # conv1 layer
    model.add(Conv2D(96, (7, 7), padding='same', strides=2, input_shape=(224, 224, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2,2), padding='same'))

    # conv2 layer
    model.add(Conv2D(256, (5, 5), padding='same', strides=2, activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2), padding='same'))

    # conv3 layer
    model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))

    # conv4 layer
    model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))

    # conv5 layer
    model.add(Conv2D(512, (3,3), padding='same', activation='relu'))
    model.add(MaxPooling2D((2, 2), padding='same'))

    # full6 layer
    model.add(Flatten())
    model.add(Dense(4096))
    model.add(Dropout(0.5))  # TODO: modify dropout ratio

    # full7 layer
    model.add(Dense(2048))
    model.add(Dropout(0.5))  # TODO: modify dropout ratio


    # softamx layer
    # model.add(Dense(101, activation='softmax'))  # ucf- 101
    model.add(Dense(51, activation='softmax'))
    model.summary()

    return model

if __name__ == '__main__':

    #####################################################
    #     import requirement data using data loader     #
    #####################################################
    """
    # UCF-101 data load
    root = '/home/jm/Two-stream_data/jpegs_256/'
    txt_root = '/home/jm/Two-stream_data/trainlist01.txt'
    loader = ucf101.Spatial(root, batch_size=640)
    loader.set_data_list(txt_root)
    """
    # HMDB-51 data loader
    root = '/home/jm/Two-stream_data/HMDB51/npy/frame'
    txt_root = '/home/jm/Two-stream_data/HMDB51/train_split1.txt'

    loader = hmdb51.Spatial(root, batch_size=batch_size)
    loader.set_data_list(txt_root)

    print('complete setting data list')

    #####################################################
    #     set convolution neural network structure      #
    #####################################################
    if using_pretrained_model:
        start_epoch_num = int(pretrained_model_name.split('_')[0]) + 1
        load_model_path = os.path.join(save_model_path, pretrained_model_name)
        spatial_stream = load_model(load_model_path)
        # spatial_stream.load_weights("/home/jm/workspace/Two-stream/hmdb_spatial_model.h5")
        print("weight loaded")

    else:
        start_epoch_num = 0
        spatial_stream = stream_conv()
        print('set network')

    """
    #weight_path = "/home/jm/workspace/Two-stream/pre-trained_model/model/vgg16_weights.h5"
    vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(224,224,3))
    #vgg16.layers.pop()
    for layer in vgg16.layers[:14]:
        layer.trainable = False

    vgg16.summary()

    img_input = Input(shape=(224,224,3))
    x = vgg16(img_input)
    x = Flatten(name='flatten')(x)
    x = Dense(250, activation='relu')(x)
    x = Dropout(0.5)(x)

    x = Dense(51, activation='softmax')(x)

    spatial_stream = Model(input=img_input, outputs=x)
    spatial_stream.summary()
    """


    """
    spatial_stream.add(Dense(1024, activation='relu'))
    spatial_stream.add(Dropout(0.5))
    spatial_stream.add(Dense(256, activation='relu'))
    spatial_stream.add(Dropout(0.5))
    spatial_stream.add(Dense(51, activation='softmax'))
    """

    print('complete')
    sgd = keras.optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    # adam = keras.optimizers.Adam(lr=1e-3, beta_1=0.99,
    #                                  beta_2=0.99, epsilon=1e-08, decay=1e-4)

    spatial_stream.compile(optimizer=sgd,
                        loss='categorical_crossentropy',
                        metrics=['accuracy'])
    print('complete network setting')


    """
    for e in range(50000):
        print('Epoch', e)
        batches = 0
        for x_batch, y_batch in gen:
            spatial_stream.fit(x_batch, y_batch)
            batches += 1
            if batches >= len(x) / 128:
                # we need to break the loop by hand because
                # the generator loops indefinitely
                break

        if e%100 == 0:

            if e == 0:
                continue

            model_json = spatial_stream.to_json()
            json_model_name = "%d_epoch_model.json" %e
            with open(json_model_name, "w") as json_file:
                json_file.write(model_json)

            weight_name = "%d_epoch_weight.h5" %e
            model_name = "%d_epoch_model.h5" %e
            spatial_stream.save_weights(weight_name)
            spatial_stream.save(model_name)
            print("Saved model to disk")
    """
    tmp_numiter = len(loader.get_data_list())/batch_size
    num_iter = int(tmp_numiter)+1 if tmp_numiter - int(tmp_numiter) > 0 else int(tmp_numiter)
    tbCallBack.set_model(spatial_stream)
    for epoch in range(start_epoch_num, start_epoch_num + num_epoch):
        print('Epoch', epoch)

        # reset batch
        loader.shuffle()
        # loader.get_data_list()
        loss_list = []
        acc_list = []
        for i in progressbar.progressbar(range(num_iter)):
        # while 1:
            batch_x, batch_y, eof = loader.next_batch()
            batch_log = spatial_stream.train_on_batch(batch_x, batch_y)
            loss_list.append(batch_log[0])
            acc_list.append(batch_log[1])

            del batch_x, batch_y
            if eof:
                break

        avg_loss = np.mean(loss_list)
        avg_acc = np.mean(acc_list)
        print("loss:", avg_loss, "acc:", avg_acc)
        write_log(tbCallBack, ["train_loss", "train_acc"], [avg_loss, avg_acc], epoch)

        if epoch % 10 == 0:
            if epoch == 0:
                continue

            model_name = "./frame_model/%d_epoch_temporal_model.h5" % epoch
            spatial_stream.save(model_name)
            print("Saved model to disk")

