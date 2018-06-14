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

# weight initializer

from keras.initializers import glorot_uniform

#########################################################
#                   tensorboard setup                   #
#########################################################
from keras.callbacks import TensorBoard, ModelCheckpoint
tbCallBack = TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)
mcCallBack = ModelCheckpoint('./flow_result/{epoch:0}')

def write_log(callback, names, logs, batch_no):
    for name, value in zip(names, logs):
        summary = tf.Summary()
        summary_value = summary.value.add()
        summary_value.simple_value = value
        summary_value.tag = name
        callback.writer.add_summary(summary, batch_no)
        callback.writer.flush()


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

"""
pytorch code
def cross_modality_pretrain(conv1_weight, channel):
    # transform the original 3 channel weight to "channel" channel
    S=0
    for i in range(3):
        S += conv1_weight[:,i,:,:]
    avg = S/3.
    new_conv1_weight = torch.FloatTensor(64,channel,7,7)
    #print type(avg),type(new_conv1_weight)
    for i in range(channel):
        new_conv1_weight[:,i,:,:] = avg.data
    return new_conv1_weight

def weight_transform(model_dict, pretrain_dict, channel):
    weight_dict  = {k:v for k, v in pretrain_dict.items() if k in model_dict}
    #print pretrain_dict.keys()
    w3 = pretrain_dict['conv1.weight']
    #print type(w3)
    if channel == 3:
        wt = w3
    else:
        wt = cross_modality_pretrain(w3,channel)

    weight_dict['conv1_custom.weight'] = wt
    model_dict.update(weight_dict)
    return model_dict
"""

"""
class AccuracyHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.acc = []

    def on_epoch_end(self, batch, logs={}):
        self.acc.append(logs.get('acc'))


history = AccuracyHistory()
"""
if __name__ == '__main__':

    #####################################################
    #     import requirement data using data loader     #
    #####################################################

    # HMDB-51 data loader
    root = '/home/jm/Two-stream_data/HMDB51/npy/flow'
    txt_root = '/home/jm/Two-stream_data/HMDB51/train_split1'

    loader = hmdb51.Temporal(root,batch_size=16)
    loader.set_data_list(txt_root)

    print('complete setting data list')

    #####################################################
    #     set convolution neural network structure      #
    #####################################################
    print('set network')
    # temporal_stream = temporal_conv()
    vgg16 = VGG16(include_top=False, input_shape=(224, 224, 20),weights=None)

    vgg16.summary()
    #print(vgg16.get_weights()[1].shape)
    #print(vgg16.get_weights()[2].shape)



    img_input = Input(shape=(224, 224, 20))
    x = vgg16(img_input)
    x = Flatten(name='flatten')(x)
    x = Dense(5000, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(1000, activation='relu')(x)
    x = Dropout(0.5)(x)

    x = Dense(51, activation='softmax')(x)

    temporal_stream = Model(input=img_input, outputs=x)
    temporal_stream.summary()

    print('complete')
    sgd = keras.optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    temporal_stream.compile(optimizer=sgd,  # 'Adam',
                        loss='categorical_crossentropy',
                        metrics=['accuracy'])
    print('complete network setting')


    tbCallBack.set_model(temporal_stream)
    for epoch in range(500):
        print('Epoch',epoch)

        # reset batch
        loader.shuffle()
        loader.get_data_list()
        batch_num = 0
        loss_list = []
        acc_list = []
        while 1:
            batch_x, batch_y, eof = loader.next_batch()
            batch_log = temporal_stream.train_on_batch(batch_x,batch_y)
            loss_list.append(batch_log[0])
            acc_list.append(batch_log[1])
            batch_num += 1
            #temporal_stream.fit(batch_x, batch_y, batch_size=None,verbose=1,callbacks=[tbCallBack])

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

            model_json = temporal_stream.to_json()
            json_model_name = "./flow_result/%d_epoch_temporal_model.json" % epoch
            with open(json_model_name, "w") as json_file:
                json_file.write(model_json)

            weight_name = "./flow_result/%d_epoch_temporal_weight.h5" % epoch
            model_name = "./flow_result/%d_epoch_temporal_model.h5" % epoch
            temporal_stream.save_weights(weight_name)
            temporal_stream.save(model_name)
            print("Saved model to disk")



    """
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
    
    """

