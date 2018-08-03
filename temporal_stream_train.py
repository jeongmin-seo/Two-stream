########################################
#     import requirement libraries     #
########################################
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Dropout
from keras.backend import set_session
from keras.models import load_model
import numpy as np
import keras.layers
import os
import progressbar

# custom module
from sklearn.metrics import log_loss
import hmdb51
import data_loader

# set the quantity of GPU memory consumed
import tensorflow as tf
config = tf.ConfigProto()
# use GPU memory in the available GPU memory capacity
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
set_session(sess)

# using pretrained model
pretrained_model_name = ''
using_pretrained_model = False
save_model_path = '/home/jm/workspace/Two-stream/flow_model'
num_epoch = 100
batch_size = 64


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


########################################
#          Set temporal model          #
########################################
def temporal_conv():

    # img_shape = Input(shape=(224, 224, 57))  # TODO: modify data size (ref Two-stream conv paper)
    model = Sequential()

    # conv1 layer
    model.add(Conv2D(96, (7, 7), padding='same', strides=2, input_shape=(224, 224, 20), activation='relu'))
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

def vallidation_1epoch(_model, _loader):
    loss_list = []
    # acc_list = []
    correct = 0
    _loader.set_test_video_list()

    for i in progressbar.progressbar(range(len(_loader.get_test_data_list()))):
        _batch_x, _batch_y, eof = _loader.next_test_video()
        result = _model.predict_on_batch(_batch_x)

        label = np.sum(_batch_y,axis=0)
        predict = np.sum(result, axis=0)
        loss_list.append(float(log_loss(label, predict)))
        if label.argmax() == predict.argmax():
            correct += 1
        # keras.metrics.categorical_crossentropy()

    return float(correct/len(loss_list)), np.asarray(loss_list).mean()

def train_1epoch(_model, _loader, _num_iter):
    # reset batch
    _loader.train_data_shuffle()
    loss_list = []
    acc_list = []
    for i in progressbar.progressbar(range(_num_iter)):
        _batch_x, _batch_y, _eof = _loader.next_train_batch()
        _batch_log = _model.train_on_batch(_batch_x, _batch_y)
        loss_list.append(_batch_log[0])
        acc_list.append(_batch_log[1])

        del _batch_x, _batch_y
        if _eof:
            break

    return np.mean(acc_list), np.mean(loss_list)

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
    root = '/home/jeongmin/workspace/data/HMDB51/preprocess/flow'
    train_txt_root = '/home/jeongmin/workspace/data/HMDB51/train_split1.txt'
    test_txt_root = '/home/jeongmin/workspace/data/HMDB51/test_split1.txt'

    train_loader = data_loader.DataLoader(root, batch_size=batch_size)
    train_loader.set_data_list(train_txt_root, train_test_type='train')

    test_loader = data_loader.DataLoader(root)
    test_loader.set_data_list(test_txt_root, train_test_type='test')

    print('complete setting data list')

    #####################################################
    #     set convolution neural network structure      #
    #####################################################
    if using_pretrained_model:
        start_epoch_num = int(pretrained_model_name.split('_')[0]) + 1
        load_model_path = os.path.join(save_model_path, pretrained_model_name)
        temporal_stream = load_model(load_model_path)
        print("weight loaded")

    else:
        start_epoch_num = 0
        temporal_stream = temporal_conv()
        print('set network')

    """
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
    """

    print('complete')
    sgd = keras.optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    temporal_stream.compile(optimizer=sgd,  # 'Adam',
                        loss='categorical_crossentropy',
                        metrics=['accuracy'])
    print('complete network setting')

    tmp_numiter = len(train_loader.get_train_data_list())/batch_size
    num_iter = int(tmp_numiter)+1 if tmp_numiter - int(tmp_numiter) > 0 else int(tmp_numiter)
    tbCallBack.set_model(temporal_stream)
    for epoch in range(start_epoch_num, start_epoch_num + num_epoch):
        print('Epoch',epoch)

        print('Epoch', epoch)

        train_acc, train_loss = train_1epoch(temporal_stream, train_loader, num_iter)
        print("train_loss:", train_loss, "train_acc:", train_acc)

        val_acc, val_loss = vallidation_1epoch(temporal_stream, test_loader)
        print("val_loss:", val_loss, "val_acc:", val_acc)

        write_log(tbCallBack, ["train_loss", "train_acc", 'validation_loss', 'validation_acc'],
                  [train_loss, train_acc, val_loss, val_acc], epoch)

        if epoch % 5 == 0:
            if epoch == 0:
                continue

            model_name = "./frame_model/%d_epoch_spatial_model.h5" % epoch
            temporal_stream.save(model_name)
            print("Saved model to disk")
