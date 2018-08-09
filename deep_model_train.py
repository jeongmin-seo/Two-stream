########################################
#     import requirement libraries     #
########################################
from keras.backend import set_session
from keras.optimizers import SGD
import os
import argparse

# custom module
import data_loader
import network
from util import write_log, train_1epoch, validation_1epoch, save_best_model

# set the quantity of GPU memory consumed
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
set_session(sess)

parser = argparse.ArgumentParser()
parser.add_argument("--mode", help="select data type, spatial or temporal", type=str, default='spatial')
parser.add_argument("--batch_size", help="mini-batch size (default: 64)", type=int, default=64)
parser.add_argument("--lr", default=1e-2, type=float, help="learning rate")
parser.add_argument("--epoch", default=100, type=int, help="number of epoch")
parser.add_argument("--model_name", type=str, help="select model")

# using pretrained model
pretrained_model_name = '20_epoch_temporal_model.h5'
using_pretrained_model = False
save_model_path = '/home/mlpa/Workspace/github/Two-stream/flow_vgg16'
# num_epoch = 100
# batch_size = 128

# model name
# stream_mode = 'spatial'  # 'temporal'
# model_name = 'resnet'  # 'vgg16'


#########################################################
#                     CallBack  setup                   #
#########################################################
from keras.callbacks import TensorBoard, ModelCheckpoint
tbCallBack = TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)
mcCallBack = ModelCheckpoint('./flow_result/{epoch:0}', monitor='val_loss',
                             verbose=1, save_best_only=True)


if __name__ == '__main__':

    global arg
    arg = parser.parse_args()

    #####################################################
    #     import requirement data using data loader     #
    #####################################################
    # HMDB-51 data loader
    root = '/home/mlpa/ssd/HMDB51/preprocess/flow'
    train_txt_root = '/home/mlpa/ssd/HMDB51/preprocess/train_split1.txt'
    test_txt_root = '/home/mlpa/ssd/HMDB51/preprocess/test_split1.txt'

    train_loader = data_loader.DataLoader(root, batch_size=arg.batch_size)
    train_loader.set_data_list(train_txt_root, train_test_type='train')

    train_val_loader = data_loader.DataLoader(root, batch_size=arg.batch_size)
    train_val_loader.set_data_list(train_txt_root, train_test_type='test')

    test_loader = data_loader.DataLoader(root)
    test_loader.set_data_list(test_txt_root, train_test_type='test')

    print('complete setting data list')

    L=0
    if arg.mode == 'spatial':
        L = None
    elif arg.mode == 'temporal':
        L = 5
    else:
        raise ValueError

    net = network.ActionNet(_mode=arg.mode, _L=L)
    #####################################################
    #     set convolution neural network structure      #
    #####################################################
    if using_pretrained_model:
        start_epoch_num = int(pretrained_model_name.split('_')[0]) + 1
        load_model_path = os.path.join(save_model_path, pretrained_model_name)
        model = net.set_pretrained_model(load_model_path)

    else:
        start_epoch_num = 0
        if 'vgg16' == arg.model_name:
            model = net.vgg16()
        elif 'resnet' == arg.model_name:
            model = net.resnet()
        else:
            print('No Model in Action network.')
            raise ValueError
        print('set network')

    print('complete')
    sgd = SGD(lr=arg.lr, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
    print('complete network setting')

    tmp_numiter = len(train_loader.get_train_data_list())/arg.batch_size
    num_iter = int(tmp_numiter)+1 if tmp_numiter - int(tmp_numiter) > 0 else int(tmp_numiter)
    tbCallBack.set_model(model)
    mcCallBack.set_model(model)

    loss_session = tf.Session()
    best_val_acc = 0
    for epoch in range(start_epoch_num, start_epoch_num + arg.epoch):
        print('Epoch', epoch)

        train_acc, train_loss = train_1epoch(model, train_loader, num_iter)
        print("train_loss:", train_loss, "train_acc:", train_acc)

        tr_val_acc, tr_val_loss = validation_1epoch(model, train_val_loader, sess)
        print("tr_val_loss:", tr_val_loss, "tr_val_acc:", tr_val_acc)

        val_acc, val_loss = validation_1epoch(model, test_loader, sess)
        print("val_loss:", val_loss, "val_acc:", val_acc)

        write_log(tbCallBack,
                  ["train_loss", "train_acc", 'validation_loss', 'validation_acc', 'tr_val_loss', 'tr_val_acc'],
                  [train_loss, train_acc, val_loss, val_acc, tr_val_loss, tr_val_acc],
                  epoch)

        best_val_acc = save_best_model(epoch, val_acc, best_val_acc, model, save_model_path)

        loss_list = []
        loss2_list = []
        correct = 0
        test_loader.set_test_video_list()


#  A  A
# (‘ㅅ‘=)
# J.M.Seo
