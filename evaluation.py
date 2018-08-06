########################################
#     import requirement libraries     #
########################################
from keras.models import load_model
import os
import progressbar
import data_loader
import hmdb51

# project setting
projectPath = '/home/jeongmin/workspace/github/Two-stream'
nEpoch = 210
eval_type = 'flow'

# data load setting
root = '/home/jeongmin/workspace/data/HMDB51/'
dataRoot = os.path.join(root, 'npy')
dataPath = os.path.join(dataRoot, eval_type)
splitPath = os.path.join(root, 'test_split1.txt')
batch_size = 16

# TODO: extract Class AP information


def load_eval_model(_project_path, _eval_type, _n_epoch):

    print('start model load')
    if _eval_type == 'frame':
        _model_name = '%d_epoch_spatial_model.h5' % _n_epoch
    elif _eval_type == 'flow':
        _model_name = '%d_epoch_temporal_model.h5' % _n_epoch
    else:
        # TODO: error exception
        pass

    model_dir_name = _eval_type + '_model'
    _model_path = os.path.join(_project_path, model_dir_name, _model_name)
    _model = load_model(_model_path)

    return _model


if __name__ == '__main__':

    # evaluation model load
    model = load_eval_model(projectPath, eval_type, nEpoch)

    # HMDB-51 data loader
    if eval_type == 'frame' or 'flow':
        loader = data_loader.DataLoader(root, batch_size=batch_size)
        loader.set_data_list(splitPath, train_test_type='test')
    else:
        #TODO: error exception
        pass


    prediction = []
    target = []
    # acc = []
    # prec = []
    # rec = []
    all = 0
    cor = 0
    tmp_numiter = len(loader.get_test_data_list()) / batch_size
    num_iter = int(tmp_numiter) + 1 if tmp_numiter - int(tmp_numiter) > 0 else int(tmp_numiter)
    for i in progressbar.progressbar(range(num_iter)):

        batch_x, batch_y, eof = loader.next_test_video()
        predict = model.predict_on_batch(batch_x)
        y_true = []
        y_pred = []

        # print(predict.shape[0])
        for i in range(predict.shape[0]):
            all += 1
            # print(batch_y[i])
            # print(predict[i])
            if batch_y[i].argmax() == predict[i].argmax():
                cor += 1

        del batch_x, batch_y
        if eof:
            break

    print(float(cor/all))
