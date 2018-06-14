########################################
#     import requirement libraries     #
########################################
from keras.models import load_model
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score
# custom module
import hmdb51

# project setting
projectPath = '/home/jm/workspace/Two-stream'
nEpoch = 290
eval_type = 'flow'

# data load setting
root = '/home/jm/Two-stream_data/HMDB51'
dataRoot = os.path.join(root, 'npy')
dataPath = os.path.join(dataRoot, eval_type)
splitPath = os.path.join(root, 'test_split1.txt')

def load_eval_model(_project_path, _eval_type, _n_epoch):

    print('start model load')
    if _eval_type == 'rgb':
        _model_name = '%d_epoch_model.h5' % _n_epoch
    elif _eval_type == 'flow':
        _model_name = '%d_epoch_temporal_model.h5' % _n_epoch
    else:
        # TODO: error exception
        pass

    _model_path = os.path.join(_project_path, _eval_type, _model_name)
    _model = load_model(_model_path)

    return _model

if __name__ == '__main__':

    # evaluation model load
    model = load_eval_model(projectPath, eval_type, nEpoch)

    # HMDB-51 data loader
    if eval_type == 'rgb':
        loader = hmdb51.Spatial(dataPath, batch_size=16)
    elif eval_type == 'flow':
        loader = hmdb51.Temporal(dataPath, batch_size=16)
    else:
        #TODO: error exception
        pass

    loader.set_data_list(splitPath)
    print('complete setting data list')

    prediction = []
    target = []
    acc = []
    prec = []
    rec = []
    all = 0
    cor = 0
    while 1:
        batch_x, batch_y, eof = loader.next_batch()
        predict = model.predict_on_batch(batch_x)

        y_true = []
        y_pred = []
        print(predict.shape[0])
        for i in range(predict.shape[0]):
            all += 1
            if batch_y[i].argmax() == predict[i].argmax():
                cor += 1
            #y_true.append(batch_y[i].argmax())
            #y_pred.append(predict[i].argmax())

        #acc.append(accuracy_score(y_true, y_pred))
        #prec.append(precision_score(y_true, y_pred,))
        #rec.append(recall_score(y_true, y_pred))

        del batch_x, batch_y
        if eof:
            break

    print(float(cor/all))
    #print(prec)
    #print(rec)

