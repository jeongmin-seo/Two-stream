from keras.models import Sequential, Model
from keras.layers import Dense, Activation
from keras.layers import Flatten, Input
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Dropout
from keras.models import load_model


class ActionNet:
    def __init__(self, _mode, _L=None):
        if _mode == 'spatial':
            self._input_shape = (224, 224, 3)
        elif _mode == 'temporal':
            self._input_shape = (224, 224, _L*2)
        else:
            raise ValueError

    def get_input_shape(self):
        return self._input_shape

    def basic(self):

        # img_shape = Input(shape=(224, 224, 57))  # TODO: modify data size (ref. Two-stream conv paper)
        _model = Sequential()

        # conv1 layer
        _model.add(Conv2D(96, (7, 7), padding='same', strides=2, input_shape=self._input_shape, activation='relu'))
        _model.add(BatchNormalization())
        _model.add(MaxPooling2D((2,2), padding='same'))

        # conv2 layer
        _model.add(Conv2D(256, (5, 5), padding='same', strides=2, activation='relu'))
        _model.add(BatchNormalization())
        _model.add(MaxPooling2D((2, 2), padding='same'))

        # conv3 layer
        _model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))

        # conv4 layer
        _model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))

        # conv5 layer
        _model.add(Conv2D(512, (3,3), padding='same', activation='relu'))
        _model.add(MaxPooling2D((2, 2), padding='same'))

        # full6 layer
        _model.add(Flatten())
        _model.add(Dense(4096))
        _model.add(Dropout(0.5))  # TODO: modify dropout ratio

        # full7 layer
        _model.add(Dense(2048))
        _model.add(Dropout(0.5))  # TODO: modify dropout ratio

        # softamx layer
        _model.add(Dense(51, activation='softmax'))
        _model.summary()

        return _model

    def vgg16(self):
        from keras.applications.vgg16 import VGG16
        vgg16 = VGG16(weights=None, include_top=False, input_shape=self._input_shape)
        #vgg16.layers.pop()
        for layer in vgg16.layers[:14]:
            layer.trainable = False

        vgg16.summary()

        img_input = Input(shape=self._input_shape)
        x = vgg16(img_input)
        x = Flatten(name='flatten')(x)
        x = Dense(250, activation='relu')(x)
        x = Dropout(0.5)(x)

        x = Dense(51, activation='softmax')(x)

        spatial_stream = Model(input=img_input, outputs=x)
        spatial_stream.summary()
        '''
        spatial_stream.add(Dense(1024, activation='relu'))
        spatial_stream.add(Dropout(0.5))
        spatial_stream.add(Dense(256, activation='relu'))
        spatial_stream.add(Dropout(0.5))
        spatial_stream.add(Dense(51, activation='softmax'))
        '''
        return spatial_stream

    def resnet(self):
        from keras.applications.resnet50 import ResNet50
        resnet = ResNet50(include_top=True, weights='imagenet', input_shape=self._input_shape, classes=51)

        return resnet

    @staticmethod
    def set_pretrained_model(_model_path):

        return load_model(_model_path)



#  A  A
# (‘ㅅ‘=)
# J.M.Seo