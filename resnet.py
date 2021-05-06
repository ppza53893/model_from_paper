import os
import warnings

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.simplefilter('ignore')

from typing import Union

import tensorflow as tf
from tensorflow.keras.layers import (Activation, Add, BatchNormalization,
                                     Conv2D, Dense, GlobalAveragePooling2D,
                                     Input, MaxPooling2D, ZeroPadding2D)
from tensorflow.keras.models import Model


class ResNet():
    """ResNet Architecture"""
    __LAYER_LOOPS = {
        '18' : [2,2,2,2],
        '34' : [3,4,6,3],
        '50' : [3,4,6,3],
        '101' : [3,4,23,3],
        '152' : [3,8,36,3]
    }

    def __init__(self, input_ch:tuple, output_ch:int, layers:str):
        try:
            self._range = self.__LAYER_LOOPS[layers]
        except KeyError:
            raise ValueError('ResNet layers must be [18, 34, 50, 101, 152].')

        self.input_ch = input_ch
        self.output_ch = output_ch
        self.layers = layers
        self.index = 0

    def conv_bn_act(self,
        x,
        filters:Union[int, tuple, list],
        kernel_size:Union[int, tuple, list],
        strides:Union[int, tuple, list],
        skip_activation=False,
        **kwargs):
        """
        Conv2D + BatchNormalization + Relu
        """

        x = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, **kwargs)(x)
        x = BatchNormalization()(x)
        if not skip_activation:
            x = Activation('relu')(x)
        return x

    def conv_x(self, x, times, initial_strides=2):
        """conv{n}_x"""
        filters = 64*(2**times)
        for i in range(self._range[self.index]):
            if i==0:
                strides = initial_strides
                y = self.conv_bn_act(x, 4*filters, 1, strides=strides, skip_activation=True)
            else:
                strides = 1
                y = x
            if self.layers == 18 or self.layers == 34:
                x = self.conv_bn_act(x, filters, 3, 2, padding='same')
                x = self.conv_bn_act(x, filters, 3, 1, skip_activation=True, padding='same')
            else:
                x = self.conv_bn_act(x, filters, 1, strides=strides)
                x = self.conv_bn_act(x, filters, 3, 1, padding='same')
                x = self.conv_bn_act(x, 4*filters, 1, 1, skip_activation=True)
            x = Add()([y, x])
            x = Activation('relu')(x)

        self.index += 1
        return x

    def build_resnet(self):
        input = Input(shape=self.input_ch)

        # 7x7 conv, 64, /2
        x = ZeroPadding2D(padding=3)(input)
        x = self.conv_bn_act(x, filters=64, kernel_size=7, strides=(2,2))

        # pool, /2
        x = ZeroPadding2D()(x)
        x = MaxPooling2D(pool_size=(3,3), strides=2)(x)

        # conv2_x
        x = self.conv_x(x, 0, initial_strides=1)

        # conv3_x
        x = self.conv_x(x, 1)

        # conv4_x
        x = self.conv_x(x, 2)

        # conv5_x
        x = self.conv_x(x, 3)

        # average pool, softmax
        x = GlobalAveragePooling2D()(x)
        x = Dense(units = self.output_ch, activation='softmax')(x)

        model = Model(inputs = input, outputs = x)
        return model

        

resnet_50  = ResNet(input_ch = (512, 512, 3), output_ch=1000, layers='50')
model = resnet_50.build_resnet()

model.summary()

