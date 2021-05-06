import collections
import dataclasses
import math
import os
import warnings

os.environ['SM_FRAMEWORK'] = 'tf.keras'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.simplefilter('ignore')

from typing import Union

import segmentation_models as sm
import tensorflow as tf
from tensorflow.keras.layers import (Activation, BatchNormalization,
                                     Concatenate, Conv2D, Conv2DTranspose,
                                     Cropping2D, Input, MaxPooling2D,
                                     ZeroPadding2D)
from tensorflow.keras.models import Model


@dataclasses.dataclass
class Unet():
    """Unet Architecture"""
    input_shape : tuple = (572, 572, 3)
    conv_padding : str = 'valid'
    output_ch : int = 2

    def conv_bn_act(self,
        x,
        filters:Union[int, tuple, list],
        kernel_size:Union[int, tuple, list],
        strides:Union[int, tuple, list],
        skip_activation=False,
        activation = 'relu',
        **kwargs):
        """
        Conv2D + BatchNormalization + Relu
        """

        x = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=self.conv_padding, **kwargs)(x)
        x = BatchNormalization()(x)
        if not skip_activation:
            x = Activation(activation)(x)
        if self.debug:
            print(x.shape)
        return x

    def encoder_stage(self, x, stage:int):
        """Encoder Convolutional"""
        filter = 2**(5+stage)
        x = MaxPooling2D()(x)
        x = self.conv_bn_act(x, filter, 3, 1)
        x = self.conv_bn_act(x, filter, 3, 1)
        return x
    
    def decoder_stage(self, x, merge_layer, stage:int):
        """Decoder Convolutional"""
        filter = 2**(5+stage)
        x = Conv2DTranspose(filter, 2, 2, padding=self.conv_padding)(x)
        crop = Cropping2D(
            (abs(merge_layer.shape[1]-x.shape[1])//2))(merge_layer)\
                if self.conv_padding == 'valid' else merge_layer
        x = Concatenate()([x, crop])
        x = self.conv_bn_act(x, filter, 3, 1)
        x = self.conv_bn_act(x, filter, 3, 1)
        return x

    def build_unet(self):
        # down sampling
        input = Input(shape=self.input_shape)
        stage1 = self.conv_bn_act(input, 64, 3, 1)
        stage1 = self.conv_bn_act(stage1, 64, 3, 1)

        stage2 = self.encoder_stage(stage1, stage=2)

        stage3 = self.encoder_stage(stage2, stage=3)

        stage4 = self.encoder_stage(stage3, stage=4)

        stage5 = self.encoder_stage(stage4, stage=5)

        # up sampling
        up4 = self.decoder_stage(stage5, stage4, 4)

        up3 = self.decoder_stage(up4, stage3, 3)

        up2 = self.decoder_stage(up3, stage2, 2)

        up1 = self.decoder_stage(up2, stage1, 1)
        output = self.conv_bn_act(up1, 2, 1, 1, activation='sigmoid')

        model = Model(inputs = input, outputs = output)

        return model

model = Unet(conv_padding='same', debug=True)
model.build_unet().summary()