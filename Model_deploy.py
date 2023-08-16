import numpy as np
import tensorflow.keras.backend as K
import os
import tensorflow as tf
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras import callbacks
from tensorflow.keras.layers import Layer, Dense, Input
from tensorflow.keras.models import Model
from Common_Model import Common_Model
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report
import datetime
import pandas as pd

from TIMNET import TIMNET
from loguru import logger

import librosa

def get_feature(file_path: str, mfcc_len: int = 39, mean_signal_length: int = 88000):
    """
    file_path: Speech signal folder
    mfcc_len: MFCC coefficient length
    mean_signal_length: MFCC feature average length
    """
    signal, fs = librosa.load(file_path)  # signal: speech signal, fs: sampling rate

    s_len = len(signal)

    if s_len < mean_signal_length:
        pad_len = mean_signal_length - s_len
        pad_rem = pad_len % 2
        pad_len //= 2
        signal = np.pad(signal, (pad_len, pad_len + pad_rem), 'constant', constant_values=0)
    else:
        pad_len = s_len - mean_signal_length
        pad_len //= 2
        signal = signal[pad_len:pad_len + mean_signal_length]
    mfcc = librosa.feature.mfcc(y=signal, sr=fs, n_mfcc=mfcc_len)
    mfcc = mfcc.T
    feature = mfcc
    return feature

class WeightLayer(Layer):
    def __init__(self, **kwargs):
        super(WeightLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[1], 1),
                                      initializer='uniform',
                                      trainable=True)
        super(WeightLayer, self).build(input_shape)

    def call(self, x):
        tempx = tf.transpose(x, [0, 2, 1])
        x = K.dot(tempx, self.kernel)
        x = tf.squeeze(x, axis=-1)
        return x

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[2])


def softmax(x, axis=-1):
    ex = K.exp(x - K.max(x, axis=axis, keepdims=True))
    return ex / K.sum(ex, axis=axis, keepdims=True)


class TIMNET_Model(Common_Model):
    def __init__(self, input_shape, class_label, **params):
        super(TIMNET_Model, self).__init__(**params)
        self.data_shape = input_shape
        self.num_classes = len(class_label)
        self.class_label = class_label
        print("TIMNET MODEL SHAPE:", input_shape)

    def create_model(self):
        self.inputs = Input(shape=(87, 39))
        self.multi_decision = TIMNET(nb_filters=39,
                                     kernel_size=2,
                                     nb_stacks=1,
                                     dilations=8,
                                     dropout_rate=0.1,
                                     activation='relu',
                                     return_sequences=True,
                                     name='TIMNET')(self.inputs)

        self.decision = WeightLayer()(self.multi_decision)
        self.predictions = Dense(self.num_classes, activation='softmax')(self.decision)
        self.model =  Model(inputs=self.inputs, outputs=self.predictions)
        weight_path = "/home/TIM-Net_SER/Code/Test_Models/CASIA_32/10-fold_weights_best_1.hdf5"
        self.model.load_weights(weight_path)

    def predict(self, x):
        """
        only use in CASIA_32
        """
        #data = np.reshape(x, (batch_size, len(x), len(x[1])))
        return self.model.predict(x)


def get_all_features(directory: str, batch_size: int = 64):
    features = []
    file_names = os.listdir(directory)
    for file_name in file_names:
        file_path = os.path.join(directory, file_name)
        feature = get_feature(file_path)
        features.append(feature)

    features = np.array(features)
    num_batches = len(features) // batch_size

    return np.array_split(features, num_batches)

if __name__ == "__main__":
    # 示例音频文件路径
    audio_file = "/home/TIM-Net_SER/Code/202-neutral-ZhaoZuoxiang.wav"

    # 创建 TIMNET_Model 对象
    input_shape = (87, 39)  # 设置适当的输入形状
    CLASS_LABELS = ("angry", "fear", "happy", "neutral", "neutral", "suprise")
    model = TIMNET_Model(input_shape, CLASS_LABELS)

    model.create_model()

    # 获取音频特征
    features = get_all_features("/home/TIM-Net_SER/test")
    print(features)

    for batch in features:
        prediction = model.predict(batch)
        print("Prediction:", prediction)