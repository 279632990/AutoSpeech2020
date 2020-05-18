#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time:    2019/9/27 10:12
# @Author:  Mecthew

import tensorflow as tf
import numpy as np
from tensorflow.python.keras import optimizers,regularizers
from tensorflow.python.keras.layers import (SpatialDropout1D, Input, GlobalMaxPool1D,
                                            Dense, Dropout, CuDNNLSTM, Activation, Lambda, Flatten)
from tensorflow.python.keras.models import Model as TFModel
from tensorflow.python.keras.losses import CategoricalCrossentropy
from CONSTANT import MAX_FRAME_NUM, IS_CUT_AUDIO, MAX_AUDIO_DURATION, AUDIO_SAMPLE_RATE
from data_process import (extract_mfcc_parallel)
from data_process import ohe2cat, get_max_length, pad_seq
from models.attention import Attention
from models.my_classifier import Classifier
from tools import log
from data_augmentation import mix_up,DropConnect,DropBlock2D
import keras.backend as K

def sparse_amsoftmax_loss(y_true, y_pred, scale=30, margin=0.35,label_smoothing=0.2):
    y_pred = y_true * (y_pred - margin) + (1 - y_true) * y_pred
    y_pred *= scale
    return K.categorical_crossentropy(y_true, y_pred, from_logits=True)

class LstmAttention(Classifier):
    def __init__(self):
        # clear_session()
        log("new LSTM")
        self.max_length = None
        self._model = None
        self.is_init = False
        self.mfcc_mean, self.mfcc_std = None, None
        self.mel_mean, self.mel_std = None, None
        self.cent_mean, self.cent_std = None, None
        self.stft_mean, self.stft_std = None, None

    def preprocess_data(self, x):
        if IS_CUT_AUDIO:
            x = [sample[0:MAX_AUDIO_DURATION * AUDIO_SAMPLE_RATE]
                 for sample in x]
        # extract mfcc
        x_mfcc = extract_mfcc_parallel(x, n_mfcc=96)
        if self.max_length is None:
            self.max_length = get_max_length(x_mfcc)
            self.max_length = min(MAX_FRAME_NUM, self.max_length)
        x_mfcc = pad_seq(x_mfcc, pad_len=self.max_length)

        return x_mfcc

    def init_model(self,
                   input_shape,
                   num_classes,
                   **kwargs):
        inputs = Input(shape=input_shape)
        sequence_len = input_shape[0]
        lstm_units_array = np.array([32, 64, 128, 256, 512])
        lstm_units = lstm_units_array[np.argmin(np.abs(lstm_units_array-sequence_len))]
        lstm_1 = CuDNNLSTM(lstm_units, return_sequences=True)(inputs)
        activation_1 = Activation('tanh')(lstm_1)
        if num_classes >= 20:
            if num_classes < 30:
                dropout1 = SpatialDropout1D(0.5)(activation_1)
                attention_1 = Attention(8, 16)([dropout1, dropout1, dropout1])
            else:
                attention_1 = Attention(8, 16)(
                    [activation_1, activation_1, activation_1])
            k_num = 10
            kmaxpool_l = Lambda(
                lambda x: tf.reshape(tf.nn.top_k(tf.transpose(x, [0, 2, 1]), k=k_num, sorted=True)[0],
                                     shape=[-1, k_num, 128]))(attention_1)
            flatten = Flatten()(kmaxpool_l)
            dropout2 = Dropout(rate=0.5)(flatten)
        else:
            dropout1 = SpatialDropout1D(0.5)(activation_1)
            attention_1 = Attention(8, 16)([dropout1, dropout1, dropout1])
            pool_l = GlobalMaxPool1D()(attention_1)
            dropout2 = Dropout(rate=0.5)(pool_l)
        dense_1 = Dense(units=256, activation='relu')(dropout2)
#         dense_1 = Dense(units=256, activation='softplus',kernel_regularizer=regularizers.l2(0.01),
#                        activity_regularizer=regularizers.l1(0.01))(dropout2)
        #dense_1 = DropConnect(Dense(units=256, activation='softplus'), prob=0.5)(dropout2)
        outputs = Dense(units=num_classes, activation='softmax')(dense_1)
        loss_fun = CategoricalCrossentropy(label_smoothing=0.2)
        model = TFModel(inputs=inputs, outputs=outputs)
        optimizer = optimizers.Nadam(
            lr=0.002,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=None,
            schedule_decay=0.004)
        model.compile(
            optimizer=optimizer,
            loss=loss_fun,
            #loss="sparse_categorical_crossentropy",
            metrics=['accuracy'])
        model.summary()
        self._model = model
        self.is_init = True

    def fit(self, train_x, train_y, validation_data_fit, round_num, **kwargs):
        val_x, val_y = validation_data_fit
        if train_x.shape[1]<180:
            if round_num >= 2:
                epochs = min(5+round_num,10)
                patience = 2
            else:
                epochs = 5
                patience = 2
        else:
            if round_num >= 2:
                epochs = min(6+round_num,12)
                patience = 2
            else:
                epochs = 6
                patience = 2
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_acc',
                patience=patience)]
        if train_y.shape[1]>7:
            train_x, train_y=mix_up(train_x, train_y)
        self._model.fit(train_x, train_y,
        #self._model.fit(train_x, ohe2cat(train_y),
                        epochs=epochs,
                        callbacks=callbacks,
                        validation_data=(val_x, val_y),
                        #validation_data=(val_x, ohe2cat(val_y)),
                        verbose=1,  # Logs once per epoch.
                        batch_size=32,
                        shuffle=True,
                        )
    def predict(self, x_test, batch_size=32):
        return self._model.predict(x_test, batch_size=batch_size)
