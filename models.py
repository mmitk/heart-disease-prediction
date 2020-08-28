import tensorflow as tf
from tensorflow import keras

import os
import tempfile

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import sklearn
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

METRICS = [
            keras.metrics.TruePositives(name='tp'),
            keras.metrics.FalsePositives(name='fp'),
            keras.metrics.TrueNegatives(name='tn'),
            keras.metrics.FalseNegatives(name='fn'),
            keras.metrics.BinaryAccuracy(name='accuracy'),
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall'),
            keras.metrics.AUC(name='auc'),
        ]

class modelObj:

    def __init__(self, n_cols=None):
        self.EPOCHS = 100
        self.BATCH_SIZE = 32

        if n_cols is not None:
            self.create_model(n_cols)
        else:
            self.model = None

    def create_model(self, n_cols, metrics=METRICS, bias=None):
        if bias is not None:
            bias = tf.keras.initializers.Constant(bias)
        model = keras.Sequential([
            keras.layers.Dense(
                16, activation='relu',
                input_shape=(n_cols,)),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(1, activation='sigmoid',bias_initializer=bias),
        ])

        model.compile(
            optimizer=keras.optimizers.Adam(lr=1e-3),
            loss=keras.losses.BinaryCrossentropy(),
            metrics=metrics)

        self.model = model
        return model

    def get_summary(self):
        if self.model is not None:
            return self.model.summary()
        else:
            return 'Model is NoneType'


class printDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 10 == 0: print('.')