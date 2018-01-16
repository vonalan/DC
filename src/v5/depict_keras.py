# -*- coding: utf-8 -*-

# requirements: 
#   Anaconda3
#   Tensorflow
#   Keras

# feature works: 
#     moving average 
#     batch norminalization
#     RNN
#     distributions
#     vector representations of words
#     continuous bag-of-words model (CBOW)
#     Skip-Gram Model

import os 
import sys 
import math
import time
import datetime as dt
import itertools
import pprint

import numpy as np 
import tensorflow as tf
from keras import backend as keras
from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras.optimizers import Adam
from keras.activations import softmax

from tensorflow.python.framework import graph_util
from sklearn.metrics import normalized_mutual_info_score as sklnmi

# import classifier


def build_basic_model(input_shape, output_shape):
    from keras.activations import softmax
    inputs = Input(shape=input_shape)
    x = inputs
    x = Dense(128, activation=softmax, use_bias=True)(x)
    outputs = x
    model = Model(inputs, outputs)
    return model

def depict_loss_layer(P, loss_function=None):
    N = keras.reshape(keras.sum(P, axis=0), (-1, 128))
    N = P / keras.pow(N, 0.5)
    D = keras.reshape(keras.sum(N, 1), (-1, 1))
    Q = N / D

    # TODO: make U trainable!!!
    prior = [1 / float(128)] * 128
    U = keras.variable(prior)
    F = keras.reshape(keras.mean(Q, axis=0), (-1, 128))

    C = Q * keras.log(Q / P)
    R = Q * keras.log(F / U)
    L = keras.reshape(keras.sum(C + R, axis=1), (-1, 1))

    return L

def build(input_shape):
    base_model = build_basic_model(input_shape, None)

    inputs = Input(shape=input_shape)
    x = inputs
    P = base_model(x)
    outputs = Lambda(depict_loss_layer)(P)
    model = Model(inputs, outputs)

    return base_model, model

def main():
    depict_input_shape = (162,)
    depict_output_shape=(128,)
    base_model, train_model = build(depict_input_shape)

    depict_loss = lambda y_true, y_pred: y_pred
    train_model.compile(optimizer=Adam(), loss=depict_loss)

    xs = np.loadtxt(r"E:\Users\kingdom\KTH\data\txt\kth_xtrain_r9.txt")
    train_model.fit(xs, xs, epochs=10, validation_split=0.1)

if __name__ == '__main__':
    main()

