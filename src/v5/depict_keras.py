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
import argparse
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



split_round = 9
database_name = 'kth'
database_root = r'E:\Users\kingdom\KTH'
num_classes = 6

parser = argparse.ArgumentParser()
parser.add_argument('--database_name', type=str, default=database_name)
parser.add_argument('--split_round', type=int, default=split_round)
parser.add_argument('--path_to_ctrain', type=str, default=os.path.join(database_root, 'data\pwd\%s_ctrain_r%d.txt'%(database_name, split_round)))
parser.add_argument('--path_to_xtrain', type=str, default=os.path.join(database_root, 'data\pwd\%s_xtrain_r%d.txt'%(database_name, split_round)))
parser.add_argument('--path_to_ytrain', type=str, default=os.path.join(database_root, 'data\pwd\%s_ytrain_r%d.txt'%(database_name, split_round)))
parser.add_argument('--path_to_ctest', type=str, default=os.path.join(database_root, 'data\pwd\%s_ctest_r%d.txt'%(database_name, split_round)))
parser.add_argument('--path_to_xtest', type=str, default=os.path.join(database_root, 'data\pwd\%s_xtest_r%d.txt'%(database_name, split_round)))
parser.add_argument('--path_to_ytest', type=str, default=os.path.join(database_root, 'data\pwd\%s_ytest_r%d.txt'%(database_name, split_round)))
parser.add_argument('--path_to_xrand', type=str, default=os.path.join(database_root, 'data\pwd\%s_xrand_r%d.txt'%(database_name, split_round)))
parser.add_argument('--depict_output_dim', type=int, default=128)
parser.add_argument('--rbfnn_input_dim', type=int, default=128)
parser.add_argument('--rbfnn_num_center', type=int, default=120)
parser.add_argument('--rbfnn_output_dim', type=int, default=num_classes)
parser.add_argument('--saved_model_dir', type=str, default='../../models/')
parser.add_argument('--saved_results_dir', type=str, default='../../results/')
FLAGS, _ = parser.parse_known_args()
# pprint.pprint(FLAGS)

sys.path.append('../v4')
# from ..v4 import classifier
import classifier

xtrain = np.loadtxt(FLAGS.path_to_xtrain)
xtest = np.loadtxt(FLAGS.path_to_xtest)
xrand = np.loadtxt(FLAGS.path_to_xrand)
print(xtrain.shape, xtest.shape, xrand.shape)

depict_loss = lambda y_true, y_pred: y_pred


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

    for i in range(3,10):
        train_model.compile(optimizer=Adam(lr=pow(10, -1 * i)), loss=depict_loss)
        train_model.fit(xtrain, xtrain, epochs=10 * (i-2), validation_split=0.2)
        ys_train = base_model.predict(xtrain)
        ys_test = base_model.predict(xtest)

        metrics = classifier.run_with_soft_assignment(ys_train, ys_test, FLAGS)
        print('i: %d'%(i))
        pprint.pprint(metrics)


if __name__ == '__main__':
    main()

