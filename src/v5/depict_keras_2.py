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
parser.add_argument('--depict_input_dim', type=int, default=-1)
parser.add_argument('--depict_output_dim', type=int, default=-1)
parser.add_argument('--rbfnn_input_dim', type=int, default=-1)
parser.add_argument('--rbfnn_num_center', type=int, default=-1)
parser.add_argument('--rbfnn_output_dim', type=int, default=num_classes)
parser.add_argument('--saved_model_dir', type=str, default='../../models/')
parser.add_argument('--saved_results_dir', type=str, default='../../results/')
parser.add_argument('--device', type=str, default='gpu')

FLAGS, _ = parser.parse_known_args()
# pprint.pprint(FLAGS)


sys.path.extend(['../v4'])
# from ..v4 import classifier
import classifier, utils, cluster

# alpha = 1e-0
depict_loss = lambda y_true, y_pred: y_pred


def build_basic_model(FLAGS):
    from keras.activations import softmax
    inputs = Input(shape=(FLAGS.depict_input_dim,))
    x = inputs
    x = Dense(FLAGS.depict_output_dim, activation=softmax, use_bias=True)(x)
    outputs = x
    model = Model(inputs, outputs)
    return model

def depict_loss_layer(args, FLAGS, alpha, prior):
    P = args
    N = keras.reshape(keras.sum(P, axis=0), (-1, FLAGS.depict_output_dim))
    N = P / keras.pow(N, 0.5)
    D = keras.reshape(keras.sum(N, 1), (-1, 1))
    Q = N / D

    # TODO: make U trainable!!!
    U = keras.variable(prior)
    F = keras.reshape(keras.mean(Q, axis=0), (-1, FLAGS.depict_output_dim))

    C = Q * keras.log(Q / P)
    R = Q * keras.log(F / U)

    # alpha = 1e-4
    L = keras.reshape(keras.sum(C + alpha * R, axis=1), (-1, 1))

    return L

def build(FLAGS, alpha, prior):
    base_model = build_basic_model(FLAGS)

    inputs = Input(shape=(FLAGS.depict_input_dim,))
    x = inputs
    P = base_model(x)
    outputs = Lambda(depict_loss_layer, arguments={'FLAGS': FLAGS, 'alpha': alpha, 'prior': prior})(P)
    model = Model(inputs, outputs)

    return base_model, model

def build_model_test(FLAGS):
    FLAGS.depict_input_dim = 162
    FLAGS.depict_output_dim = 128
    prior = [1 / float(FLAGS.depict_output_dim)] * FLAGS.depict_output_dim
    alpha = 1.0
    base_model, model = build(FLAGS, alpha, prior)
    # for i, layer in enumerate(model.layers):
    #     print(i, layer.name)
    base_model.summary()
    model.summary()

def main():
    alpha = 1.0
    print(alpha)

    xtrain = np.loadtxt(FLAGS.path_to_xtrain)
    xtest = np.loadtxt(FLAGS.path_to_xtest)
    xrand = np.loadtxt(FLAGS.path_to_xrand)
    print(xtrain.shape, xtest.shape, xrand.shape)

    FLAGS.depict_input_dim = 162
    FLAGS.rbfnn_num_center = 120
    for i in range(7, 16 + 1):
        k = 1 << i
        FLAGS.depict_output_dim = k
        FLAGS.rbfnn_input_dim = k
        pprint.pprint(FLAGS)

        # kms = cluster.build_kmeans_model_with_fixed_input(FLAGS, xrand)
        kms = cluster.build_kmeans_model_with_random_input(FLAGS, xtrain)
        qtrain = kms.predict(xtrain)
        qtrain = np.histogram(qtrain, FLAGS.depict_output_dim, range=(0, FLAGS.depict_output_dim))[0]
        qtrain = qtrain / qtrain.sum()
        print(qtrain)

        base_model, train_model = build(FLAGS, alpha, qtrain)

        for i in range(10):
            train_model.compile(optimizer=Adam(lr=1e-4), loss=depict_loss)
            train_model.fit(xtrain, xtrain, epochs=1, validation_split=0.2)
            ys_train = base_model.predict(xtrain)
            ys_test = base_model.predict(xtest)
            metrics = classifier.run_with_soft_assignment(ys_train, ys_test, FLAGS)

            # metrics = classifier.run(ys_train, ys_test, FLAGS)

            print('num_cluster: %d, iteration: %d, alpha: %f' % (k, i, alpha))
            pprint.pprint(metrics)
            # utils.write_results(FLAGS, metrics, i, postfix='alpha_%.12f'%(alpha))

if __name__ == '__main__':
    if FLAGS.device == 'cpu':
        with tf.device('/cpu:0'):
            main()
    else:
        main()

    # build_model_test(FLAGS)
