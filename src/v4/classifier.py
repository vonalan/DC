# -*- coding: utf-8 -*-

# requirements: 
#   Anaconda3
#   Tensorflow

# feature works: 
#     moving average 
#     batch norminalization
#     RNN

import os 
import sys 
import math
import time
import datetime as dt
import itertools
import pprint

import numpy as np 
import tensorflow as tf 
from tensorflow.python.framework import graph_util
from sklearn.preprocessing import minmax_scale as sklscale
from sklearn.metrics import mean_squared_error as sklmse
from sklearn.metrics import accuracy_score as sklacc
from sklearn.metrics import normalized_mutual_info_score as sklnmi
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

import utils
import rbfnn
import stsm_pseudo as stsm

def calc_err(y_true, y_predict):
    return sklmse(y_true, y_predict, multioutput='raw_values')

def calc_acc(y_true, y_predict):
    tidx = np.argmax(y_true, axis=1)
    pidx = np.argmax(y_predict, axis=1)
    eidx = (tidx == pidx)
    acc = eidx.sum() / eidx.shape[0]
    return acc

def one_hot(ys, n_values=6):
    # TODO: one hot label
    tmp_train = np.zeros((ys.shape[0], n_values))
    for i, label in enumerate(ys.tolist()):
        tmp_train[i, label] = 1
    ys = tmp_train
    return ys

def run_svm(xs_train, ts_train, xs_test, ts_test, FLAGS=None):
    ts_train = np.argmax(ts_train, axis=1)
    ts_test = np.argmax(ts_test, axis=1)

    svc = SVC()
    svc.fit(xs_train, ts_train)

    ys_train = svc.predict(xs_train)
    ys_test = svc.predict(xs_test)

    ys_train = one_hot(ys_train)
    ts_train = one_hot(ts_train)

    ys_test = one_hot(ys_test)
    ts_test = one_hot(ts_test)

    err_train = calc_err(ts_train, ys_train)
    acc_train = calc_acc(ts_train, ys_train)

    err_test = calc_err(ts_test, ys_test)
    acc_test = calc_acc(ts_test, ys_test)

    stsm_train = 0.25

    return dict(err_train=err_train,
                acc_train=acc_train,
                err_test=err_test,
                acc_test=acc_test,
                stsm_train=stsm_train)

def run_rbfnn_dev(xs_train, ts_train, xs_test, ts_test, FLAGS=None):
    # assert FLAGS.rbfnn_num_center % FLAGS.rbfnn_output_dim == 0

    network = rbfnn.Network(4096, 120, 6)
    network.fit(xs_train, ts_train)

    ys_train = network.predict(xs_train)
    ys_test = network.predict(xs_test)

    err_train = calc_err(ts_train, ys_train)
    acc_train = calc_acc(ts_train, ys_train)

    err_test = calc_err(ts_test, ys_test)
    acc_test = calc_acc(ts_test, ys_test)

    stsm_train = stsm.calc_stsm_vector(network, xs_train, cost_func=calc_err)

    return dict(err_train=err_train,
                acc_train=acc_train,
                err_test=err_test,
                acc_test=acc_test,
                stsm_train=stsm_train)

def run_rbfnn(xs_train, ts_train, xs_test, ts_test, FLAGS):
    assert FLAGS.rbfnn_num_center % FLAGS.rbfnn_output_dim == 0

    network = rbfnn.Network(FLAGS.depict_output_dim, FLAGS.rbfnn_num_center, FLAGS.rbfnn_output_dim)
    network.fit(xs_train, ts_train)

    ys_train = network.predict(xs_train)
    ys_test = network.predict(xs_test)

    err_train = calc_err(ts_train, ys_train)
    acc_train = calc_acc(ts_train, ys_train)

    err_test = calc_err(ts_test, ys_test)
    acc_test = calc_acc(ts_test, ys_test)

    stsm_train = stsm.calc_stsm_vector(network, xs_train, cost_func=calc_err)

    return dict(err_train=err_train,
                acc_train=acc_train,
                err_test=err_test,
                acc_test=acc_test,
                stsm_train=stsm_train)

def run(outputs_train, outputs_eval, FLAGS):
    c_train = np.loadtxt(FLAGS.path_to_ctrain)
    c_test = np.loadtxt(FLAGS.path_to_ctest)

    hist_train = utils.build_word_vector(outputs_train, c_train, FLAGS.depict_output_dim)
    hist_test = utils.build_word_vector(outputs_eval, c_test, FLAGS.depict_output_dim)
    xs_train = sklscale(hist_train)
    xs_test = sklscale(hist_test)

    # step_03 load
    ts_train = np.loadtxt(FLAGS.path_to_ytrain)
    ts_test = np.loadtxt(FLAGS.path_to_ytest)

    # TODO: rbfnn, svm, softmax
    return run_rbfnn(xs_train, ts_train, xs_test, ts_test, FLAGS)
    # return run_svm(xs_train, ts_train, xs_test, ts_test, FLAGS)

def pseudo():
    x_train = np.loadtxt(r'D:\Users\kingdom\GIT\DC_OLD\data\temp\x_train.txt')
    y_train = np.loadtxt(r'D:\Users\kingdom\GIT\DC_OLD\data\temp\y_train.txt')

    x_test = np.loadtxt(r'D:\Users\kingdom\GIT\DC_OLD\data\temp\x_testa.txt')
    y_test = np.loadtxt(r'D:\Users\kingdom\GIT\DC_OLD\data\temp\y_testa.txt')

    results = run_rbfnn_dev(x_train, y_train, x_test, y_test)
    pprint.pprint(results)

if __name__ == '__main__':
    pseudo()