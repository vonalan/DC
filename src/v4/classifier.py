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

def run_svm(xs_train, ts_train, xs_test, ts_test, FLAGS):
    ts_train = np.argmax(ts_train, axis=1)
    ts_test = np.argmax(ts_test, axis=1)

    svc = SVC()
    svc.fit(xs_train, ts_train)

    ys_train = svc.predict(xs_train)
    ys_test = svc.predict(xs_test)

    # TODO: one hot label
    encoder = OneHotEncoder(n_values=FLAGS.rbfnn_output_dim)
    encoder.fit([i for i in range(FLAGS.rbfnn_output_dim)])

    ys_train = encoder.fit_transform(ys_train)
    ys_test = encoder.fit_transform(ys_test)
    ts_train = encoder.fit_transform(ts_train)
    ts_test = encoder.fit_transform(ts_test)

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
    c_train = np.loadtxt(FLAGS.path_to_ctrain).astype(np.int)
    c_test = np.loadtxt(FLAGS.path_to_ctest).astype(np.int)

    outputs_train = np.random.randint(0, FLAGS.depict_output_dim-1, (c_train.sum()))
    outputs_test = np.random.randint(0, FLAGS.depict_output_dim-1, (c_test.sum()))
    hist_train = utils.build_word_vector(outputs_train, c_train, FLAGS.depict_output_dim)
    hist_test = utils.build_word_vector(outputs_test, c_test, FLAGS.depict_output_dim)
    xs_train = sklscale(hist_train)
    xs_test = sklscale(hist_test)

    ts_train = np.loadtxt(FLAGS.path_to_ytrain)
    ts_test = np.loadtxt(FLAGS.path_to_ytest)

    metrics = run_rbfnn(xs_train, ts_train, xs_test, ts_test)
    # print(metrics)
    pprint.pprint(metrics)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--depict_input_dim',
        type=int,
        default=162
    )
    parser.add_argument(
        '--depict_output_dim',
        type=int,
        default=120
    )
    parser.add_argument(
        '--path_to_ctrain',
        type=str,
        default=r'D:\Users\kingdom\GIT\DC_OLD\data\kth_ctrain_r9.txt'
    )
    parser.add_argument(
        '--path_to_xtrain',
        type=str,
        default=r'D:\Users\kingdom\GIT\DC_OLD\data\kth_xtrain_r9.txt'
    )
    parser.add_argument(
        '--path_to_ytrain',
        type=str,
        default=r'D:\Users\kingdom\GIT\DC_OLD\data\kth_ytrain_r9.txt'
    )
    parser.add_argument(
        '--path_to_ctest',
        type=str,
        default=r'D:\Users\kingdom\GIT\DC_OLD\data\kth_ctest_r9.txt'
    )
    parser.add_argument(
        '--path_to_xtest',
        type=str,
        default=r'D:\Users\kingdom\GIT\DC_OLD\data\kth_xtest_r9.txt'
    )
    parser.add_argument(
        '--path_to_ytest',
        type=str,
        default=r'D:\Users\kingdom\GIT\DC_OLD\data\kth_ytest_r9.txt'
    )
    FLAGS, unparsed = parser.parse_known_args()
    pseudo()