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

import numpy as np 
import tensorflow as tf 
from tensorflow.python.framework import graph_util
from sklearn.preprocessing import minmax_scale as sklscale
from sklearn.metrics import mean_squared_error as sklmse
from sklearn.metrics import accuracy_score as sklacc
from sklearn.metrics import normalized_mutual_info_score as sklnmi
from sklearn.svm import SVC

import utils
import rbfnn

def calc_err(y_true, y_predict):
    return sklmse(y_true, y_predict, multioutput='raw_values')

def calc_acc(y_true, y_predict):
    tidx = np.argmax(y_true, axis=1)
    pidx = np.argmax(y_predict, axis=1)
    eidx = (tidx == pidx)
    acc = eidx.sum() / eidx.shape[0]
    return acc

def run_svm(xs_train, ts_train, xs_test, ts_test):
    svc = SVC()
    svc.fit(xs_train, ts_train)

    ys_train = svc.predict(xs_train)
    ys_test = svc.predict(xs_test)

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

def run_rbfnn(xs_train, ts_train, xs_test, ts_test):
    network = rbfnn.Network()
    network.train(xs_train)

    ys_train = network.predict(xs_train)
    ys_test = network.predict(xs_test)

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

def run(outputs_train, outputs_eval, bins, xs_train=None, ts_train=None, xs_test=None, ys_test=None):
    # step_02 build word vectors
    c_train = np.loadtxt(cline_path)
    c_test = np.loadtxt()
    hist_train = utils.build_word_vector(outputs_train, c_train, bins)
    hist_test = utils.build_word_vector(outputs_eval, c_test, bins)
    xs_train = sklscale(hist_train)
    xs_test = sklscale(hist_test)

    # step_03 load
    ts_train = None
    ts_test = None

    return run_rbfnn(xs_train, ts_train, xs_test, ts_test)