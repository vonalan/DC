# -*- coding: utf-8 -*-

import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error as sklmse
from sklearn.preprocessing import minmax_scale as sklscale
# from sklearn.metrics import normalized_mutual_info_score as sklnmi

import rbfnn 

def calc_err(y_true, y_predict):
    return sklmse(y_true, y_predict, multioutput='raw_values')
def calc_acc(y_true, y_predict):
    tidx = np.argmax(y_true, axis=1)
    pidx = np.argmax(y_predict, axis=1)
    eidx = (tidx == pidx)
    acc = eidx.sum() / eidx.shape[0]
    return acc

load = lambda x: (np.loadtxt('../data/x_%s.txt'%(x)), 
                  np.loadtxt('../data/y_%s.txt'%(x)))

x_train, y_train = load('train') 
x_testa, y_testa = load('testa') 

print(x_train.shape, y_train.shape)
print(x_testa.shape, y_testa.shape)

network = rbfnn.RBFNN(indim=x_train.shape[1], numCenter=120, outdim=6)
network.fit(x_train, y_train)
o_train, o_testa = network.predict(x_train), network.predict(x_testa)

acc_train, acc_testa = calc_acc(y_train, o_train), calc_acc(y_testa, o_testa)
err_train, err_testa = calc_err(y_train, o_train).sum(), calc_err(y_testa, o_testa).sum()

print('err_train: %f, acc_train: %f, err_testa: %f, acc_testa: %f'%(err_train, acc_train, err_testa, acc_testa))