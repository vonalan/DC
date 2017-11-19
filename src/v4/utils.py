# -*- coding: utf-8 -*-

import os
import sys

import numpy as np
import tensorflow as tf

from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error as sklmse
from sklearn.preprocessing import minmax_scale as sklscale
from sklearn.metrics import normalized_mutual_info_score as sklnmi


def build_word_vector(lineDataSet, lineCount, bins=4096):
    Hist = np.zeros((0, bins))
    begin = 0
    for i in range(lineCount.shape[0]):
        end = begin + int(lineCount[i])
        hist = np.histogram(lineDataSet[begin:end, 0], bins, range=(0, bins))[0]
        Hist = np.vstack((Hist, hist))
        begin = end
    return Hist

def calc_err(y_true, y_predict):
    return sklmse(y_true, y_predict, multioutput='raw_values')

def calc_acc(y_true, y_predict):
    tidx = np.argmax(y_true, axis=1)
    pidx = np.argmax(y_predict, axis=1)
    eidx = (tidx == pidx)
    acc = eidx.sum() / eidx.shape[0]
    return acc

def writeLog(root, name, string):
    if not os.path.exists(root): os.mkdir(root)
    file = r'%s/%s.txt'%(root, name)
    wf = open(file, 'a')
    wf.write(string)
    wf.write('\n')
    wf.close()

def saveModel(sess, root, name, epoch):
    dstdir = os.path.join(root, name)
    if not os.path.exists(dstdir): os.mkdir(dstdir)
    saver = tf.train.Saver()
    saver.save(sess, os.path.join(dstdir, '%s_%d.ckpt' % (name, epoch)))

def loadModel(root, name, epoch):
    dstdir = os.path.join(root, name)
    sess = tf.Session()
    saver = tf.train.Saver()
    saver.save(sess, os.path.join(dstdir, '%s_%d.ckpt' % (name, epoch)))
    return sess

def kmeans(root, name, X=None, outdim=4096, factor=4):
    import random
    from sklearn.externals import joblib
    from sklearn.cluster import KMeans
    dstdir = os.path.join(root, name)
    if not os.path.exists(dstdir): os.mkdir(dstdir)
    num_samples = 100000 if factor <= 0 else outdim * factor
    mfile = os.path.join(dstdir, r"kth_kmeans_r9_k%d_m%d.m" % (outdim, num_samples))
    if not os.path.exists(mfile):
        indices = np.arange(X.shape[0])
        np.random.shuffle(indices)
        xs = X[indices[:num_samples], :]
        print(xs.shape)
        kms = KMeans(n_clusters=outdim).fit(xs)
        joblib.dump(kms, mfile, compress=3)
    else:
        kms = joblib.load(mfile)
    return kms


if __name__ == "__main__":
    print("I'm ok! ")