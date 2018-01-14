# -*- coding: utf-8 -*-

import os
import sys
import math

import numpy as np
import tensorflow as tf

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import minmax_scale as sklscale
from sklearn.metrics import normalized_mutual_info_score as sklnmi


def build_word_vector(lineDataSet, lineCount, bins=4096):
    Hist = np.zeros((0, bins))
    begin = 0
    for i in range(lineCount.shape[0]):
        end = begin + int(lineCount[i])
        hist = np.histogram(lineDataSet[begin:end], bins, range=(0, bins))[0]
        Hist = np.vstack((Hist, hist))
        begin = end
    return Hist

def build_data_generator(filenames=None, shuffle=False, batch_size=1):
    # dataset = np.zeros((0, 128))
    dataset = None
    for filename in filenames:
        mini_batch = np.loadtxt(filename)
        if isinstance(dataset, np.ndarray):
            dataset = np.vstack((dataset, mini_batch))
        else:
            dataset = mini_batch
    print(dataset.shape)
    num_samples, num_features = dataset.shape
    num_batches = int(math.ceil(num_samples/batch_size))
    indices = np.arange(num_samples)
    if shuffle:
        np.random.shuffle(indices)
    for batch in range(num_batches):
        start = batch * batch_size
        end = min((batch + 1) * batch_size, num_samples)
        mini_batch = dataset[start:end,:]
        yield mini_batch
        # return mini_batch

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
    filenames = ['../../data/x_1000_128.txt', '../../data/x_1000_128_kmeans_10.txt']
    data_generator = build_data_generator(filenames=filenames, shuffle=False, batch_size=100)

    import time
    import itertools

    for i in itertools.count():
        try:
            xs = next(data_generator)
            print(i, xs.shape, xs.flatten()[:5])
            time.sleep(0.5)
        except Exception:
            data_generator = build_data_generator(filenames=filenames, shuffle=False, batch_size=100)
            xs = next(data_generator)
            print(i, xs.shape, xs.flatten()[:5])
            time.sleep(0.5)
        if i > 1000:
            break