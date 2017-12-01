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

def build_text_line_reader(filenames=None, shuffle=False, batch_size=1):
    dataset = np.zeros((0, 162))
    for filename in filenames:
        mini_batch = np.loadtxt(filename)
        # if dataset == None:
        #     dataset = mini_batch
        # else:
        #     dataset.vstack((dataset, mini_batch))
        np.vstack((dataset, mini_batch))
    num_samples, num_features = dataset.shape
    num_batches = int(math.ceil(num_samples/batch_size))
    indices = np.arange(num_samples)
    if shuffle:
        np.random.shuffle(indices)
    for batch in range(num_batches):
        start = batch * batch_size
        end = (batch + 1) * batch_size
        if end > num_samples:
            end = num_samples
        mini_batch = dataset[start:end,:]
        yield mini_batch


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