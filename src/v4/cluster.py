# -*- coding: utf-8 -*-

import os
import sys
import pprint

import numpy as np
import tensorflow as tf

import classifier

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import minmax_scale as sklscale
from sklearn.metrics import normalized_mutual_info_score as sklnmi

FLAGS = None


def build_word_vector(lineDataSet, lineCount, bins=4096):
    Hist = np.zeros((0, bins))
    begin = 0
    for i in range(lineCount.shape[0]):
        end = begin + int(lineCount[i])
        hist = np.histogram(lineDataSet[begin:end], bins, range=(0, bins))[0]
        Hist = np.vstack((Hist, hist))
        begin = end
    return Hist


def build_kmeans_model_with_fixed_input(FLAGS):
    from sklearn.externals import joblib
    from sklearn.cluster import KMeans

    xrand = np.loadtxt(FLAGS.path_to_xrand)
    print(xrand.shape)

    num_samples = xrand.shape[0]
    assert num_samples == 100000

    if not os.path.exists(FLAGS.save_model_dir): os.makedirs(FLAGS.save_model_dir)
    mfile = os.path.join(FLAGS.save_model_dir, r"%s_r%d_kmeans_k%d_m%d.m" % (FLAGS.database_name, FLAGS.split_round, FLAGS.depict_output_dim, num_samples))
    if not os.path.exists(mfile):
        kms = KMeans(n_clusters=FLAGS.depict_output_dim).fit(xrand)
        joblib.dump(kms, mfile, compress=3)
        print('%s is saved! ' % (mfile))
    else:
        kms = joblib.load(mfile)
        print('%s is loaded! ' % (mfile))
    return kms


def build_kmeans_model_with_random_input(root, name, X=None, outdim=4096, factor=4):
    from sklearn.externals import joblib
    from sklearn.cluster import KMeans

    dstdir = os.path.join(root, name)
    if not os.path.exists(dstdir): os.makedirs(dstdir)
    # num_samples = 100000 if factor <= 0 else outdim * factor
    num_samples = 100000
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

def main():
    xs_train = np.loadtxt(FLAGS.path_to_xtrain)
    xs_test = np.loadtxt(FLAGS.path_to_xtest)
    kms = build_kmeans_model_with_random_input(FLAGS.model_dir, 'kmeans', xs_train, FLAGS.depict_output_dim)
    outputs_train = kms.predict(xs_train)
    output_test = kms.predict(xs_test)
    metrics = classifier.run(outputs_train, output_test, FLAGS)
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
        default=4096
    )
    parser.add_argument(
        '--model_dir',
        type=str,
        default=r'D:\Users\kingdom\GIT\DC_OLD\model'
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
    parser.add_argument(
        '--rbfnn_num_center',
        type=int,
        default=90
    )
    parser.add_argument(
        '--rbfnn_output_dim',
        type=int,
        default=6
    )
    FLAGS, unparsed = parser.parse_known_args()
    pprint.pprint(FLAGS)
    main()