# -*- coding: utf-8 -*-

import os
import sys
import pprint
import argparse

import numpy as np
import tensorflow as tf

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import minmax_scale as sklscale
from sklearn.metrics import normalized_mutual_info_score as sklnmi


parser = argparse.ArgumentParser()
parser.add_argument('database_name', type='str', default='ucf')
parser.add_argument('split_round', type=int, default=9)
parser.add_argument('--path_to_ctrain', type=str, default=r'D:\Users\kingdom\GIT\DC_OLD\data\kth_ctrain_r9.txt')
parser.add_argument('--path_to_xtrain', type=str, default=r'D:\Users\kingdom\GIT\DC_OLD\data\kth_xtrain_r9.txt')
parser.add_argument('--path_to_ytrain', type=str, default=r'D:\Users\kingdom\GIT\DC_OLD\data\kth_ytrain_r9.txt')
parser.add_argument('--path_to_ctest', type=str, default=r'D:\Users\kingdom\GIT\DC_OLD\data\kth_ctest_r9.txt')
parser.add_argument('--path_to_xtest', type=str, default=r'D:\Users\kingdom\GIT\DC_OLD\data\kth_xtest_r9.txt')
parser.add_argument('--path_to_ytest', type=str, default=r'D:\Users\kingdom\GIT\DC_OLD\data\kth_ytest_r9.txt')
parser.add_argument('--path_to_xrand', type=str, default=r'D:\Users\kingdom\GIT\DC_OLD\data\kth_xrand_r9.txt')
parser.add_argument('--depict_output_dim', type=int, default=-1)
parser.add_argument('--rbfnn_input_dim', type=int, default=-1)
parser.add_argument('--rbfnn_num_center', type=int, default=-1)
parser.add_argument('--rbfnn_output_dim', type=int, default=-1)
parser.add_argument('--save_model_dir', type=str, default='../../models/')
parser.add_argument('--save_results_dir', type=str, default='../../results/')
FLAGS, _ = parser.parse_known_args()


import cluster
import classifier


xtrain = np.loadtxt(FLAGS.path_to_xtrain)
xtest = np.loadtxt(FLAGS.path_to_xtest)


for i in range(7, 16 + 1):
    num_cluster = 1 << i
    FLAGS.depict_output_dim = num_cluster
    FLAGS.rbfnn_input_dim = num_cluster
    kms = cluster.build_kmeans_model_with_fixed_inputs(FLAGS, num_cluster)
    ca_train = kms.predict(xtrain)
    ca_test = kms.predict(xtest)
    metrics = classifier.run(ca_train, ca_test, FLAGS)

    # TODO:
    if not os.path.exists(FLAGS.save_results_dir): os.makedirs(FLAGS.save_results_dir)
    outfile = os.path.join(FLAGS.save_results_dir, '%s_%d_kmeans_results.txt'%(FLAGS.database_name, FLAGS.split_rounds))
    with open(outfile, 'a') as f:
        line = list()
        line.extend([FLAGS.rbfnn_num_center, FLAGS.depict_output_dim, 0])
        line.extend(metrics['err_train'].tolist())
        line.extend([metrics['acc_train']])
        line.extend(metrics['stsm_train'].tolist())
        line.extend(metrics['err_test'].tolist())
        line.extend([metrics['acc_test']])
        line = [str(item) for item in line]
        line = ' '.join(line)
        f.write(line)
        f.write('\n')
