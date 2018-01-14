# -*- coding: utf-8 -*-

# requirements: 
#   Anaconda3
#   Tensorflow

# feature works: 
#     moving average 
#     batch norminalization
#     RNN
#     distributions
#     vector representations of words
#         continuous bag-of-words model (CBOW)
#         Skip-Gram Model

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
from sklearn.metrics import normalized_mutual_info_score as sklnmi

import classifier

FLAGS = None

def build_text_line_reader(filenames=None, shuffle=False, batch_size=1):
    filenames = tf.placeholder(tf.string, shape=[None])
    dataset = tf.data.TextLineDataset(filenames)
    dataset = dataset.map(lambda string: tf.string_split([string]).values)
    dataset = dataset.map(lambda string: tf.string_to_number(string, out_type=tf.float32))
    if shuffle: dataset = dataset.shuffle(10000)
    if batch_size > 1: dataset = dataset.batch(batch_size)
    iterator = dataset.make_initializable_iterator()
    next_elements = iterator.get_next()
    return filenames, iterator, next_elements

def variable_summaries(var, name=''):
    with tf.variable_scope(name):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
          stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)

def build_depict_graph(inputs, kernel_shape, bias_shape):
    weights = tf.get_variable("weights", kernel_shape,
        initializer=tf.random_normal_initializer())
    biases = tf.get_variable("biases", bias_shape,
        initializer=tf.constant_initializer(0.0))
    weighted_sum = tf.add(tf.matmul(inputs, weights), biases)
    # variable_summaries(weighted_sum, 'weighted_sum')
    # TODO: solve the overflow problem of softmax activations
    # TODO: tf.nn.softmax(weighted_sum) > 1e-32 (func_02, learning_rate=1e-2), or
    # TODO: tf.nn.softmax(weighted_sum) >= 9.99e-31 (func_02, learning_rate=1e-2)
    # return tf.nn.softmax(tf.layers.batch_normalization(weighted_sum))
    return tf.nn.softmax(weighted_sum)

def build_train_graph(defalut_inputs, input_dim, output_dim, func=''):
    inputs = tf.placeholder_with_default(defalut_inputs, shape=[None, input_dim], name='train_input')
    with tf.variable_scope('depict', reuse=False):
        graph = build_depict_graph(inputs, [input_dim, output_dim], [output_dim])
    with tf.variable_scope('target_distribution'):
        P = graph
        N = tf.reshape(tf.reduce_sum(P, axis=0), (-1, output_dim))
        N = tf.div(P, tf.pow(N, 0.5))
        D = tf.reshape(tf.reduce_sum(N, 1), (-1, 1))
        Q = tf.div(N, D)
    with tf.variable_scope('regularization'):
        prior = [1 / float(output_dim)] * output_dim
        U = tf.reshape(tf.constant(prior), (-1, output_dim))
        F = tf.reshape(tf.reduce_mean(Q, axis=0), (-1, output_dim))
    with tf.variable_scope('cost'):
        # TODO: the range of tf.div(Q, P)
        if func == 'func_04':
            with tf.variable_scope('func_04'):
                C = tf.multiply(Q, tf.log(tf.div(Q, P)))
                R = tf.multiply(Q, tf.log(tf.div(F, U)))
                L = tf.reshape(tf.reduce_sum(C + R, axis=1), (-1, 1))
                cost = tf.reduce_mean(L, name='cost')
        elif func == 'func_02':
            with tf.variable_scope('func_02'):
                C = tf.multiply(Q, tf.log(tf.div(Q, P)))
                L = tf.reshape(tf.reduce_sum(C, axis=1), (-1, 1))
                cost = tf.reduce_mean(L, name='cost')
        elif func == 'func_08':
            with tf.variable_scope('func_08'):
                C = -tf.multiply(Q, tf.log(P))
                L = tf.reshape(tf.reduce_sum(C, axis=1), (-1, 1))
                cost = tf.reduce_mean(L, name='cost')
        else:
            cost = 0
            raise Exception('the loss function must be assigned! ')
    variable_summaries(P, 'predicted_distribution')
    variable_summaries(Q, 'target_distribution')
    tf.summary.scalar('training_cost', cost)
    optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(cost)
    return inputs, cost, optimizer

def build_eval_graph(default_inputs, input_dim, output_dim):
    inputs = tf.placeholder_with_default(default_inputs, shape=[None, input_dim], name='eval_input')
    with tf.variable_scope('depict', reuse=False):
        graph = build_depict_graph(inputs, [input_dim, output_dim], [output_dim])
    with tf.variable_scope('eval_output'):
        P = graph
        outputs = tf.argmax(P, axis=1, name='output')
    return inputs, outputs

def build_infer_graph(default_inputs, input_dim, output_dim):
    inputs = tf.placeholder_with_default(default_inputs, shape=[None, input_dim], name='infer_inputs')
    with tf.variable_scope('depict', reuse=False):
        graph = build_depict_graph(inputs, [input_dim, output_dim], [output_dim])
    with tf.variable_scope('infer_output'):
        P = graph
        outputs = tf.argmax(P, axis=1, name='output')
    return inputs, outputs

def build_metrics_graph(scope_name):
    with tf.variable_scope(scope_name):
        err_train = tf.placeholder(tf.float32, [FLAGS.rbfnn_output_dim])
        acc_train = tf.placeholder(tf.float32, ())
        err_test = tf.placeholder(tf.float32, [FLAGS.rbfnn_output_dim])
        acc_test = tf.placeholder(tf.float32, ())
        stsm_train = tf.placeholder(tf.float32, [FLAGS.rbfnn_output_dim])

        err_train_collipse = tf.reduce_mean(err_train)
        err_test_collipse = tf.reduce_mean(err_test)
        stsm_train_collipse = tf.reduce_mean(stsm_train)

        tf.summary.scalar('err_train', err_train_collipse)
        tf.summary.scalar('acc_train', acc_train)
        tf.summary.scalar('stsm_train', stsm_train_collipse)
        tf.summary.scalar('err_test', err_test_collipse)
        tf.summary.scalar('acc_test', acc_test)
    return dict(err_train=err_train,
                acc_train=acc_train,
                stsm_train=stsm_train,
                err_test=err_test,
                acc_test=acc_test)

def metrics_to_metrics(sess, merger, metrics_1, metrics_2):
    summary = sess.run(merger, feed_dict={
        metrics_1['err_train']: metrics_2['err_train'],
        metrics_1['acc_train']: metrics_2['acc_train'],
        metrics_1['stsm_train']: metrics_2['stsm_train'],
        metrics_1['err_test']: metrics_2['err_test'],
        metrics_1['acc_test']: metrics_2['acc_test'],
    })
    return summary

def prepare_file_system():
  # Setup the directory we'll write summaries to for TensorBoard
  if tf.gfile.Exists(FLAGS.summaries_dir):
    tf.gfile.DeleteRecursively(FLAGS.summaries_dir)
  tf.gfile.MakeDirs(FLAGS.summaries_dir)
  if tf.gfile.Exists(FLAGS.checkpoints_dir):
    tf.gfile.DeleteRecursively(FLAGS.checkpoints_dir)
  tf.gfile.MakeDirs(FLAGS.summaries_dir)
  return

# TODO: OOP
class Model(object):
    def __init__(self):
        pass
    def build(self):
        pass

def main():
    tf.logging.set_verbosity(tf.logging.INFO)
    prepare_file_system()

    # FLAGS.eval_step_interval = 1
    # FLAGS.infer_step_interal = 10

    # TODO: OOP
    train_graph = tf.Graph()
    with train_graph.as_default():
        train_filenames, train_iterator, train_elements = \
            build_text_line_reader(shuffle=True, batch_size=FLAGS.train_batch_size)
        train_inputs, train_cost, optimizer = build_train_graph(
            train_elements, FLAGS.depict_input_dim, FLAGS.depict_output_dim, func=FLAGS.loss_function)
        train_saver = tf.train.Saver()
        train_merger = tf.summary.merge_all()
        train_initializer = tf.global_variables_initializer()
        # train_parameters = tf.trainable_variables()
    eval_graph = tf.Graph()
    with eval_graph.as_default():
        eval_filenames, eval_iterator, eval_elements = \
            build_text_line_reader(shuffle=True, batch_size=FLAGS.eval_batch_size)
        eval_inputs, eval_outputs = build_eval_graph(
            eval_elements, FLAGS.depict_input_dim, FLAGS.depict_output_dim)
        eval_saver = tf.train.Saver()
        eval_merger = tf.summary.merge_all()
        eval_initializer = tf.global_variables_initializer()
        # eval_parameters = tf.trainable_variables()
    infer_graph = tf.Graph()
    with infer_graph.as_default():
        infer_filenames, infer_iterator, infer_elements = \
            build_text_line_reader(shuffle=False, batch_size=FLAGS.infer_batch_size)
        infer_inputs, infer_outputs = build_infer_graph(
            infer_elements, FLAGS.depict_input_dim, FLAGS.depict_output_dim)
        rbfnn_metrics = build_metrics_graph('rbfnn')
        # kmeans_metrics = build_metrics_graph('kmeans')
        infer_saver = tf.train.Saver()
        infer_merger = tf.summary.merge_all()
        infer_initializer = tf.global_variables_initializer()

    config = tf.ConfigProto(device_count={"GPU": 1})
    train_sess = tf.Session(graph=train_graph, config=config)
    eval_sess = tf.Session(graph=eval_graph, config=config)
    infer_sess = tf.Session(graph=infer_graph, config=config)

    # train_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/train', train_graph)
    # validation_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/validation', eval_graph)
    # infer_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/inference', infer_graph)

    results = dict()

    train_sess.run(train_initializer)
    # train_sess.run(train_iterator.initializer, feed_dict={train_filenames: [FLAGS.path_to_xtrain]})
    # train_sess.run(train_iterator.initializer)
    import utils
    train_generator = utils.build_data_generator(filenames=[FLAGS.path_to_xtrain], shuffle=True, batch_size=FLAGS.train_batch_size)
    for i in itertools.count():
        if i > FLAGS.how_many_training_steps:
            break

        try:
            # xs_train = train_sess.run(train_elements)
            xs_train = train_generator.next()
            # print(xs_train)
        except tf.errors.OutOfRangeError:
            # train_sess.run(train_iterator.initializer, feed_dict={train_filenames: [FLAGS.path_to_xtrain]})
            train_generator = utils.build_data_generator(filenames=[FLAGS.path_to_xtrain], shuffle=True, batch_size=FLAGS.train_batch_size)
            xs_train = train_sess.run(train_elements)
        # train_summary, _ = train_sess.run([optimizer, train_merger]) #
        _, training_cost, train_summary = train_sess.run([optimizer, train_cost, train_merger], feed_dict={train_inputs: xs_train})

        # train_writer.add_summary(train_summary, i)
        # print('epoch: %6d, training cost: %.8f'%(i, training_cost))
        # time.sleep(1)

        '''
        # if i % FLAGS.eval_step_interval == 0:
        if i % pow(10, len(str(i)) - 1) == 0:
            # print(train_sess.run(train_parameters[0]))
            checkpoint_path = train_saver.save(train_sess, FLAGS.checkpoints_dir + '/checkpoints', global_step=i)
            eval_saver.restore(eval_sess, checkpoint_path)
            # print(eval_sess.run(eval_parameters[0]))
            eval_sess.run(eval_iterator.initializer, feed_dict={eval_filenames: [FLAGS.path_to_xtest]})
            while FLAGS.data_to_eval:
                try:
                    xs_eval = eval_sess.run(eval_elements)
                except tf.errors.OutOfRangeError:
                    # eval_sess.run(eval_iterator.initializer,
                    #                feed_dict={eval_filenames: [r'../../data/x_1000_128.txt']})
                    # xs_eval = eval_sess.run(eval_elements)
                    break
                # training_outputs = eval_sess.run(eval_outputs, feed_dict={eval_inputs: xs_train})
                # evaluation_outputs = eval_sess.run(eval_outputs, feed_dict={eval_inputs: xs_eval})
                evaluation_cost, eval_summary = train_sess.run([train_cost, train_merger], feed_dict={train_inputs: xs_eval})
                tf.logging.info("epoch: %d, training cost: %f"%(i, training_cost))
                tf.logging.info("epoch: %d, evaluation cost: %f" % (i, evaluation_cost))
                # validation_writer.add_summary(eval_summary, i)
                break
        '''

        # if i % FLAGS.infer_step_interval == 0:
        if i % pow(10, len(str(i)) - 1) == 0:
            checkpoint_path = train_saver.save(train_sess, FLAGS.checkpoints_dir + '/checkpoints', global_step=i)
            train_saver.save(train_sess, FLAGS.saved_model_dir + '/checkpoints_' + str(FLAGS.depict_output_dim), global_step=i)
            infer_saver.restore(infer_sess, checkpoint_path)

            infers_train = []
            # infer_sess.run(infer_iterator.initializer, feed_dict={infer_filenames: [FLAGS.path_to_xtrain]})
            infer_generator = utils.build_data_generator(filenames=[FLAGS.path_to_xtrain], shuffle=False, batch_size=FLAGS.infer_batch_size)
            while FLAGS.data_to_infer:
                try:
                    # xs_infer = infer_sess.run(infer_elements)
                    xs_infer = infer_generator.next()
                except tf.errors.OutOfRangeError:
                    break
                ys_infer = infer_sess.run(infer_outputs, feed_dict={infer_inputs: xs_infer})
                infers_train.extend(ys_infer)
            # print(infers_train)

            infers_test = []
            # infer_sess.run(infer_iterator.initializer, feed_dict={infer_filenames: [FLAGS.path_to_xtest]})
            infer_generator = utils.build_data_generator(filenames=[FLAGS.path_to_xtest], shuffle=False, batch_size=FLAGS.infer_batch_size)
            while FLAGS.data_to_infer:
                try:
                    # xs_infer = infer_sess.run(infer_elements)
                    xs_infer = infer_generator.next()
                except tf.errors.OutOfRangeError:
                    break
                ys_infer = infer_sess.run(infer_outputs, feed_dict={infer_inputs: xs_infer})
                # print(xs_infer.shape, xs_infer.flatten())
                # print(ys_infer.shape, ys_infer.flatten())
                infers_test.extend(ys_infer)
            # print(infers_test)
            metrics = classifier.run(infers_train, infers_test, FLAGS)
            pprint.pprint(metrics)
            infer_summary = metrics_to_metrics(infer_sess, infer_merger, rbfnn_metrics, metrics)
            # infer_writer.add_summary(infer_summary, i)
            results[i] = metrics

            # TODO:
            with open('../../results/results.txt', 'a') as f:
                line = list()
                line.extend([FLAGS.rbfnn_num_center, FLAGS.depict_output_dim, i])
                line.extend(metrics['err_train'].tolist())
                line.extend([metrics['acc_train']])
                line.extend(metrics['stsm_train'].tolist())
                line.extend(metrics['err_test'].tolist())
                line.extend([metrics['acc_test']])
                line = [str(item) for item in line]
                line = ' '.join(line)
                f.write(line)
                f.write('\n')
    train_sess.close()
    eval_sess.close()
    infer_sess.close()
    return results

if __name__ == "__main__":
    import time

    class CONFIGS(object):
        path_to_ctrain = r"D:\Users\kingdom\Datasets\UCF\data\pwd\ucf_ctrain_r9.txt"
        path_to_xtrain = r"D:\Users\kingdom\Datasets\UCF\data\pwd\ucf_xtrain_r9.txt"
        path_to_ytrain = r"D:\Users\kingdom\Datasets\UCF\data\pwd\ucf_ytrain_r9.txt"
        path_to_ctest = r"D:\Users\kingdom\Datasets\UCF\data\pwd\ucf_ctest_r9.txt"
        path_to_xtest = r"D:\Users\kingdom\Datasets\UCF\data\pwd\ucf_xtest_r9.txt"
        path_to_ytest = r"D:\Users\kingdom\Datasets\UCF\data\pwd\ucf_ytest_r9.txt"
        summaries_dir = r'../../temp/logs'
        checkpoints_dir = r'../../temp/models'
        saved_model_dir = '../../models'
        how_many_training_steps = 1000
        learning_rate = 0.01
        eval_step_interval = 10
        infer_step_interval = 100
        train_batch_size = 10000
        eval_batch_size = 10000
        infer_batch_size = 10000
        data_to_eval = True
        data_to_infer = True
        loss_function = 'func_04'

        def __init__(self, depict_input_dim=162, depict_output_dim=1024,
                     rbfnn_input_dim=4096, rbfnn_num_center=120, rbfnn_output_dim=10):
            self.depict_input_dim = depict_input_dim
            self.depict_output_dim = depict_output_dim
            self.rbfnn_input_dim = self.depict_output_dim
            self.rbfnn_num_center = rbfnn_num_center
            self.rbfnn_output_dim = rbfnn_output_dim

    results = dict()
    m = 120
    for i in range(7, 16 + 1):
        k = 1 << i
        FLAGS = CONFIGS(depict_output_dim=k, rbfnn_num_center=m)
        print(FLAGS.depict_output_dim)
        metrics = main()
        results[k] = metrics
        # print(results)
        time.sleep(1)
    # print(results)
    #
    # with open('../../results/result.txt', 'w') as f:
    #     for k, v in results.items():
    #         for e, metrics in v.items():
    #             line = list()
    #             line.extend([m, k, e])
    #             line.extend(metrics['err_train'].tolist())
    #             line.extend([metrics['acc_train']])
    #             line.extend(metrics['stsm_train'].tolist())
    #             line.extend(metrics['err_test'].tolist())
    #             line.extend([metrics['acc_test']])
    #             line = ' '.join(line)
    #             f.write(line)