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
import datetime as dt

import numpy as np 
import tensorflow as tf 
from tensorflow.python.framework import graph_util
from sklearn.metrics import normalized_mutual_info_score as sklnmi

def read_data_from_text(filenames=None, shuffle=False):
  filenames = tf.placeholder(tf.string, shape=[None])
  dataset = tf.data.TextLineDataset(filenames)
  # dataset = dataset.map(lambda item : tf.string_split(item, " "))
  # dataset = dataset.map(lambda item : tf.string_to_number(item))
  # dataset = dataset.map(lambda item : tf.to_float(item))

  '''
  map, shuffle, batch
  '''

  if shuffle:
    pass

  iterator = dataset.make_initializable_iterator()
  next_element = iterator.get_next()
  return filenames, iterator, next_element

def build_depict_graph():
    pass 
def build_train_graph(): 
    pass 
def build_eval_graph(): 
    pass 
def build_infer_graph(): 
    pass 
#
# train_graph = tf.Graph()
# eval_graph = tf.Graph()
# infer_graph = tf.Graph()
#
# with train_graph.as_default():
#   train_iterator = ...
#   train_model = BuildTrainModel(train_iterator)
#   initializer = tf.global_variables_initializer()
#
# with eval_graph.as_default():
#   eval_iterator = ...
#   eval_model = BuildEvalModel(eval_iterator)
#
# with infer_graph.as_default():
#   infer_iterator, infer_inputs = ...
#   infer_model = BuildInferenceModel(infer_iterator)
#
# checkpoints_path = "/tmp/model/checkpoints"
#
# train_sess = tf.Session(graph=train_graph)
# eval_sess = tf.Session(graph=eval_graph)
# infer_sess = tf.Session(graph=infer_graph)
#
# train_sess.run(initializer)
# train_sess.run(train_iterator.initializer)
#
# for i in itertools.count():
#
#   train_model.train(train_sess)
#
#   if i % EVAL_STEPS == 0:
#     checkpoint_path = train_model.saver.save(train_sess, checkpoints_path, global_step=i)
#     eval_model.saver.restore(eval_sess, checkpoint_path)
#     eval_sess.run(eval_iterator.initializer)
#     while data_to_eval:
#       eval_model.eval(eval_sess)
#
#   if i % INFER_STEPS == 0:
#     checkpoint_path = train_model.saver.save(train_sess, checkpoints_path, global_step=i)
#     infer_model.saver.restore(infer_sess, checkpoint_path)
#     infer_sess.run(infer_iterator.initializer, feed_dict={infer_inputs: infer_input_data})
#     while data_to_infer:
#       infer_model.infer(infer_sess)

if __name__ == "__main__":
  # train_filenames, train_iterator, train_elements = read_data_from_text(shuffle=True)
  #
  # sess = tf.Session()
  # sess.run(train_iterator.initializer, feed_dict={train_filenames:[r'F:\Users\Kingdom\Desktop\GIT\DC\data\kth_xrand_r9_reduce.txt']})
  # while True:
  #   try:
  #     print(sess.run(train_elements))
  #   except Exception:
  #     break
  #   break

  tfstr = tf.constant([b'2012 2012 2012'])
  tfstr = tf.string_split(tfstr)
  tfstr = tf.map_fn(lambda x:tf.string_to_number(x), tfstr.values)
  # tfstr = tf.string_to_number(tfstr)
  sess = tf.Session()
  print(sess.run(tfstr))