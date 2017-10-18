#!/usr/bin/env python
# coding=utf-8

import numpy as np 
import tensorflow as tf


def save_graph_to_file(sess, graph, graph_file_name):
      output_graph_def = graph_util.convert_variables_to_constants(
      sess, graph.as_graph_def(), [FLAGS.final_tensor_name])
  with gfile.FastGFile(graph_file_name, 'wb') as f:
    f.write(output_graph_def.SerializeToString())
  return

def create_model_graph(model_info):
    with tf.Graph().as_default() as graph:
        model_path = os.path.join(FLAGS.model_dir, model_info['model_file_name'])
        with gfile.FastGFile(model_path, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        bottleneck_tensor, resized_input_tensor = (tf.import_graph_def(
            graph_def,
            name='',
            return_elements=[
                model_info['bottleneck_tensor_name'],
                model_info['resized_input_tensor_name'],
            ]))
    return graph, bottleneck_tensor, resized_input_tensor

def main(): 
    pass 