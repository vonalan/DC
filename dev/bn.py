#!/usr/bin/env python
# coding=utf-8

import Keras as K

class BatchNorminalization(object):
    def __init__(self):
        self.running_mean = 0.0
        self.running_std = 1.0

    def bacth_norminalization(self,X):
        update_moving_mean = moving_averages.assign_moving_average(moving_mean, mean, BN_DECAY)
        update_moving_variance = moving_averages.assign_moving_average(moving_variance, variance, BN_DECAY)
        tf.add_to_collection(UPDATE_OPS_COLLECTION, update_moving_mean)
        tf.add_to_collection(UPDATE_OPS_COLLECTION, update_moving_variance)
        mean, variance = control_flow_ops.cond(['is_training'], lambda: (mean, variance),
                                               lambda: (moving_mean, moving_variance))