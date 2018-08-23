# -*- coding: utf-8 -*-
# created by kinredon

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from transform2tfrecord import get_data
import time
import cv2
slim = tf.contrib.slim
trunc_normal = lambda stddev: tf.truncated_normal_initializer(0.0, stddev)
from flip_gradient import flip_gradient
from utils import *

# source data
source_train_filename = './tfrecords_inception2/source_train.tfrecords'
source_train_images , source_train_labels = get_data(source_train_filename, batch_size=2088)
source_test_filename = './tfrecords_inception2/source_test.tfrecords'
source_test_images , source_test_labels = get_data(source_test_filename, batch_size=522)

# target data
target_train_filename = './tfrecords_inception2/target_train.tfrecords'
target_train_images , target_train_labels = get_data(target_train_filename, batch_size=1488)
target_test_filename = './tfrecords_inception2/target_test.tfrecords'
target_test_images , target_test_labels = get_data(target_test_filename, batch_size=372)

# num_test = 240
combined_test_imgs = np.vstack([source_test_images, target_test_images])
combined_test_labels = np.vstack([source_test_labels, target_test_labels])
combined_test_domain = np.vstack([np.tile([1., 0.], [522, 1]),
        np.tile([0., 1.], [372, 1])])

def print_activations(t):
    print(t.op.name, ' ', t.get_shape().as_list())



def inception_v3_base(inputs, scope=None):

  with tf.variable_scope(scope, 'InceptionV3', [inputs]):
    with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d],
                        stride=1, padding='VALID'):
      # 299 x 299 x 3
      net = slim.conv2d(inputs, 32, [3, 3], stride=2, scope='Conv2d_1a_3x3')
      # 149 x 149 x 32
      net = slim.conv2d(net, 32, [3, 3], scope='Conv2d_2a_3x3')
      # 147 x 147 x 32
      net = slim.conv2d(net, 64, [3, 3], padding='SAME', scope='Conv2d_2b_3x3')
      # 147 x 147 x 64
      net = slim.max_pool2d(net, [3, 3], stride=2, scope='MaxPool_3a_3x3')
      # 73 x 73 x 64
      net = slim.conv2d(net, 80, [1, 1], scope='Conv2d_3b_1x1')
      # 73 x 73 x 80.
      net = slim.conv2d(net, 192, [3, 3], scope='Conv2d_4a_3x3')
      # 71 x 71 x 192.
      net = slim.max_pool2d(net, [3, 3], stride=2, scope='MaxPool_5a_3x3')
      # 35 x 35 x 192.

    # Inception blocks
    with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d],
                        stride=1, padding='SAME'):
      # mixed: 35 x 35 x 256.
      with tf.variable_scope('Mixed_5b'):
        with tf.variable_scope('Branch_0'):
          branch_0 = slim.conv2d(net, 64, [1, 1], scope='Conv2d_0a_1x1')
        with tf.variable_scope('Branch_1'):
          branch_1 = slim.conv2d(net, 48, [1, 1], scope='Conv2d_0a_1x1')
          branch_1 = slim.conv2d(branch_1, 64, [5, 5], scope='Conv2d_0b_5x5')
        with tf.variable_scope('Branch_2'):
          branch_2 = slim.conv2d(net, 64, [1, 1], scope='Conv2d_0a_1x1')
          branch_2 = slim.conv2d(branch_2, 96, [3, 3], scope='Conv2d_0b_3x3')
          branch_2 = slim.conv2d(branch_2, 96, [3, 3], scope='Conv2d_0c_3x3')
        with tf.variable_scope('Branch_3'):
          branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
          branch_3 = slim.conv2d(branch_3, 32, [1, 1], scope='Conv2d_0b_1x1')
        net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)

      # mixed_1: 35 x 35 x 288.
      with tf.variable_scope('Mixed_5c'):
        with tf.variable_scope('Branch_0'):
          branch_0 = slim.conv2d(net, 64, [1, 1], scope='Conv2d_0a_1x1')
        with tf.variable_scope('Branch_1'):
          branch_1 = slim.conv2d(net, 48, [1, 1], scope='Conv2d_0b_1x1')
          branch_1 = slim.conv2d(branch_1, 64, [5, 5], scope='Conv_1_0c_5x5')
        with tf.variable_scope('Branch_2'):
          branch_2 = slim.conv2d(net, 64, [1, 1], scope='Conv2d_0a_1x1')
          branch_2 = slim.conv2d(branch_2, 96, [3, 3], scope='Conv2d_0b_3x3')
          branch_2 = slim.conv2d(branch_2, 96, [3, 3], scope='Conv2d_0c_3x3')
        with tf.variable_scope('Branch_3'):
          branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
          branch_3 = slim.conv2d(branch_3, 64, [1, 1], scope='Conv2d_0b_1x1')
        net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)

      # mixed_2: 35 x 35 x 288.
      with tf.variable_scope('Mixed_5d'):
        with tf.variable_scope('Branch_0'):
          branch_0 = slim.conv2d(net, 64, [1, 1], scope='Conv2d_0a_1x1')
        with tf.variable_scope('Branch_1'):
          branch_1 = slim.conv2d(net, 48, [1, 1], scope='Conv2d_0a_1x1')
          branch_1 = slim.conv2d(branch_1, 64, [5, 5], scope='Conv2d_0b_5x5')
        with tf.variable_scope('Branch_2'):
          branch_2 = slim.conv2d(net, 64, [1, 1], scope='Conv2d_0a_1x1')
          branch_2 = slim.conv2d(branch_2, 96, [3, 3], scope='Conv2d_0b_3x3')
          branch_2 = slim.conv2d(branch_2, 96, [3, 3], scope='Conv2d_0c_3x3')
        with tf.variable_scope('Branch_3'):
          branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
          branch_3 = slim.conv2d(branch_3, 64, [1, 1], scope='Conv2d_0b_1x1')
        net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)

      # mixed_3: 17 x 17 x 768.
      with tf.variable_scope('Mixed_6a'):
        with tf.variable_scope('Branch_0'):
          branch_0 = slim.conv2d(net, 384, [3, 3], stride=2,
                                 padding='VALID', scope='Conv2d_1a_1x1')
        with tf.variable_scope('Branch_1'):
          branch_1 = slim.conv2d(net, 64, [1, 1], scope='Conv2d_0a_1x1')
          branch_1 = slim.conv2d(branch_1, 96, [3, 3], scope='Conv2d_0b_3x3')
          branch_1 = slim.conv2d(branch_1, 96, [3, 3], stride=2,
                                 padding='VALID', scope='Conv2d_1a_1x1')
        with tf.variable_scope('Branch_2'):
          branch_2 = slim.max_pool2d(net, [3, 3], stride=2, padding='VALID',
                                     scope='MaxPool_1a_3x3')
        net = tf.concat([branch_0, branch_1, branch_2], 3)

      # mixed4: 17 x 17 x 768.
      with tf.variable_scope('Mixed_6b'):
        with tf.variable_scope('Branch_0'):
          branch_0 = slim.conv2d(net, 192, [1, 1], scope='Conv2d_0a_1x1')
        with tf.variable_scope('Branch_1'):
          branch_1 = slim.conv2d(net, 128, [1, 1], scope='Conv2d_0a_1x1')
          branch_1 = slim.conv2d(branch_1, 128, [1, 7], scope='Conv2d_0b_1x7')
          branch_1 = slim.conv2d(branch_1, 192, [7, 1], scope='Conv2d_0c_7x1')
        with tf.variable_scope('Branch_2'):
          branch_2 = slim.conv2d(net, 128, [1, 1], scope='Conv2d_0a_1x1')
          branch_2 = slim.conv2d(branch_2, 128, [7, 1], scope='Conv2d_0b_7x1')
          branch_2 = slim.conv2d(branch_2, 128, [1, 7], scope='Conv2d_0c_1x7')
          branch_2 = slim.conv2d(branch_2, 128, [7, 1], scope='Conv2d_0d_7x1')
          branch_2 = slim.conv2d(branch_2, 192, [1, 7], scope='Conv2d_0e_1x7')
        with tf.variable_scope('Branch_3'):
          branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
          branch_3 = slim.conv2d(branch_3, 192, [1, 1], scope='Conv2d_0b_1x1')
        net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)

      # mixed_5: 17 x 17 x 768.
      with tf.variable_scope('Mixed_6c'):
        with tf.variable_scope('Branch_0'):
          branch_0 = slim.conv2d(net, 192, [1, 1], scope='Conv2d_0a_1x1')
        with tf.variable_scope('Branch_1'):
          branch_1 = slim.conv2d(net, 160, [1, 1], scope='Conv2d_0a_1x1')
          branch_1 = slim.conv2d(branch_1, 160, [1, 7], scope='Conv2d_0b_1x7')
          branch_1 = slim.conv2d(branch_1, 192, [7, 1], scope='Conv2d_0c_7x1')
        with tf.variable_scope('Branch_2'):
          branch_2 = slim.conv2d(net, 160, [1, 1], scope='Conv2d_0a_1x1')
          branch_2 = slim.conv2d(branch_2, 160, [7, 1], scope='Conv2d_0b_7x1')
          branch_2 = slim.conv2d(branch_2, 160, [1, 7], scope='Conv2d_0c_1x7')
          branch_2 = slim.conv2d(branch_2, 160, [7, 1], scope='Conv2d_0d_7x1')
          branch_2 = slim.conv2d(branch_2, 192, [1, 7], scope='Conv2d_0e_1x7')
        with tf.variable_scope('Branch_3'):
          branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
          branch_3 = slim.conv2d(branch_3, 192, [1, 1], scope='Conv2d_0b_1x1')
        net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)
      # mixed_6: 17 x 17 x 768.
      with tf.variable_scope('Mixed_6d'):
        with tf.variable_scope('Branch_0'):
          branch_0 = slim.conv2d(net, 192, [1, 1], scope='Conv2d_0a_1x1')
        with tf.variable_scope('Branch_1'):
          branch_1 = slim.conv2d(net, 160, [1, 1], scope='Conv2d_0a_1x1')
          branch_1 = slim.conv2d(branch_1, 160, [1, 7], scope='Conv2d_0b_1x7')
          branch_1 = slim.conv2d(branch_1, 192, [7, 1], scope='Conv2d_0c_7x1')
        with tf.variable_scope('Branch_2'):
          branch_2 = slim.conv2d(net, 160, [1, 1], scope='Conv2d_0a_1x1')
          branch_2 = slim.conv2d(branch_2, 160, [7, 1], scope='Conv2d_0b_7x1')
          branch_2 = slim.conv2d(branch_2, 160, [1, 7], scope='Conv2d_0c_1x7')
          branch_2 = slim.conv2d(branch_2, 160, [7, 1], scope='Conv2d_0d_7x1')
          branch_2 = slim.conv2d(branch_2, 192, [1, 7], scope='Conv2d_0e_1x7')
        with tf.variable_scope('Branch_3'):
          branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
          branch_3 = slim.conv2d(branch_3, 192, [1, 1], scope='Conv2d_0b_1x1')
        net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)

      # mixed_7: 17 x 17 x 768.
      with tf.variable_scope('Mixed_6e'):
        with tf.variable_scope('Branch_0'):
          branch_0 = slim.conv2d(net, 192, [1, 1], scope='Conv2d_0a_1x1')
        with tf.variable_scope('Branch_1'):
          branch_1 = slim.conv2d(net, 192, [1, 1], scope='Conv2d_0a_1x1')
          branch_1 = slim.conv2d(branch_1, 192, [1, 7], scope='Conv2d_0b_1x7')
          branch_1 = slim.conv2d(branch_1, 192, [7, 1], scope='Conv2d_0c_7x1')
        with tf.variable_scope('Branch_2'):
          branch_2 = slim.conv2d(net, 192, [1, 1], scope='Conv2d_0a_1x1')
          branch_2 = slim.conv2d(branch_2, 192, [7, 1], scope='Conv2d_0b_7x1')
          branch_2 = slim.conv2d(branch_2, 192, [1, 7], scope='Conv2d_0c_1x7')
          branch_2 = slim.conv2d(branch_2, 192, [7, 1], scope='Conv2d_0d_7x1')
          branch_2 = slim.conv2d(branch_2, 192, [1, 7], scope='Conv2d_0e_1x7')
        with tf.variable_scope('Branch_3'):
          branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
          branch_3 = slim.conv2d(branch_3, 192, [1, 1], scope='Conv2d_0b_1x1')
        net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)

      # mixed_8: 8 x 8 x 1280.
      with tf.variable_scope('Mixed_7a'):
        with tf.variable_scope('Branch_0'):
          branch_0 = slim.conv2d(net, 192, [1, 1], scope='Conv2d_0a_1x1')
          branch_0 = slim.conv2d(branch_0, 320, [3, 3], stride=2,
                                 padding='VALID', scope='Conv2d_1a_3x3')
        with tf.variable_scope('Branch_1'):
          branch_1 = slim.conv2d(net, 192, [1, 1], scope='Conv2d_0a_1x1')
          branch_1 = slim.conv2d(branch_1, 192, [1, 7], scope='Conv2d_0b_1x7')
          branch_1 = slim.conv2d(branch_1, 192, [7, 1], scope='Conv2d_0c_7x1')
          branch_1 = slim.conv2d(branch_1, 192, [3, 3], stride=2,
                                 padding='VALID', scope='Conv2d_1a_3x3')
        with tf.variable_scope('Branch_2'):
          branch_2 = slim.max_pool2d(net, [3, 3], stride=2, padding='VALID',
                                     scope='MaxPool_1a_3x3')
        net = tf.concat([branch_0, branch_1, branch_2], 3)
      # mixed_9: 8 x 8 x 2048.
      with tf.variable_scope('Mixed_7b'):
        with tf.variable_scope('Branch_0'):
          branch_0 = slim.conv2d(net, 320, [1, 1], scope='Conv2d_0a_1x1')
        with tf.variable_scope('Branch_1'):
          branch_1 = slim.conv2d(net, 384, [1, 1], scope='Conv2d_0a_1x1')
          branch_1 = tf.concat([
              slim.conv2d(branch_1, 384, [1, 3], scope='Conv2d_0b_1x3'),
              slim.conv2d(branch_1, 384, [3, 1], scope='Conv2d_0b_3x1')], 3)
        with tf.variable_scope('Branch_2'):
          branch_2 = slim.conv2d(net, 448, [1, 1], scope='Conv2d_0a_1x1')
          branch_2 = slim.conv2d(
              branch_2, 384, [3, 3], scope='Conv2d_0b_3x3')
          branch_2 = tf.concat([
              slim.conv2d(branch_2, 384, [1, 3], scope='Conv2d_0c_1x3'),
              slim.conv2d(branch_2, 384, [3, 1], scope='Conv2d_0d_3x1')], 3)
        with tf.variable_scope('Branch_3'):
          branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
          branch_3 = slim.conv2d(
              branch_3, 192, [1, 1], scope='Conv2d_0b_1x1')
        net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)

      # mixed_10: 8 x 8 x 2048.
      with tf.variable_scope('Mixed_7c'):
        with tf.variable_scope('Branch_0'):
          branch_0 = slim.conv2d(net, 320, [1, 1], scope='Conv2d_0a_1x1')
        with tf.variable_scope('Branch_1'):
          branch_1 = slim.conv2d(net, 384, [1, 1], scope='Conv2d_0a_1x1')
          branch_1 = tf.concat([
              slim.conv2d(branch_1, 384, [1, 3], scope='Conv2d_0b_1x3'),
              slim.conv2d(branch_1, 384, [3, 1], scope='Conv2d_0c_3x1')], 3)
        with tf.variable_scope('Branch_2'):
          branch_2 = slim.conv2d(net, 448, [1, 1], scope='Conv2d_0a_1x1')
          branch_2 = slim.conv2d(
              branch_2, 384, [3, 3], scope='Conv2d_0b_3x3')
          branch_2 = tf.concat([
              slim.conv2d(branch_2, 384, [1, 3], scope='Conv2d_0c_1x3'),
              slim.conv2d(branch_2, 384, [3, 1], scope='Conv2d_0d_3x1')], 3)
        with tf.variable_scope('Branch_3'):
          branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
          branch_3 = slim.conv2d(
              branch_3, 192, [1, 1], scope='Conv2d_0b_1x1')
        net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)
      return net


def inception_v3_feature_extractor(inputs,
                                    num_classes=8,
                                    is_training=True,
                                    reuse=None,
                                    scope='InceptionV3_feature_extractor'):
    with tf.variable_scope(scope, 'InceptionV3_feature_extractor_', [inputs, num_classes],
                           reuse=reuse) as scope:
        with slim.arg_scope([slim.batch_norm, slim.dropout],
                            is_training=is_training):
            net = inception_v3_base(inputs, scope=scope)
            print(net.get_shape().as_list())

            net = slim.avg_pool2d(net, [8, 8], padding='VALID',
                                  scope='AvgPool_1a_8x8')
            return net


def inception_v3_label_prediction(inputs,
                                num_classes=8,
                                is_training=True,
                                dropout_keep_prob=0.8,
                                spatial_squeeze=True,
                                reuse=None,
                                scope='InceptionV3_label_prediction'):
    with tf.variable_scope(scope, 'InceptionV3_label_prediction_', [inputs, num_classes],
                           reuse=reuse) as scope:
        with slim.arg_scope([slim.batch_norm, slim.dropout],
                            is_training=is_training):
            # Final pooling and prediction
            with tf.variable_scope('Logits'):
                print_activations(inputs)
                # 1 x 1 x 2048
                net = slim.dropout(inputs, keep_prob=dropout_keep_prob, scope='Dropout_1b')
                # 2048
                print_activations(net)
                net = slim.conv2d(net, 100, [1, 1], activation_fn=None,
                                  normalizer_fn=None, scope='Conv2d_1c_1x1')
                logits = slim.conv2d(net, num_classes, [1, 1], activation_fn=None,
                                     normalizer_fn=None, scope='Conv2d_2c_1x1')
                if spatial_squeeze:
                    logits = tf.squeeze(logits, [1, 2], name='SpatialSqueeze')
                    # 8
    return logits



batch_size = 64
log_dir = 'summary_inception'

class Model(object):

    def __init__(self, keep_prob):
        self._build_model(keep_prob)

    def _build_model(self, keep_prob):
        self.X = tf.placeholder(tf.float32, [None, 299, 299, 3])  # 输入
        self.y = tf.placeholder(tf.float32, [None, 8])  # 标签
        self.domain = tf.placeholder(tf.float32, [None, 2])  # 域标签
        self.l = tf.placeholder(tf.float32, [])  # 反转
        self.train = tf.placeholder(tf.bool, [])  #

        # CNN model for feature extraction
        with tf.variable_scope('feature_extractor'):
            net = inception_v3_feature_extractor(self.X, 8)

        # The domain-invariant feature
        self.feature = net
        # The domain-invariant feature
        print_activations(self.feature)
        with tf.variable_scope('label_predictor'):
            # Switches to route target examples (second half of batch) differently
            # depending on train or test mode.
            all_features = lambda: self.feature
            source_features = lambda: tf.slice(self.feature, [0, 0, 0, 0], [batch_size // 2, 1, 1, -1])  # back_size的一半
            classify_feats = tf.cond(self.train, source_features, all_features)  # 如果train=true，返回source_features
            print_activations(classify_feats)

            all_labels = lambda: self.y
            source_labels = lambda: tf.slice(self.y, [0, 0], [batch_size // 2, -1])
            self.classify_labels = tf.cond(self.train, source_labels, all_labels)  # 与前面同理

            logits = inception_v3_label_prediction(classify_feats, 8, dropout_keep_prob=keep_prob)

            self.pred = tf.nn.softmax(logits)
            self.pred_loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=self.classify_labels)

        # Small MLP for domain prediction with adversarial loss
        with tf.variable_scope('domain_predictor'):
            self.feature = tf.squeeze(self.feature, [1, 2], name='SpatialSqueeze')
            # Flip the gradient when backpropagating through this operation
            feat = flip_gradient(self.feature, self.l)

            d_W_fc0 = weight_variable([2048, 1000])
            d_b_fc0 = bias_variable([1000])
            d_h_fc0 = tf.nn.relu(tf.matmul(feat, d_W_fc0) + d_b_fc0)

            d_W_fc1 = weight_variable([1000, 100])
            d_b_fc1 = bias_variable([100])
            d_h_fc1 = tf.matmul(d_h_fc0, d_W_fc1) + d_b_fc1

            d_W_fc2 = weight_variable([100, 2])
            d_b_fc2 = bias_variable([2])
            d_logits = tf.matmul(d_h_fc1, d_W_fc2) + d_b_fc2

            self.domain_pred = tf.nn.softmax(d_logits)
            self.domain_loss = tf.nn.softmax_cross_entropy_with_logits(logits=d_logits, labels=self.domain)

# Build the model graph
graph = tf.get_default_graph()
with graph.as_default():
    model = Model(0.8)

    learning_rate = tf.placeholder(tf.float32, [])

    pred_loss = tf.reduce_mean(model.pred_loss)
    pred_loss_summary = tf.summary.scalar('pred_loss', pred_loss)
    domain_loss = tf.reduce_mean(model.domain_loss)
    total_loss = pred_loss + domain_loss
    total_loss_summary = tf.summary.scalar('total_loss', total_loss)

    regular_train_op = tf.train.AdamOptimizer(learning_rate).minimize(pred_loss)
    dann_train_op = tf.train.AdamOptimizer(learning_rate).minimize(total_loss)
    # regular_train_op = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(pred_loss)
    # dann_train_op = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(total_loss)

    # Evaluation
    correct_label_pred = tf.equal(tf.argmax(model.classify_labels, 1), tf.argmax(model.pred, 1))
    label_acc = tf.reduce_mean(tf.cast(correct_label_pred, tf.float32))
    tf.summary.scalar('label_accuracy', label_acc)

    correct_domain_pred = tf.equal(tf.argmax(model.domain, 1), tf.argmax(model.domain_pred, 1))
    domain_acc = tf.reduce_mean(tf.cast(correct_domain_pred, tf.float32))

    tf.summary.scalar('domain_accuracy', domain_acc)
    merged = tf.summary.merge_all()


def train_and_evaluate(training_mode, graph, model, num_steps=8600, verbose=False):
    """Helper to run the model with different training modes."""

    with tf.Session(graph=graph) as sess:

        train_writer = tf.summary.FileWriter(log_dir + '/train', sess.graph)
        test_source_writer = tf.summary.FileWriter(log_dir + '/test_source')
        test_target_writer = tf.summary.FileWriter(log_dir + '/test_target')

        tf.global_variables_initializer().run()

        # Batch generators
        gen_source_batch = batch_generator(
            [source_train_images, source_train_labels], batch_size // 2)
        gen_target_batch = batch_generator(
            [target_train_images, target_train_labels], batch_size // 2)
        gen_source_only_batch = batch_generator(
            [source_train_images, source_train_labels], batch_size)
        gen_target_only_batch = batch_generator(
            [target_train_images, target_train_labels], batch_size)
        gen_source_test = batch_generator(
            [source_test_images, source_test_labels], batch_size)
        gen_target_test = batch_generator(
            [target_test_images, target_test_labels], batch_size)

        domain_labels = np.vstack([np.tile([1., 0.], [batch_size // 2, 1]),
                                   np.tile([0., 1.], [batch_size // 2, 1])])

        # Training loop
        for i in range(num_steps):

            # Adaptation param and learning rate schedule as described in the paper
            p = float(i) / num_steps
            l = 2. / (1. + np.exp(-10. * p)) - 1
            # lr = 0.01 / (1. + 10 * p) ** 0.75
            lr = 0.00001
            # Training step
            if training_mode == 'dann':

                # X0, y0 = get_data(source_train_filename, batch_size // 2)
                # X1, y1 = get_data(target_train_filename, batch_size // 2)
                X0, y0 = next(gen_source_batch)
                X1, y1 = next(gen_target_batch)
                X = np.vstack([X0, X1])
                y = np.vstack([y0, y1])
                start_time = time.time()
                _, batch_loss, dloss, ploss, d_acc, p_acc = sess.run(
                    [dann_train_op, total_loss, domain_loss, pred_loss, domain_acc, label_acc],
                    feed_dict={model.X: X, model.y: y, model.domain: domain_labels,
                               model.train: True, model.l: l, learning_rate: lr})
                duration = time.time() - start_time

                if verbose and i % 10 == 0:
                # if verbose:
                    print('step {}, loss: {}  d_acc: {}  p_acc: {}  p: {}  l: {}  lr: {} duration: {}'.format(
                        i, batch_loss, d_acc, p_acc, p, l, lr, duration))
                    source_test_images_part, source_test_labels_part = next(gen_source_test)
                    target_test_images_part, target_test_labels_part = next(gen_target_test)
                    X1, y1 = next(gen_target_batch)
                    source_acc = sess.run([label_acc],
                                          feed_dict={model.X: source_test_images_part, model.y: source_test_labels_part,
                                                     model.train: False})

                    target_acc = sess.run([label_acc],
                                          feed_dict={model.X: target_test_images_part, model.y: target_test_labels_part,
                                                     model.train: False})
                    print("source_acc:{}\t target_acc:{}".format(source_acc, target_acc))


            elif training_mode == 'source':
                # X, y = get_data(source_train_filename, batch_size)
                X, y = next(gen_source_only_batch)
                start_time = time.time()
                summary, _, batch_loss, p_acc = sess.run([merged, regular_train_op, pred_loss, label_acc],
                                         feed_dict={model.X: X, model.y: y, model.domain: domain_labels, model.train: False,
                                                    model.l: l, learning_rate: lr})
                duration = time.time() - start_time
                if verbose and i % 10 == 0:
                # if verbose:
                    print('step {}, loss: {}  p_acc: {}  p: {}  l: {}  lr: {} duration:{}'.format(
                        i, batch_loss, p_acc, p, l, lr, duration))
                    train_writer.add_summary(summary, i)

                    source_test_images_part, source_test_labels_part = next(gen_source_test)
                    target_test_images_part, target_test_labels_part = next(gen_target_test)
                    X1, y1 = next(gen_target_batch)
                    summary, source_acc = sess.run([merged, label_acc],
                                          feed_dict={model.X: source_test_images_part, model.y: source_test_labels_part,
                                                     model.domain: combined_test_domain[0:64], model.train: False})
                    test_source_writer.add_summary(summary, i)

                    summary, target_acc = sess.run([merged, label_acc],
                                          feed_dict={model.X: target_test_images_part, model.y: target_test_labels_part,
                                                     model.domain: combined_test_domain[0:64], model.train: False})
                    print("source_acc:{}\t target_acc:{}".format(source_acc, target_acc))
                    test_target_writer.add_summary(summary, i)

            elif training_mode == 'target':
                # X, y = get_data(target_train_filename, batch_size)
                X, y = next(gen_target_only_batch)
                _, batch_loss = sess.run([regular_train_op, pred_loss],
                                         feed_dict={model.X: X, model.y: y, model.train: False,
                                                    model.l: l, learning_rate: lr})

        gen_source_test = batch_generator(
            [source_test_images, source_test_labels], batch_size, shuffle=False)
        gen_target_test = batch_generator(
            [target_test_images, target_test_labels], batch_size, shuffle=False)
        source_test_images_part, source_test_labels_part = next(gen_source_test)
        target_test_images_part, target_test_labels_part = next(gen_target_test)
        # Compute final evaluation on test data
        batch_count = int(522 / batch_size)
        source_acc = 0
        for i in range(batch_count):
            source_acc += sess.run(label_acc,
                            feed_dict={model.X: source_test_images_part, model.y: source_test_labels_part,
                                         model.train: False})
        source_acc /= batch_count

        batch_count = int(372 / batch_size)
        target_acc = 0
        for i in range(batch_count):
            target_acc += sess.run(label_acc,
                                  feed_dict={model.X: target_test_images_part, model.y: target_test_labels_part,
                                             model.train: False})
        target_acc /= batch_count

        test_domain_acc = sess.run(domain_acc,
                                   feed_dict={model.X: combined_test_imgs[0:200],
                                              model.domain: combined_test_domain[0:200], model.l: 1.0})

    train_writer.close()
    test_source_writer.close()
    test_target_writer.close()
    return source_acc, target_acc, test_domain_acc


if __name__ == "__main__":

    # print('\nDomain adaptation training')
    # source_acc, target_acc, d_acc = train_and_evaluate('dann', graph, model, verbose=True)
    # print('Source accuracy:', source_acc)
    # print('Target accuracy:', target_acc)
    # print('Domain accuracy:', d_acc)

    print('\nSource only training')
    source_acc, target_acc, _ = train_and_evaluate('source', graph, model, verbose=True)
    print('Source accuracy:', source_acc)
    print('Target accuracy:', target_acc)

