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

from flip_gradient import flip_gradient
from utils import *


def print_activations(t):
    print(t.op.name, ' ', t.get_shape().as_list())

# source data
source_train_filename = './tfrecords/source_train.tfrecords'
source_train_images , source_train_labels = get_data(source_train_filename, batch_size=2088)
source_test_filename = './tfrecords/source_test.tfrecords'
source_test_images , source_test_labels = get_data(source_test_filename, batch_size=522)

# target data
target_train_filename = './tfrecords/target_train.tfrecords'
target_train_images , target_train_labels = get_data(target_train_filename, batch_size=1488)
target_test_filename = './tfrecords/target_test.tfrecords'
target_test_images , target_test_labels = get_data(target_test_filename, batch_size=372)

# num_test = 240
combined_test_imgs = np.vstack([source_test_images, target_test_images])
combined_test_labels = np.vstack([source_test_labels, target_test_labels])
combined_test_domain = np.vstack([np.tile([1., 0.], [522, 1]),
        np.tile([0., 1.], [372, 1])])

batch_size = 64

class MNISTModel(object):
    """Simple MNIST domain adaptation model."""

    def __init__(self):
        self._build_model()

    def _build_model(self):
        self.X = tf.placeholder(tf.float32, [None, 224, 224, 3])  # 输入
        self.y = tf.placeholder(tf.float32, [None, 8])  # 标签
        self.domain = tf.placeholder(tf.float32, [None, 2])  # 域标签
        self.l = tf.placeholder(tf.float32, [])  # 反转
        self.train = tf.placeholder(tf.bool, [])  #


        # CNN model for feature extraction
        with tf.variable_scope('feature_extractor'):
            parameters = []
            # conv1
            with tf.name_scope('conv1') as scope:
                kernel = tf.Variable(tf.truncated_normal([11, 11, 3, 64], dtype=tf.float32,
                                                         stddev=1e-1), name='weights')
                conv = tf.nn.conv2d(self.X, kernel, [1, 4, 4, 1], padding='SAME')
                biases = tf.Variable(tf.constant(0.1, shape=[64], dtype=tf.float32),
                                     trainable=True, name='biases')
                bias = tf.nn.bias_add(conv, biases)
                conv1 = tf.nn.relu(bias, name=scope)
                print_activations(conv1)
                parameters += [kernel, biases]


                # pool1
            lrn1 = tf.nn.lrn(conv1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='lrn1')
            pool1 = tf.nn.max_pool(lrn1,
                                   ksize=[1, 3, 3, 1],
                                   strides=[1, 2, 2, 1],
                                   padding='VALID',
                                   name='pool1')
            print_activations(pool1)

            # conv2
            with tf.name_scope('conv2') as scope:
                kernel = tf.Variable(tf.truncated_normal([5, 5, 64, 192], dtype=tf.float32,
                                                         stddev=1e-1), name='weights')
                conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
                biases = tf.Variable(tf.constant(0.1, shape=[192], dtype=tf.float32),
                                     trainable=True, name='biases')
                bias = tf.nn.bias_add(conv, biases)
                conv2 = tf.nn.relu(bias, name=scope)
                parameters += [kernel, biases]
            print_activations(conv2)

            # pool2
            lrn2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='lrn2')
            pool2 = tf.nn.max_pool(lrn2,
                                   ksize=[1, 3, 3, 1],
                                   strides=[1, 2, 2, 1],
                                   padding='VALID',
                                   name='pool2')
            print_activations(pool2)

            # conv3
            with tf.name_scope('conv3') as scope:
                kernel = tf.Variable(tf.truncated_normal([3, 3, 192, 384],
                                                         dtype=tf.float32,
                                                         stddev=1e-1), name='weights')
                conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
                biases = tf.Variable(tf.constant(0.1, shape=[384], dtype=tf.float32),
                                     trainable=True, name='biases')
                bias = tf.nn.bias_add(conv, biases)
                conv3 = tf.nn.relu(bias, name=scope)
                parameters += [kernel, biases]
                print_activations(conv3)

                # conv4
            with tf.name_scope('conv4') as scope:
                kernel = tf.Variable(tf.truncated_normal([3, 3, 384, 256],
                                                         dtype=tf.float32,
                                                         stddev=1e-1), name='weights')
                conv = tf.nn.conv2d(conv3, kernel, [1, 1, 1, 1], padding='SAME')
                biases = tf.Variable(tf.constant(0.1, shape=[256], dtype=tf.float32),
                                     trainable=True, name='biases')
                bias = tf.nn.bias_add(conv, biases)
                conv4 = tf.nn.relu(bias, name=scope)
                parameters += [kernel, biases]
                print_activations(conv4)

                # conv5
            with tf.name_scope('conv5') as scope:
                kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256],
                                                         dtype=tf.float32,
                                                         stddev=1e-1), name='weights')
                conv = tf.nn.conv2d(conv4, kernel, [1, 1, 1, 1], padding='SAME')
                biases = tf.Variable(tf.constant(0.1, shape=[256], dtype=tf.float32),
                                     trainable=True, name='biases')
                bias = tf.nn.bias_add(conv, biases)
                conv5 = tf.nn.relu(bias, name=scope)
                parameters += [kernel, biases]
                print_activations(conv5)

                # pool5
            pool5 = tf.nn.max_pool(conv5,
                                   ksize=[1, 3, 3, 1],
                                   strides=[1, 2, 2, 1],
                                   padding='VALID',
                                   name='pool5')
            print_activations(pool5)

            # The domain-invariant feature
            self.feature = tf.reshape(pool5, [-1, 6 * 6 * 256])

        with tf.variable_scope('label_predictor'):
            # Switches to route target examples (second half of batch) differently
            # depending on train or test mode.
            all_features = lambda: self.feature
            source_features = lambda: tf.slice(self.feature, [0, 0], [batch_size // 2, -1])  # back_size的一半
            classify_feats = tf.cond(self.train, source_features, all_features)  # 如果train=true，返回source_features

            all_labels = lambda: self.y
            source_labels = lambda: tf.slice(self.y, [0, 0], [batch_size // 2, -1])
            self.classify_labels = tf.cond(self.train, source_labels, all_labels)  # 与前面同理

            W_fc0 = weight_variable([6 * 6 * 256, 4096])
            b_fc0 = bias_variable([4096])
            h_fc0 = tf.nn.relu(tf.matmul(classify_feats, W_fc0) + b_fc0)

            W_fc1 = weight_variable([4096, 4096])
            b_fc1 = bias_variable([4096])
            h_fc1 = tf.nn.relu(tf.matmul(h_fc0, W_fc1) + b_fc1)

            W_fc2 = weight_variable([4096, 1000])
            b_fc2 = bias_variable([1000])
            h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)

            W_fc3 = weight_variable([1000, 8])
            b_fc3 = bias_variable([8])
            logits = tf.matmul(h_fc2, W_fc3) + b_fc3
            self.pred = tf.nn.softmax(logits)
            self.pred_loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=self.classify_labels)

        # Small MLP for domain prediction with adversarial loss
        with tf.variable_scope('domain_predictor'):
            # Flip the gradient when backpropagating through this operation
            feat = flip_gradient(self.feature, self.l)

            d_W_fc0 = weight_variable([6 * 6 * 256, 1000])
            d_b_fc0 = bias_variable([1000])
            d_h_fc0 = tf.nn.relu(tf.matmul(feat, d_W_fc0) + d_b_fc0)

            d_W_fc1 = weight_variable([1000, 2])
            d_b_fc1 = bias_variable([2])
            d_logits = tf.matmul(d_h_fc0, d_W_fc1) + d_b_fc1

            self.domain_pred = tf.nn.softmax(d_logits)
            self.domain_loss = tf.nn.softmax_cross_entropy_with_logits(logits=d_logits, labels=self.domain)

# Build the model graph
graph = tf.get_default_graph()
with graph.as_default():
    model = MNISTModel()
    print(source_test_labels[0])
    print(source_test_labels[0])
    print(target_train_labels[0])
    print(target_test_labels[0])
    learning_rate = tf.placeholder(tf.float32, [])

    pred_loss = tf.reduce_mean(model.pred_loss)
    # pred_loss_summary = tf.summary.scalar('pred_loss', pred_loss)
    domain_loss = tf.reduce_mean(model.domain_loss)
    total_loss = pred_loss + domain_loss
    # total_loss_summary = tf.summary.scalar('total_loss', total_loss)

    regular_train_op = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(pred_loss)
    dann_train_op = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(total_loss)

    # Evaluation
    correct_label_pred = tf.equal(tf.argmax(model.classify_labels, 1), tf.argmax(model.pred, 1))
    label_acc = tf.reduce_mean(tf.cast(correct_label_pred, tf.float32))
    correct_domain_pred = tf.equal(tf.argmax(model.domain, 1), tf.argmax(model.domain_pred, 1))
    domain_acc = tf.reduce_mean(tf.cast(correct_domain_pred, tf.float32))

def train_and_evaluate(training_mode, graph, model, num_steps=8600, verbose=False):
    """Helper to run the model with different training modes."""

    with tf.Session(graph=graph) as sess:
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

        domain_labels = np.vstack([np.tile([1., 0.], [batch_size // 2, 1]),
                                   np.tile([0., 1.], [batch_size // 2, 1])])

        # Training loop
        for i in range(num_steps):

            # Adaptation param and learning rate schedule as described in the paper
            p = float(i) / num_steps
            l = 2. / (1. + np.exp(-10. * p)) - 1
            # lr = 0.01 / (1. + 10 * p) ** 0.75
            lr = 0.000001
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

                if verbose and i % 50 == 0:

                    print('step {}, loss: {}  d_acc: {}  p_acc: {}  p: {}  l: {}  lr: {} duration: {}'.format(
                        i, batch_loss, d_acc, p_acc, p, l, lr, duration))
                    source_acc = sess.run(label_acc,
                                          feed_dict={model.X: source_test_images, model.y: source_test_labels,
                                                     model.train: False})

                    target_acc = sess.run(label_acc,
                                          feed_dict={model.X: target_test_images, model.y: target_test_labels,
                                                     model.train: False})
                    print("source_acc:{}\t target_acc:{}".format(source_acc, target_acc))


            elif training_mode == 'source':
                # X, y = get_data(source_train_filename, batch_size)
                X, y = next(gen_source_only_batch)
                start_time = time.time()
                _, batch_loss, p_acc = sess.run([regular_train_op, pred_loss, label_acc],
                                         feed_dict={model.X: X, model.y: y, model.train: False,
                                                    model.l: l, learning_rate: lr})
                duration = time.time() - start_time
                if verbose and i % 1500 == 0:
                    print('step {}, loss: {}  p_acc: {}  p: {}  l: {}  lr: {} duration:{}'.format(
                        i, batch_loss, p_acc, p, l, lr, duration))
                    source_acc = sess.run(label_acc,
                                          feed_dict={model.X: source_test_images, model.y: source_test_labels,
                                                     model.train: False})

                    target_acc = sess.run(label_acc,
                                          feed_dict={model.X: target_test_images, model.y: target_test_labels,
                                                     model.train: False})
                    print("source_acc:{}\t target_acc:{}".format(source_acc, target_acc))

            elif training_mode == 'target':
                # X, y = get_data(target_train_filename, batch_size)
                X, y = next(gen_target_only_batch)
                _, batch_loss = sess.run([regular_train_op, pred_loss],
                                         feed_dict={model.X: X, model.y: y, model.train: False,
                                                    model.l: l, learning_rate: lr})

        # Compute final evaluation on test data
        source_acc = sess.run(label_acc,
                              feed_dict={model.X: source_test_images, model.y: source_test_labels,
                                         model.train: False})

        target_acc = sess.run(label_acc,
                              feed_dict={model.X: target_test_images, model.y: target_test_labels,
                                         model.train: False})

        test_domain_acc = sess.run(domain_acc,
                                   feed_dict={model.X: combined_test_imgs,
                                              model.domain: combined_test_domain, model.l: 1.0})

        test_emb = sess.run(model.feature, feed_dict={model.X: combined_test_imgs})

    return source_acc, target_acc, test_domain_acc, test_emb




if __name__ == "__main__":

    print('\nDomain adaptation training')
    source_acc, target_acc, d_acc, dann_emb = train_and_evaluate('dann', graph, model, verbose=True)
    print('Source accuracy:', source_acc)
    print('Target accuracy:', target_acc)
    print('Domain accuracy:', d_acc)

    print('\nSource only training')
    source_acc, target_acc, _, source_only_emb = train_and_evaluate('source', graph, model, verbose=True)
    print('Source accuracy:', source_acc)
    print('Target accuracy:', target_acc)

