# -*- coding: utf-8 -*-
# created by kinredon

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from transform2tfrecord import get_data
import time

from flip_gradient import flip_gradient
from utils import *

log_dir = 'summary_vgg'


def print_activations(t):
    print(t.op.name, ' ', t.get_shape().as_list())


def conv_op(input_op, name, kh, kw, n_out, dh, dw):
    n_in = input_op.get_shape()[-1].value
    with tf.name_scope(name) as scope:
        kernel = tf.get_variable(scope+"w",
                                 shape=[kh, kw, n_in, n_out],
                                 dtype=tf.float32,
                                 initializer=tf.contrib.layers.xavier_initializer_conv2d())
        conv = tf.nn.conv2d(input_op, kernel, (1, dh, dw, 1), padding='SAME')
        bias_init_val = tf.constant(0.0, shape=[n_out], dtype=tf.float32)
        biases = tf.Variable(bias_init_val, trainable=True, name='b')
        z = tf.nn.bias_add(conv, biases)
        activation = tf.nn.relu(z, name=scope)
        return activation


def fc_op(input_op, name, n_out):
    n_in = input_op.get_shape()[-1].value

    with tf.name_scope(name) as scope:
        kernel = tf.get_variable(scope+"w",
                                 shape=[n_in, n_out],
                                 dtype=tf.float32,
                                 initializer=tf.contrib.layers.xavier_initializer())
        biases = tf.Variable(tf.constant(0.1, shape=[n_out], dtype=tf.float32), name='b')
        activation = tf.nn.relu_layer(input_op, kernel, biases, name=scope)
        return activation


def mpool_op(input_op, name, kh, kw, dh, dw):
    return tf.nn.max_pool(input_op,
                          ksize=[1, kh, kw, 1],
                          strides=[1, dh, dw, 1],
                          padding='SAME',
                          name=name)

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

# test data
combined_test_imgs = np.vstack([source_test_images, target_test_images])
combined_test_labels = np.vstack([source_test_labels, target_test_labels])
combined_test_domain = np.vstack([np.tile([1., 0.], [522, 1]),
        np.tile([0., 1.], [372, 1])])

batch_size = 64


class Model(object):
    """Simple MNIST domain adaptation model."""

    def __init__(self, keep_prob):
        self._build_model(keep_prob)

    def _build_model(self, keep_prob):
        self.X = tf.placeholder(tf.float32, [None, 224, 224, 3])  # 输入
        self.y = tf.placeholder(tf.float32, [None, 8])  # 标签
        self.domain = tf.placeholder(tf.float32, [None, 2])  # 域标签
        self.l = tf.placeholder(tf.float32, [])  # 反转
        self.train = tf.placeholder(tf.bool, [])  #


        # CNN model for feature extraction
        with tf.variable_scope('feature_extractor'):

            # block 1 -- outputs 112x112x64
            conv1_1 = conv_op(self.X, name="conv1_1", kh=3, kw=3, n_out=64, dh=1, dw=1)
            conv1_2 = conv_op(conv1_1, name="conv1_2", kh=3, kw=3, n_out=64, dh=1, dw=1)
            pool1 = mpool_op(conv1_2, name="pool1", kh=2, kw=2, dw=2, dh=2)

            # block 2 -- outputs 56x56x128
            conv2_1 = conv_op(pool1, name="conv2_1", kh=3, kw=3, n_out=128, dh=1, dw=1)
            conv2_2 = conv_op(conv2_1, name="conv2_2", kh=3, kw=3, n_out=128, dh=1, dw=1)
            pool2 = mpool_op(conv2_2, name="pool2", kh=2, kw=2, dh=2, dw=2)

            # # block 3 -- outputs 28x28x256
            conv3_1 = conv_op(pool2, name="conv3_1", kh=3, kw=3, n_out=256, dh=1, dw=1)
            conv3_2 = conv_op(conv3_1, name="conv3_2", kh=3, kw=3, n_out=256, dh=1, dw=1)
            conv3_3 = conv_op(conv3_2, name="conv3_3", kh=3, kw=3, n_out=256, dh=1, dw=1)
            pool3 = mpool_op(conv3_3, name="pool3", kh=2, kw=2, dh=2, dw=2)

            # block 4 -- outputs 14x14x512
            conv4_1 = conv_op(pool3, name="conv4_1", kh=3, kw=3, n_out=512, dh=1, dw=1)
            conv4_2 = conv_op(conv4_1, name="conv4_2", kh=3, kw=3, n_out=512, dh=1, dw=1)
            conv4_3 = conv_op(conv4_2, name="conv4_3", kh=3, kw=3, n_out=512, dh=1, dw=1)
            pool4 = mpool_op(conv4_3, name="pool4", kh=2, kw=2, dh=2, dw=2)

            # block 5 -- outputs 7x7x512
            conv5_1 = conv_op(pool4, name="conv5_1", kh=3, kw=3, n_out=512, dh=1, dw=1)
            conv5_2 = conv_op(conv5_1, name="conv5_2", kh=3, kw=3, n_out=512, dh=1, dw=1)
            conv5_3 = conv_op(conv5_2, name="conv5_3", kh=3, kw=3, n_out=512, dh=1, dw=1)
            pool5 = mpool_op(conv5_3, name="pool5", kh=2, kw=2, dw=2, dh=2)

            # flatten
            shp = pool5.get_shape()
            # print(shp)
            flattened_shape = shp[1].value * shp[2].value * shp[3].value
            # print(flattened_shape)
            resh1 = tf.reshape(pool5, [-1, flattened_shape], name="resh1")

            # The domain-invariant feature
            self.feature = resh1

        with tf.variable_scope('label_predictor'):
            # Switches to route target examples (second half of batch) differently
            # depending on train or test mode.
            all_features = lambda: self.feature
            source_features = lambda: tf.slice(self.feature, [0, 0], [batch_size // 2, -1])  # back_size的一半
            classify_feats = tf.cond(self.train, source_features, all_features)  # 如果train=true，返回source_features

            all_labels = lambda: self.y
            source_labels = lambda: tf.slice(self.y, [0, 0], [batch_size // 2, -1])
            self.classify_labels = tf.cond(self.train, source_labels, all_labels)  # 与前面同理

            fc6 = fc_op(classify_feats, name="fc6", n_out=4096)
            fc6_drop = tf.nn.dropout(fc6, keep_prob, name="fc6_drop")

            fc7 = fc_op(fc6_drop, name="fc7", n_out=4096)
            fc7_drop = tf.nn.dropout(fc7, keep_prob, name="fc7_drop")

            fc8 = fc_op(fc7_drop, name="fc8", n_out=1000)
            fc9 = fc_op(fc8, name="fc9", n_out=100)
            fc10 = fc_op(fc9, name="fc10", n_out=8)

            self.pred = tf.nn.softmax(fc10)
            self.pred_loss = tf.nn.softmax_cross_entropy_with_logits(logits=fc10, labels=self.classify_labels)

        # Small MLP for domain prediction with adversarial loss
        with tf.variable_scope('domain_predictor'):
            # Flip the gradient when backpropagating through this operation
            feat = flip_gradient(self.feature, self.l)

            d_W_fc0 = weight_variable([shp[1].value * shp[2].value * shp[3].value, 1000])
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

    # Evaluation
    correct_label_pred = tf.equal(tf.argmax(model.classify_labels, 1), tf.argmax(model.pred, 1))
    label_acc = tf.reduce_mean(tf.cast(correct_label_pred, tf.float32))
    tf.summary.scalar('label_accuracy', label_acc)

    correct_domain_pred = tf.equal(tf.argmax(model.domain, 1), tf.argmax(model.domain_pred, 1))
    domain_acc = tf.reduce_mean(tf.cast(correct_domain_pred, tf.float32))

    tf.summary.scalar('domain_accuracy', domain_acc)
    merged = tf.summary.merge_all()


def train_and_evaluate(training_mode, graph, model, num_steps=3000, verbose=False):
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
                    source_acc = sess.run(label_acc,
                                          feed_dict={model.X: source_test_images_part, model.y: source_test_labels_part,
                                                     model.train: False})

                    target_acc = sess.run(label_acc,
                                          feed_dict={model.X: target_test_images_part, model.y: target_test_labels_part,
                                                     model.train: False})
                    print("source_acc:{}\t target_acc:{}".format(source_acc, target_acc))


            elif training_mode == 'source':
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

    train_writer.close()
    test_source_writer.close()
    test_target_writer.close()
    return source_acc, target_acc, test_domain_acc, test_emb


if __name__ == "__main__":

    print('\nDomain adaptation training')
    source_acc, target_acc, d_acc, dann_emb = train_and_evaluate('dann', graph, model, verbose=True)
    print('Source accuracy:', source_acc)
    print('Target accuracy:', target_acc)
    print('Domain accuracy:', d_acc)

    # print('\nSource only training')
    # source_acc, target_acc, _, source_only_emb = train_and_evaluate('source', graph, model, verbose=True)
    # print('Source accuracy:', source_acc)
    # print('Target accuracy:', target_acc)

