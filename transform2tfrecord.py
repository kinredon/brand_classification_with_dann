# -*- coding: utf-8 -*-
# created by kinredon
# Function: image convert to tfrecords

import tensorflow as tf
import numpy as np
import cv2
import os
import os.path
from PIL import Image

# 参数
IMAGE_WIDTH = 224
IMAGE_HEIGHT= 224
IMAGE_CHANEL = 3
BATCH_SIZE = 32


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def convert_to_one_hot(y, C):
    return np.eye(C)[y.reshape(-1)]


def id_to_label():
    lines = []
    with open('labels.txt', 'r') as f:
        lines = f.readlines()
    label_to_name = {}
    for i, line in enumerate(lines):
        label_to_name[i] = line
    return label_to_name

def load_file(examples_list_file):
    '''
    加载文件内容
    :param examples_list_file: 存储图片地址和类别信息的文件名
    :return: examples, labels, example_num
    '''
    lines = []
    with open(examples_list_file, 'r') as f:
        lines = f.readlines()
    examples = []
    labels = []
    for line in lines:
        example, label = line.split(' ')
        examples.append(example)
        labels.append(int(label))
    return np.asarray(examples), np.asarray(labels), len(lines)


def extract_image(filename, resize_height, resize_width):
    image = cv2.imread(filename)
    image = cv2.resize(image, (resize_height, resize_width))
    b, g, r = cv2.split(image)
    rgb_image = cv2.merge([r, g, b])
    return rgb_image


def transform2tfrecord(file, name, output_directory):
    '''
    将图片转化为tfrecord的形式。tfrecord包含样本以及其类别信息
    :param file: 存储图片地址和类别信息的文件名
    :param name: tfrecord文件存储名
    :param output_directory: tfrecord文件输出目录
    :param resize_height: 调整后图片的高度
    :param resize_width: 调整后图片的宽度
    :return: null
    '''
    if not os.path.exists(output_directory) or os.path.isfile(output_directory):
        os.makedirs(output_directory)
    _examples, _labels, examples_num = load_file(file)
    filename = output_directory + "/" + name + '.tfrecords'
    writer = tf.python_io.TFRecordWriter(filename)
    for i, [example, label] in enumerate(zip(_examples, _labels)):
        print('No.%d' % (i))
        image = extract_image(example, IMAGE_WIDTH, IMAGE_HEIGHT)
        print('shape: %d, %d, %d, label: %d' % (image.shape[0], image.shape[1], image.shape[2], label))
        image_raw = image.tostring()
        example = tf.train.Example(features=tf.train.Features(feature={
            'image_raw': _bytes_feature(image_raw),
            'height': _int64_feature(image.shape[0]),
            'width': _int64_feature(image.shape[1]),
            'depth': _int64_feature(image.shape[2]),
            'label': _int64_feature(label)
        }))
        writer.write(example.SerializeToString())
    writer.close()


def disp_tfrecords(tfrecord_list_file):
    filename_queue = tf.train.string_input_producer([tfrecord_list_file])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
            'image_raw': tf.FixedLenFeature([], tf.string),
            'height': tf.FixedLenFeature([], tf.int64),
            'width': tf.FixedLenFeature([], tf.int64),
            'depth': tf.FixedLenFeature([], tf.int64),
            'label': tf.FixedLenFeature([], tf.int64)
        }
    )
    image = tf.decode_raw(features['image_raw'], tf.uint8)
    # print(repr(image))
    height = features['height']
    width = features['width']
    depth = features['depth']
    label = tf.cast(features['label'], tf.int32)
    init_op = tf.initialize_all_variables()
    resultImg = []
    resultLabel = []

    with tf.Session() as sess:
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        for i in range(3):
            # image_eval = image.eval()
            # resultLabel.append(label.eval())
            # image_eval_reshape = image_eval.reshape([height.eval(), width.eval(), depth.eval()])
            # resultImg.append(image_eval_reshape)
            # pilimg = Image.fromarray(np.asarray(image_eval_reshape))
            # print(id_to_label()[label.eval()])
            # pilimg.show()
            print(label.eval())
        coord.request_stop()
        coord.join(threads)
        sess.close()
    return resultImg, resultLabel


def read_tfrecord(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
            'image_raw': tf.FixedLenFeature([], tf.string),
            'width': tf.FixedLenFeature([], tf.int64),
            'depth': tf.FixedLenFeature([], tf.int64),
            'label': tf.FixedLenFeature([], tf.int64)
        }
    )
    image = tf.decode_raw(features['image_raw'], tf.uint8)
    label = tf.cast(features['label'], tf.int32)
    # image
    image.set_shape([IMAGE_WIDTH * IMAGE_HEIGHT * IMAGE_CHANEL])
    image = tf.reshape(image, [IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANEL])
    # normalize
    # image = tf.cast(image, tf.float32) * (1. / 255) - 0.5
    image = tf.cast(image, tf.float32)
    pixel_mean = tf.reduce_mean(image, (0, 1, 2))
    image = (tf.cast(image, tf.float32) - pixel_mean) / 255.
    return image, label


# 用于获取一个batch_size的图像和label
def inputs(filename_queuetemp, batch_size = 32):

    with tf.name_scope('input') as scope:
        filename_queue = tf.train.string_input_producer([filename_queuetemp])
    image,label = read_tfrecord(filename_queue)
    # 随机获得batch_size大小的图像和label
    # images,labels = tf.train.shuffle_batch([image, label],
    #     batch_size= batch_size,
    #     num_threads=3,
    #     capacity=10 + 3 * batch_size,
    #     min_after_dequeue=10
    # )
    images, labels = tf.train.batch([image, label], batch_size=batch_size)
    return images, labels


def get_data(filename_queuetemp, batch_size = 32):
    # 生成测试数据
    images, labels = inputs(filename_queuetemp, batch_size)  # 读取函数

    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())
    with tf.Session() as sess:
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        sample, l = sess.run([images, labels])
        coord.request_stop()
        coord.join(threads)
    # 分为八个类
    l = convert_to_one_hot(l, 8)

    return sample, l

# def test():
#     transform2tfrecord(train_file, name, output_directory, resize_height, resize_width)  # 转化函数
#     img, label = disp_tfrecords(output_directory + '/' + name + '.tfrecords')  # 显示函数
#     img, label = read_tfrecord(output_directory + '/' + name + '.tfrecords')  # 读取函数
#     print(label)


if __name__ == '__main__':
    # test()
    # 参数设置
    train_file = 'source_list_train.txt'                    # 训练图片
    # name = 'source_train'                                   # 生成train.tfrecords
    # output_directory = './tfrecords_inception'
    # # source_test_filename = './tfrecords/source_test.tfrecords'
    # # source_test_images, source_test_labels = get_data(source_test_filename, batch_size=522)
    # # print(source_test_labels[0])
    # # disp_tfrecords(tfrecord_list_file=output_directory + "/" + name + ".tfrecords")
    # # 生成源训练数据
    # transform2tfrecord(train_file, name, output_directory)  # 转化函数
    # # print(label)
    #
    # # 生成源测试数据
    # train_file = 'source_list_test.txt'                     # 训练图片
    # name = 'source_test'                                    # 生成train.tfrecords
    # transform2tfrecord(train_file, name, output_directory)  # 转化函数
    #
    # # 生成目标测试数据
    # train_file = 'target_list_train.txt'                     # 训练图片
    # name = 'target_train'                                    # 生成train.tfrecords
    # transform2tfrecord(train_file, name, output_directory)  # 转化函数
    #
    # # 生成目标测试数据
    # train_file = 'target_list_test.txt'                     # 训练图片
    # name = 'target_test'                                    # 生成train.tfrecords
    # transform2tfrecord(train_file, name, output_directory)  # 转化函数

    # disp_tfrecords('./tfrecords/source_test.tfrecords')
    # # 生成源测试数据
    # train_file = 'target' \
    #              '_list_test.txt'                     # 训练图片
    # name = 'target_test_hhh'                                    # 生成train.tfrecords
    # transform2tfrecord(train_file, name, output_directory)  # 转化函数


