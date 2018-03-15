__author__ = 'qcp'
# -*- coding:utf-8 -*-
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf

reader = tf.TFRecordReader()
# 创建一个队列来维护输入文件列表
filename_queue = tf.train.string_input_producer(['path/to/output.tfrecords'])

_, serialized_example = reader.read(filename_queue)

# 解析读入的一个样例。
# 如果需要解析多个样例，可以用parse_example函数
features = tf.parse_single_example(
    serialized_example,
    features={
        # TensorFlow提供两种不同的属性解析方法。
        #一种方法是tf.FixedLenFeature，解析结果为一个Tensor。
        #另一种方法是tf.VarLenFeature，解析结果为SparseTensor，用于处理稀疏数据。
        #这里的解析数据的格式需要和写入数据的格式一致。
        'image_raw': tf.FixedLenFeature([], tf.string),
        'pixels': tf.FixedLenFeature([], tf.int64),
        'label': tf.FixedLenFeature([], tf.int64)})

# tf.decode_raw可以将字符串解析成图像对应的像素数组
images = tf.decode_raw(features['image_raw'], tf.uint8)
labels = tf.cast(features['label'], tf.int32)
pixels = tf.cast(features['pixels'], tf.int32)

sess = tf.Session()
#启动多线程处理输入数据
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

#每次运行可以读取TFRecord文件中的一个样例
#所有样例都读完后，再次样例中程序会重新从头读取
for i in range(10):
    image, label, pixel = sess.run([images, labels, pixels])