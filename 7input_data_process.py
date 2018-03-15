__author__ = 'qcp'
# -*- coding:utf-8 -*-
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np
import threading

# 给定一张图像，随机调整图像的色彩。因为调整疗毒、对比度、饱和度和色相的顺序会影响最后得到的结果，
# 所以可以定义多种不同的顺序。具体使用哪一种顺序可以在训练数据预处理时随机选择一种，
# 这样可以进一步降低无关因素对模型的影响。
#
def distort_color(image, color_ordering=0):
    if color_ordering == 0:
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
    elif color_ordering == 1:
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
    elif color_ordering == 2:
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
    # 其余略
    return tf.clip_by_value(image, 0.1, 1.0)


# 给定一张解码后的图像、目标图像的尺寸以及图像上的标注框，此函数可以对给出的图像进行预处理。
# 这个函数的输入图像是图像识别问题中原始的训练图像，
# 输出是神经网络模型的输入层。
# 只处理模型的训练数据，对于预测数据，一般不需要使用随机变换步骤。
def preprocess_for_train(image, height, width, bbox):
    # 如果没有提供标注框，则认为整个图像就是需要关注的部分。
    if bbox is None:
        bbox = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4])

    # 转换图像张量的类型
    if image.dtype != tf.float32:
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)

    # 随机截取图像，减小需要关注的物体大小对图像识别算法的影响
    bbox_begin, bbox_size, _ = tf.image.sample_distorted_bounding_box(
        tf.shape(image), bounding_boxes=bbox, min_object_covered=0.1)
    distorted_image = tf.slice(image, bbox_begin, bbox_size)
    # 将随机截取的图像调整为神经网络输入层的大小。大小调整的算法是随机选择的。
    distorted_image = tf.image.resize_images(
        distorted_image, [height, width], method=np.random.randint(4))
    # method也可以用这种方法指定：tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    # size是一个参数 [,1]
    #随机左右翻转图像
    distorted_image = tf.image.random_flip_left_right(distorted_image)
    #使用随机顺序调整图像色彩
    distorted_image = distort_color(distorted_image, np.random.randint(2))
    return distorted_image


files = tf.train.match_filenames_once('path/to/output*')
filename_queue = tf.train.string_input_producer(files, shuffle=False)

_, serialized_example = reader.read(filename_queue)

# 解析读入的一个样例。
# 如果需要解析多个样例，可以用parse_example函数
features = tf.parse_single_example(
    serialized_example,
    features={
        'image': tf.FixedLenFeature([], tf.string),
        'label': tf.FixedLenFeature([], tf.int64),
        'height': tf.FixedLenFeature([], tf.int64),
        'width': tf.FixedLenFeature([], tf.int64),
        'channels': tf.FixedLenFeature([], tf.int64)})
image = features['image']
label = features['label']
height = features['height']
width = features['width']
channels = features['channels']

# 从原始图像数据解析出像素矩阵，并根据图像尺寸还原图像
decoded_image = tf.decode_raw(image, tf.uint8)
decoded_image.set_shape([height, width, channels])
# 定义神经网络输入层图片的大小
image_size = 299
distorted_image = preprocess_for_train(
    decoded_image, image_size, image_size, None)

#将处理后的图像和标签通过 整理成神经网络训练时需要的batch
min_after_dequeue = 10000
batch_size = 100
capacity = min_after_dequeue + 3 * batch_size
image_batch, label_batch = tf.train.shuffle_batch(
    [distorted_image, label], batch_size=batch_size,
    capacity=capacity, min_after_dequeue=min_after_dequeue)

logit=inference(image_batch)
loss=calc_loss(logit,label_batch)
train_step=tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    tf.local_variables_initializer().run()
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    #训练神经网络入。
    for i in range(TRAINING_ROUNDS):
        sess.run(train_step)
    coord.request_stop()
    coord.join(threads)
