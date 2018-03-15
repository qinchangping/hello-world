__author__ = 'qcp'
# -*- coding:utf-8 -*-
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf

# 卷及神经网络前向传播过程
# 过滤器权重变量和偏置项变量
# 参数变量是一个四维矩阵，前两个表示过滤器尺寸，第三个表示当前层的深度，第四个表示过滤器深度。
filter_weight = tf.get_variable(
    'weight', [5, 5, 3, 16], initializer=tf.truncated_normal_initializer(stddev=0.1))

#过滤器深度个不同的偏置项
biases = tf.get_variable(
    'biases', [16], initializer=tf.truncated_normal_initializer(stddev=0.1))

#tf.nn.conv2d函数，实现卷积层前向传播
#第一个参数为当前层的节点矩阵。注意是四维矩阵，后三个维度对应一个节点矩阵，第一个维度对应一个输入batch。
#第二个参数提供卷积层的权重，
# 第三个参数为不同维度上的步长。虽然是个长度为4的数组，但第一个和第四个维度要求一定是1，
# 这是因为卷积层的步长只对矩阵的长和宽有效。
#第四个参数是填充（padding）方法，TensorFlow提供SAME和VALID两种选择，
# 其中SAME表示添加全0填充，VALID表示不添加。
conv = tf.nn.conv2d(input, filter_weight, strides=[1, 1, 1, 1], padding='SAME')

#tf.nn.bias_add函数方便地给每一个节点加上偏置项。
#注意这里不能直接使用加法，因为矩阵上不同位置上的节点都需要加上同样的偏置项。
#下一层社交网络大小为2×2矩阵，而偏置项是一个数，不能直接加
bias = tf.nn.bias_add(conv, biases)

#将计算结果通过relu激活函数完成去线性化
actived_conv = tf.nn.relu(bias)

#最大池化层的前向传播过程
#ksize过滤器尺寸，第一个和最后一个必须是1
pool = tf.nn.max_pool(actived_conv, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

#平均池化层函数是tf.nn.avg_pool()

x = tf.placeholder(dtype=tf.float32, shape=(BATCH_SIZE,mnist_inference_LeNet5.IMAGE_SIZE,mnist_inference_LeNet5.IMAGE_SIZE,mnist_inference_LeNet5.NUM_CHANNELS), name='x-input')
y_ = tf.placeholder(dtype=tf.float32, shape=(None, mnist_inference_LeNet5.OUTPUT_NODE), name='y-input')