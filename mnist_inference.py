__author__ = 'qcp'
# -*- coding:utf-8 -*-
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

# 神经网络相关参数
INPUT_NODE = 784  # 输入层节点数，对于MNIST数据集，=图片像素
OUTPUT_NODE = 10  # 输出层节点数，=类别数。区分0~9，共10类
# 神经网络相关参数
LAYER1_NODE = 500  # 隐藏层节点数。1层，500个节点


def get_weight_variable(shape, regularizer):
	weights = tf.get_variable('weights', shape,
							  initializer=tf.truncated_normal_initializer(stddev=0.1))
	if regularizer != None:
		tf.add_to_collection('losses', regularizer(weights))
	return weights


def inference(input_tensor, regularizer):
	#在这里通过tf.get_variable或tf.Variable没有本质区别，因为在训练或是测试中没有在同一个程序中多次调用这个函数。
	#如果在同一个程序中多次调用，在第一次调用之后需要将reuse设置为True。
	with tf.variable_scope('layer1'):
		weights = get_weight_variable([INPUT_NODE, LAYER1_NODE], regularizer)
		biases = tf.get_variable('biases', [LAYER1_NODE], initializer=tf.constant_initializer(0.0))
		layer1 = tf.nn.relu(tf.matmul(input_tensor, weights) + biases)
	with tf.variable_scope('layer2'):
		weights = get_weight_variable([LAYER1_NODE, OUTPUT_NODE], regularizer)
		biases = tf.get_variable('biases', [OUTPUT_NODE], initializer=tf.constant_initializer(0.0))
		layer2 = tf.matmul(layer1, weights) + biases
	return layer2