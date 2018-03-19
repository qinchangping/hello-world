__author__ = 'qcp'
# -*- coding:utf-8 -*-
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

# 数据集相关参数
INPUT_NODE = 784  # 输入层节点数，对于MNIST数据集，=图片像素
OUTPUT_NODE = 10  # 输出层节点数，=类别数。区分0~9，共10类
# 神经网络相关参数
LAYER1_NODE = 500  # 隐藏层节点数。1层，500个节点
BATCH_SIZE = 100  # 一个训练batch中的训练数据个数。数字越小，越接近随机梯度下降，越大越接近梯度下降
LEARNING_RATE_BASE = 0.001  # 学习率
LEARNING_RATE_DECAY = 0.99  # 学习率的衰减率
REGULARIZATION_RATE = 0.0001  # 描述模型复杂度的正则化项在损失函数中的系数
TRAINING_STEPS = 2000  # 训练轮数
MOVING_AVERAGE_DECAY = 0.99  # 滑动平均衰减率

SUMMARY_DIR = 'path/to/log'

'''
tf.merge_all_summaries() 改为：summary_op = tf.summary.merge_all()
tf.train.SummaryWriter 改为：tf.summary.FileWriter
tf.scalar_summary 改为：tf.summary.scalar
histogram_summary 改为：tf.summary.histogram
'''


def variable_summaries(var, name):
    with tf.name_scope('summaries'):
        tf.summary.histogram(name, var)
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean/' + name, mean)
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev/' + name, stddev)


def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            weights = tf.Variable(tf.truncated_normal([input_dim, output_dim], stddev=0.1))
            variable_summaries(weights, layer_name + '/weights')
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.constant(0.1, shape=[output_dim]))
            variable_summaries(biases, layer_name + '/biases')
        with tf.name_scope('Wx_plus_b'):
            preactivate = tf.matmul(input_tensor, weights) + biases
            tf.summary.histogram(layer_name + '/preactivations', preactivate)
        activations = act(preactivate, name='activation')
        tf.summary.histogram(layer_name + '/activations', activations)
        return activations


def main(_):
    mnist = input_data.read_data_sets('path/to/MNIST_data', one_hot=True)
    with tf.name_scope('input'):
        x = tf.placeholder(dtype=tf.float32, shape=(None, INPUT_NODE), name='x-input')
        y_ = tf.placeholder(dtype=tf.float32, shape=(None, OUTPUT_NODE), name='y-input')
    with tf.name_scope('input_reshape'):
        image_shaped_input = tf.reshape(x, [-1, 28, 28, 1])
        tf.summary.image('input', image_shaped_input, 10)

    hidden1 = nn_layer(x, INPUT_NODE, LAYER1_NODE, 'layer1')
    y = nn_layer(hidden1, LAYER1_NODE, OUTPUT_NODE, 'layer2', act=tf.identity)

    with tf.name_scope('cross_entropy'):
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
        tf.summary.scalar('corss_entropy', cross_entropy)

    with tf.name_scope('train'):
        train_step = tf.train.AdamOptimizer(LEARNING_RATE_BASE).minimize(cross_entropy)

    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', accuracy)

    merged = tf.summary.merge_all()

    with tf.Session() as sess:
        summary_writer = tf.summary.FileWriter(SUMMARY_DIR, sess.graph)
        tf.global_variables_initializer().run()
        for i in range(TRAINING_STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            summary, _ = sess.run([merged, train_step], feed_dict={x: xs, y_: ys})
            summary_writer.add_summary(summary, i)
    summary_writer.close()
    os.system('tensorboard --logdir=path/to/log')


if __name__ == '__main__':
    tf.app.run()


