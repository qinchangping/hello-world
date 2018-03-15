__author__ = 'qcp'
# -*- coding:utf-8 -*-
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import mnist_inference_LeNet5
import numpy as np

BATCH_SIZE = 100  # 一个训练batch中的训练数据个数。数字越小，越接近随机梯度下降，越大越接近梯度下降
LEARNING_RATE_BASE = 0.01  # 学习率
LEARNING_RATE_DECAY = 0.99  # 学习率的衰减率
REGULARIZATION_RATE = 0.0001  # 描述模型复杂度的正则化项在损失函数中的系数
TRAINING_STEPS = 30000  # 训练轮数1w
MOVING_AVERAGE_DECAY = 0.99  # 滑动平均衰减率

MODEL_SAVE_PATH = 'path/to/model/'
MODEL_NAME = 'model.ckpt'


def train(mnist):
    # 输入层，28*28*1
    x = tf.placeholder(dtype=tf.float32, shape=(BATCH_SIZE,
                                                mnist_inference_LeNet5.IMAGE_SIZE,
                                                mnist_inference_LeNet5.IMAGE_SIZE,
                                                mnist_inference_LeNet5.NUM_CHANNELS), name='x-input')
    # 10个输出节点
    y_ = tf.placeholder(dtype=tf.float32, shape=(None, mnist_inference_LeNet5.OUTPUT_NODE), name='y-input')

    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    y = mnist_inference_LeNet5.inference(x, train=True, regularizer=regularizer)
    global_step = tf.Variable(0, trainable=False)

    # 给定滑动平均衰减率和训练轮数变量，初始化滑动平均类
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    # 在所有代表神经网络参数的变量上使用滑动平均。
    variable_averages_op = variable_averages.apply(tf.trainable_variables())

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    # 计算在当前batch中所有样例的交叉熵平均值
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
    # 设置指数衰减的学习率
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        mnist.train.num_examples / BATCH_SIZE,
        LEARNING_RATE_DECAY)
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    with tf.control_dependencies([train_step, variable_averages_op]):
        train_op = tf.no_op(name='train')

    # 初始化TensorFlow持久化类
    saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        for i in range(TRAINING_STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            reshaped_xs = np.reshape(xs, newshape=(BATCH_SIZE,
                                                mnist_inference_LeNet5.IMAGE_SIZE,
                                                mnist_inference_LeNet5.IMAGE_SIZE,
                                                mnist_inference_LeNet5.NUM_CHANNELS))
            a, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: reshaped_xs, y_: ys})
            if i % 1000 == 0:
                print('After %d training steps, loss on training batch is %g' % (step, loss_value))
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)


def main(argv=None):
    mnist = input_data.read_data_sets('path/to/MNIST_data', one_hot=True)
    train(mnist)


if __name__ == '__main__':
    tf.app.run()