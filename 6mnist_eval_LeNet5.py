__author__ = 'qcp'
# -*- coding:utf-8 -*-
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import mnist_inference_LeNet5
import mnist_train_LeNet5
import numpy as np
import matplotlib.pyplot as plt
# 每10秒加载一次最新模型，并在测试数据上测试其正确率
EVAL_INTERVAL_SECS = 100


def evaluate(mnist):
    with tf.Graph().as_default() as g:
        x = tf.placeholder(dtype=tf.float32, shape=(mnist.validation.num_examples,
                                                    mnist_inference_LeNet5.IMAGE_SIZE,
                                                    mnist_inference_LeNet5.IMAGE_SIZE,
                                                    mnist_inference_LeNet5.NUM_CHANNELS), name='x-input')
        y_ = tf.placeholder(dtype=tf.float32, shape=(None, mnist_inference_LeNet5.OUTPUT_NODE), name='y-input')

        x1 = np.reshape(mnist.validation.images, newshape=(mnist.validation.num_examples,
                                                           mnist_inference_LeNet5.IMAGE_SIZE,
                                                           mnist_inference_LeNet5.IMAGE_SIZE,
                                                           mnist_inference_LeNet5.NUM_CHANNELS))

        debug = False
        if debug:
            print(np.shape(mnist.validation.images))
            print(np.shape(x1))
            # print(x1[7])
            fig = plt.figure()
            ax = fig.add_subplot(111)
            A = x1[1].reshape(28, 28)
            ax.imshow(A)
            plt.show()

        validate_feed = {x: x1,
                         y_: mnist.validation.labels}

        y = mnist_inference_LeNet5.inference(x, False, None)  # 测试时不关心正则化损失，所以设置为None
        # print(np.shape(y))
        #with tf.Session() as sess:
        #   print(sess.run(y))
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # 用变量重命名的方式来加载模型，这样在inference中就不需要调用滑动平均函数来获取平均值了。
        # 这样完全可以共用mnist_inference中定义的前向传播过程。
        variable_averages = tf.train.ExponentialMovingAverage(mnist_train_LeNet5.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        while True:
            with tf.Session() as sess:
                # tf.train.get_checkpoint_state函数会通过checkpoint文件自动找到目录中的最新模型文件名
                ckpt = tf.train.get_checkpoint_state(mnist_train_LeNet5.MODEL_SAVE_PATH)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                    accuracy_score = sess.run(accuracy, feed_dict=validate_feed)
                    print('After %s training steps, validation accuracy is %g' % (global_step, accuracy_score))
                else:
                    print('No checkpoint file found.')
                    return
            time.sleep(EVAL_INTERVAL_SECS)


def main(argv=None):
    mnist = input_data.read_data_sets('path/to/MNIST_data', one_hot=True)
    evaluate(mnist)


if __name__ == '__main__':
    tf.app.run()