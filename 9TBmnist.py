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
LEARNING_RATE_BASE = 0.8  # 学习率
LEARNING_RATE_DECAY = 0.99  # 学习率的衰减率
REGULARIZATION_RATE = 0.0001  # 描述模型复杂度的正则化项在损失函数中的系数
TRAINING_STEPS = 2000  # 训练轮数
MOVING_AVERAGE_DECAY = 0.99  # 滑动平均衰减率


def inference(input_tensor, avg_class, weights1, biases1, weights2, biases2):
    '''辅助函数，给定社交网络的输入和所有参数，计算神经网络的前向传播结果。
    使用ReLu激活函数的三层全连接神经网络。
    也支持传入用于计算参数平均值的类，方便在测试时使用滑动平均模型。'''
    if avg_class == None:
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights1) + biases1)
        # 计算输出层前向传播结果。
        # 因为计算损失函数时会一并计算softmax函数，所以这里不需要加入激活函数。
        # 而且不加入softmax不会影响预测结果，因为预测时使用的是不同类别对应节点输出值的相对大小，
        # 有没有softmax层对最后分类结果的计算没有影响。
        # 于是在计算整个神经网络的前向传播时可以不加入最后的softmax层。
        return tf.matmul(layer1, weights2) + biases2
    else:
        # 先使用avg_class.average函数计算得出变量的滑动平均值，再计算前向传播
        layer1 = tf.nn.relu(tf.matmul(input_tensor, avg_class.average(weights1)) + avg_class.average(biases1))
        return tf.matmul(layer1, avg_class.average(weights2)) + avg_class.average(biases2)


# 训练模型的过程
def train(mnist):
    with tf.name_scope('input'):
        x = tf.placeholder(dtype=tf.float32, shape=(None, INPUT_NODE), name='x-input')
        y_ = tf.placeholder(dtype=tf.float32, shape=(None, OUTPUT_NODE), name='y-input')

    with tf.name_scope('hidden_layer'):
        # 生成隐藏层的参数
        weights1 = tf.Variable(tf.truncated_normal([INPUT_NODE, LAYER1_NODE], stddev=0.1), name='weights1')
        biases1 = tf.Variable(tf.constant(0.1, shape=[LAYER1_NODE]), name='bias1')

        # 生成输出层的参数
        weights2 = tf.Variable(tf.truncated_normal([LAYER1_NODE, OUTPUT_NODE], stddev=0.1), name='weights2')
        biases2 = tf.Variable(tf.constant(0.1, shape=[OUTPUT_NODE]), name='bias2')

    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    y = inference(x, None, weights1, biases1, weights2, biases2)
    global_step = tf.Variable(0, trainable=False)

    with tf.name_scope('moving_averages'):
        variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
        variable_averages_op = variable_averages.apply(tf.trainable_variables())

    average_y = inference(x, variable_averages, weights1, biases1, weights2, biases2)

    with tf.name_scope('loss_function'):
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
        cross_entropy_mean = tf.reduce_mean(cross_entropy)
        regularization = regularizer(weights1) + regularizer(weights2)
        loss = cross_entropy_mean + regularization

    with tf.name_scope('train_step'):
        learning_rate = tf.train.exponential_decay(
            LEARNING_RATE_BASE,
            global_step,
            mnist.train.num_examples / BATCH_SIZE,
            LEARNING_RATE_DECAY)

        train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
        train_op = tf.group(train_step, variable_averages_op)

    # 检验使用了滑动平均模型的神经网络前向传播结果是否正确。
    # tf.argmax(average_y, 1)计算每个样例的预测答案，
    # average_y是一个batch_size*10的二维数组，每一行表示一个样例的前向传播结果。
    # tf.argmax的第二个参数1表示选取最大值的操作仅在第一个维度进行，即只在行中选取最大值对应的下标，
    # 于是得到一个长度为batch的一维数组，每个样例对应的数字识别结果。
    # tf.equal判断两个张量的每一维是否相等，相等返回True，否则返回False。
    correct_prediction = tf.equal(tf.argmax(average_y, 1), tf.argmax(y_, 1))
    #先将布尔型数值转换为实数型，然后计算平均值，即模型在这一组数据上的正确率
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, dtype=tf.float32))
    writer = tf.summary.FileWriter('path/to/log', tf.get_default_graph())
    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        #准备验证数据。
        #一般在神经网络的训练过程中会通过验证数据来大致判断停止的条件和评判训练的效果。
        validate_feed = {x: mnist.validation.images,
                         y_: mnist.validation.labels}
        #准备测试数据。在真实应用中，这部分数据在训练时是不可见的，在此只是作为模型优劣的最后评价标准。
        test_feed = {x: mnist.test.images,
                     y_: mnist.test.labels}

        #迭代训练神经网络
        for i in range(TRAINING_STEPS):
            #每1000轮输出一次在验证数据集上的测试结果
            #因为MNIST数据集比较小，所以一次可以处理所有的验证数据。为了计算方便，本程序没有将验证数据划分为更小的batch。
            #当神经网络模型比较复杂或验证数据比较大时，太大的batch会导致计算时间过长甚至发生内存溢出错误。
            if i % 1000 == 0:
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                xs, ys = mnist.train.next_batch(BATCH_SIZE)
                sess.run(train_op, feed_dict={x: xs, y_: ys},
                         options=run_options, run_metadata=run_metadata)
                validate_acc = sess.run(accuracy, feed_dict=validate_feed)
                print('After %d training steps, validation accuracy using average model is %g' % (i, validate_acc))
            #产生这一轮使用的一个batch的训练数据，并运行训练过程
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            sess.run(train_op, feed_dict={x: xs, y_: ys})
            writer.add_run_metadata(run_metadata, 'step%03d' % i)


        #在训练结束之后，在测试数据上检测神经网络模型的最终正确率
        test_acc = sess.run(accuracy, feed_dict=test_feed)
        print('After %d training steps, test accuracy using average model is %g' % (TRAINING_STEPS, test_acc))

    writer.close()
    os.system('tensorboard --logdir=path/to/log')


def main(argv=None):
    mnist = input_data.read_data_sets('path/to/MNIST_data', one_hot=True)
    train(mnist)

# TensorFlow提供的一个主程序入口，tf.app.run()会调用上面定义的main函数
if __name__ == '__main__':
    tf.app.run()