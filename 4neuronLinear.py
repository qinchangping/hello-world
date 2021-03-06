__author__ = 'qcp'
# -*- coding:utf-8 -*-
import tensorflow as tf
import os
from numpy.random import RandomState

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

batch_size = 8

w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))
# 通过参数seed设置随机数种子，保证每次运行结果是一样的

# x = tf.constant([[0.7, 0.9]])
# 定义placeholder作为存放数据的地方。维度不一定要定义，但如果维度是确定的，定义可以减少出错的可能。
# 在shape的一个维度上使用None可以方便地使用不大的batch。
# 在训练时需要把数据分成较小的batch，但是在测试时可以一次使用全部数据。
# 当数据集比较小时，这样可以方便测试，但数据集比较大时，将大量数据放入一个batch可能导致内存溢出。
x = tf.placeholder(dtype=tf.float32, shape=(None, 2), name='x-input')
y_ = tf.placeholder(dtype=tf.float32, shape=(None, 1), name='y-input')

# 定义神经网络前向传播过程
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

# 定义损失函数
cross_entropy = -tf.reduce_mean(y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0)))

# 定义学习率
learning_rate = 0.001

# 定义反向传播算法来优化神经网络中的参数
train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

#通过随机数生成一个模拟数据集
rdm = RandomState(1)
dataset_size = 128
X = rdm.rand(dataset_size, 2)

#定义规则来给出样本标签
#在这里，x1 + x2 < 1的样例被认为是正样本，用1表示，其他为负样本，用0表示
Y = [[int(x1 + x2 < 1)] for (x1, x2) in X]

with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    print(sess.run(w1))
    print(sess.run(w2))
    '''
    训练前的神经网络参数：
    w1=[[-0.8113182   1.4845988   0.06532937]
        [-2.4427042   0.0992484   0.5912243 ]]
    w2=[[-0.8113182 ]
        [ 1.4845988 ]
        [ 0.06532937]]
    '''

    #设定训练轮数
    STEPS = 5000
    for i in range(STEPS):
        #每次选取batch_size个样本进行训练
        start = (i * batch_size) % dataset_size
        end = min(start + batch_size, dataset_size)

        #通过选取的样本训练神经网络并更新参数
        sess.run(train_step, feed_dict={x: X[start:end], y_: Y[start:end]})

        #每隔一段时间计算在所有数据上的交叉熵
        if i % 1000 == 0:
            total_cross_entropy = sess.run(cross_entropy, feed_dict={x: X, y_: Y})
            print('After %d training steps, cross entropy on all data is %g' % (i, total_cross_entropy))
            '''
            输出结果：
            After 0 training steps, cross entropy on all data is 0.0674925
            After 1000 training steps, cross entropy on all data is 0.0163385
            After 2000 training steps, cross entropy on all data is 0.00907547
            After 3000 training steps, cross entropy on all data is 0.00714436
            After 4000 training steps, cross entropy on all data is 0.00578471
           '''
    print(sess.run(w1))
    print(sess.run(w2))
    '''
    训练后的神经网络参数：
    w1=[[-1.9618275  2.582354   1.6820377]
        [-3.4681718  1.0698231  2.11789  ]]
    w2=[[-1.824715 ]
        [ 2.6854665]
        [ 1.418195 ]]
    '''
            #print(sess.run(y))
            #输出[[3.957578]]
            #print(sess.run(y, feed_dict={x: [[0.7, 0.9], [0.1, 0.4], [0.5, 0.8]]}))

            #weights = tf.Variable(tf.random_normal([2, 3], stddev=2))  # stddev=标准差
            #biases = tf.Variable(tf.zeros([3]))
            # w2 = tf.Variable(weights.initialized_value())
            #w3 = tf.Variable(weights.initialized_value() * 2)
            # print(weights.eval(session=sess))
            #print(sess.run(weights))