__author__ = 'qcp'
# -*- coding:utf-8 -*-
import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
with tf.name_scope('input1'):
    input1 = tf.constant([1.0, 2.0, 3.0], name='input1')
with tf.name_scope('input2'):
    input2 = tf.Variable(tf.random_uniform([3]), name='input2')
output = tf.add_n([input1, input2], name='add')

#请切换到tf.summary.FileWriter接口和行为是相同的; 这只是一个重命名。
#Writes Summary protocol buffers to event files.
#https://www.tensorflow.org/api_docs/python/tf/summary/FileWriter
writer = tf.summary.FileWriter('path/to/log', tf.get_default_graph())
writer.close()
os.system('tensorboard --logdir=path/to/log')

'''
with tf.variable_scope('foo'):
    # 在命名空间foo下获取变量bar，得到的变量名称为foo/bar
    a = tf.get_variable('bar', [1])
    print(a.name)  # foo/bar:0

with tf.variable_scope('bar'):
    # 在命名空间bar下获取变量bar，得到的变量名称为bar/bar
    b = tf.get_variable('bar', [1])
    print(b.name)  # bar/bar:0

with tf.name_scope('a'):
    # 使用tf.Variable()函数生成变量会受tf.name_scope影响，于是变量名称为
    a = tf.Variable([1])
    print(a.name)  # a/Variable:0

    # 使用tf.get_variable函数生成变量不受tf.name_scope影响，
    # 于是变量不在a这个命名空间中
    a = tf.get_variable('b', [1])
    print(a.name)  #b:0

with tf.name_scope('b'):
    #ValueError: Variable b already exists, disallowed. Did you mean
    #  to set reuse=True or reuse=tf.AUTO_REUSE in VarScope? Originally defined at:
    tf.get_variable('b',[1])

'''
