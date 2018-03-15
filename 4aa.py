__author__ = 'qcp'
# -*- coding:utf-8 -*-
import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# hello = tf.constant('hello,tensorflow!')
sess = tf.Session()
# print(sess.run(hello))
a = tf.constant([1.0, 2.0], name='a')
b = tf.constant([2.0, 3.0], name='b')
result = a + b
# 以下三种写法具有相同功能：
# 1
print(sess.run(result))
# 2
with sess.as_default():
    print(result.eval())
# 3
print(result.eval(session=sess))

# print(a.graph)
# print(tf.get_default_graph())

g1 = tf.Graph()
with g1.as_default():
    v = tf.get_variable('v', shape=[1], initializer=tf.zeros_initializer)

g2 = tf.Graph()
with g2.as_default():
    v = tf.get_variable('v', shape=[1], initializer=tf.ones_initializer)
    v1 = tf.get_variable('v1', initializer=tf.ones(shape=[1,2]))#这样显然更简洁

with tf.Session(graph=g1) as sess:
    tf.global_variables_initializer().run()
    with tf.variable_scope('', reuse=True):
        print(sess.run(tf.get_variable('v')))

with tf.Session(graph=g2) as sess:
    tf.global_variables_initializer().run()
    with tf.variable_scope('', reuse=True):
        print(sess.run(tf.get_variable('v')))
        print(sess.run(tf.get_variable('v1')))
