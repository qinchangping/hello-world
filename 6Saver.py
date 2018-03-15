__author__ = 'qcp'
# -*- coding:utf-8 -*-
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf

# v1 = tf.Variable(tf.constant(1.0, shape=[1]), name='v1')
# v2 = tf.Variable(tf.constant(2.0, shape=[1]), name='v2')
# result = v1 + v2
# init_op = tf.global_variables_initializer()
# saver = tf.train.Saver()
# with tf.Session() as sess:
# sess.run(init_op)
# saver.save(sess, 'path/to/model/model.ckpt')

'''
saver = tf.train.import_meta_graph('path/to/model/model.ckpt.meta')
with tf.Session() as sess:
    saver.restore(sess,'path/to/model/model.ckpt')
    print(sess.run(tf.get_default_graph().get_tensor_by_name('add:0')))
'''

'''
#加载变量并重命名,通过字典指定，将原来v1加载到v11中
v11 = tf.Variable(tf.constant(1.0, shape=[1]), name='other-v1')
v22 = tf.Variable(tf.constant(2.0, shape=[1]), name='other-v2')
saver=tf.train.Saver({'v1':v11,'v2':v22})
'''

'''
v = tf.Variable(0, dtype=tf.float32, name='v')
for var in tf.global_variables():
    print(var.name)  #输出 v:0
ema = tf.train.ExponentialMovingAverage(0.99)
maintain_average_op = ema.apply(tf.global_variables())
#在申明滑动平均模型后，TensorFlow会自动生成一个影子变量v/ExponentialMovingAverage
#输出：
#v:0
#v/ExponentialMovingAverage:0
for var in tf.global_variables():
    print(var.name)
saver = tf.train.Saver()
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    sess.run(tf.assign(v, 10))
    sess.run(maintain_average_op)
    saver.save(sess, 'path/to/model/model.ckpt')
    print(sess.run([v, ema.average(v)]))
v = tf.Variable(0, dtype=tf.float32, name='v')
saver = tf.train.Saver({'v/ExponentialMovingAverage': v})
#通过变量重命名将滑动平均值直接赋值给v
with tf.Session() as sess:
    saver.restore(sess,'path/to/model/model.ckpt')
    print(sess.run(v))
'''

#为了方便加载重命名滑动平均变量，tf.train.ExponentialMovingAverage类提供了
# variable_to_restore函数来生成tf.train.Saver类所需要的变量重命名字典
'''
v = tf.Variable(0, dtype=tf.float32, name='v')
ema = tf.train.ExponentialMovingAverage(0.99)
print(ema.variables_to_restore())
#output: {'v/ExponentialMovingAverage': <tf.Variable 'v:0' shape=() dtype=float32_ref>}
saver = tf.train.Saver(ema.variables_to_restore())
with tf.Session() as sess:
    saver.restore(sess, 'path/to/model/model.ckpt')
    print(sess.run(v))
'''

#tensorflow提供了convert_variables_to_constants函数，可以将计算图中的变量及其取值通过常量的方式保存，
# 整个TensorFlow计算图可以统一存放在一个文件中。
'''
from tensorflow.python.framework import graph_util

v1 = tf.Variable(tf.constant(1.0, shape=[1]), name='v1')
v2 = tf.Variable(tf.constant(2.0, shape=[1]), name='v2')
result = v1 + v2
init_op = tf.global_variables_initializer()
saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(init_op)
    #导出当前计算图的GraphDef部分，只需要这一部分就可以完成从输入层到输出层的计算过程
    graph_def = tf.get_default_graph().as_graph_def()
    #将图中的变量及其取值转化为常量，同时将图中不必要的节点去掉
    #一些系统运算也会被转化为计算图中的节点（比如变量初始化操作）。
    #如果只关心程序中定义的某些计算时，和这些计算无关的节点就没有必要导出并保存了。
    #在下面一行代码中，最后一个参数['add']给出了需要保存的节点名称。
    #注意这里给出的是计算节点的名称，所以没有后面的:0
    output_graph_def = graph_util.convert_variables_to_constants(
        sess, graph_def, ['add'])
    with tf.gfile.GFile('path/to/model/combined_model.pb','wb') as f:
        f.write(output_graph_def.SerializeToString())
'''

'''
#下面代码可以直接计算定义的假发运算的结果。当只需要得到计算图中某个节点的取值时，是一个更加方便的方法。
from tensorflow.python.platform import gfile

with tf.Session() as sess:
    model_filename = 'path/to/model/combined_model.pb'
    #读取保存的模型文件，将文件解析成对应的GraphDef Protocol Buffer。
    with gfile.FastGFile(model_filename, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    #将graph_def中保存的图加载到当前图中。
    #return_elements=['add:0']给出了返回的张量的名称。注意与节点名称区分。
    result = tf.import_graph_def(graph_def, return_elements=['add:0'])
    print(sess.run(result))
'''

v1 = tf.Variable(tf.constant(1.0, shape=[1]), name='v1')
v2 = tf.Variable(tf.constant(2.0, shape=[1]), name='v2')
result = v1 + v2
saver = tf.train.Saver()
saver.export_meta_graph('path/to/model/model.ckpt.meta.json', as_text=True)