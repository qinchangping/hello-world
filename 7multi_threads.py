__author__ = 'qcp'
# -*- coding:utf-8 -*-
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np
import threading
import time

'''
# 线程中运行的程序，每隔1秒判断是否需要停止并打印自己的ID
def MyLoop(coord, worker_id):
    # 使用tf。Coordinator类提供的协同工具判断当前线程是否需要停止
    while not coord.should_stop():
        # 随机停止所有线程
        if np.random.rand() < 0.1:
            print('stop from id: %d' % worker_id)
            #调用coord.request_stop()来通知其他线程停止
            coord.request_stop()
        else:
            print('working on id: %d' % worker_id)
        time.sleep(1)


coord = tf.train.Coordinator()
#声明创建5个线程
threads = [threading.Thread(target=MyLoop, args=(coord, i,)) for i in range(5)]
#启动所有线程
for t in threads: t.start()
#等待所有线程退出
coord.join(threads)
'''

# QueueRunner
'''
queue = tf.FIFOQueue(100, 'float')
enqueue_op = queue.enqueue([tf.random_normal([1])])
# 使用tf.train.QueueRunner来创建多个线程运行队列的入队操作
# 第一个参数是被操作的队列，
#[enqueue_op] * 5表示需要启动5个线程，每个线程中运行的是enqueue_op
qr = tf.train.QueueRunner(queue, [enqueue_op] * 5)

#将定义过的QueueRunner加入TensorFlow计算图上指定的集合
#函数没有指定集合，则加入默认集合tf.GraphKeys.QUEUE_RUNNERS
tf.train.add_queue_runner(qr)

#定义出队操作
out_tensor = queue.dequeue()

with tf.Session() as sess:
    coord = tf.train.Coordinator()
    #使用tf.train.QueueRunner时，需要明确调用tf.train.start_queue_runners来启动所有线程。
    #否则因为没有线程运行入队操作，当调用出队操作时，程序会一直等待入队操作被运行。
    #tf.train.start_queue_runners函数会默认启动tf.GraphKeys.QUEUE_RUNNERS集合中
    # 所有的QueueRunner，所以一般来说，tf.train.add_queue_runner和tf.train.start_queue_runners
    #函数会指定同一个集合。
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    for i in range(3): print(sess.run(out_tensor)[0])
    coord.request_stop()
    coord.join(threads)
'''


# 7.3.2输入文件队列
# part1：生成样例数据
'''
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

#模拟海量数据情况下将数据写入不同的文件，num_shards定义了总共写入多少个文件，
# instance_per_shard定义了每个文件中有多少个数据
num_shards = 2
instances_per_shard = 2
for i in range(num_shards):
    #将数据分为多个文件时，可以将不同文件以类似000n-of-000m的后缀区分。
    #方面通过正则表达式获取文件列表，又在文件名中加入了更多信息。
    filename = ('path/to/data.tfrecords-%.5d-of-%.5d' % (i, num_shards))
    writer = tf.python_io.TFRecordWriter(filename)
    #将数据封装成Example结构写入TFRecord文件
    for j in range(instances_per_shard):
        example = tf.train.Example(features=tf.train.Features(feature={
            'i': _int64_feature(i),
            'j': _int64_feature(j)}))
        writer.write(example.SerializeToString())
    writer.close()
'''

# part2：读取

# 使用tf.train.match_filenames_once函数来获取文件列表
files = tf.train.match_filenames_once('path/to/data.tfrecords-*')
# 通过tf.train.string_input_producer函数创建输入队列，输入队列中的文件列表为上述files，
# 一般在解决真实问题时，shuffle=True
filename_queue = tf.train.string_input_producer(files, shuffle=False)
# 读取并解析一个样例
reader = tf.TFRecordReader()
_, serialized_example = reader.read(filename_queue)
features = tf.parse_single_example(
    serialized_example,
    features={
        'i': tf.FixedLenFeature([], tf.int64),
        'j': tf.FixedLenFeature([], tf.int64)})
'''
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    # The tf.train.match_filenames_once() function internally creates a "local variable"
    # that stores the matched filenames, but this local variable must be initialized.
    tf.local_variables_initializer().run()
    print(sess.run(files))

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    for i in range(6):
        #每次读一个样例，所有样例读完后自动从头开始。
        # 如果限制tf.train.string_input_producer的num_epochs=1，则报错。
        print(sess.run([features['i'], features['j']]))
    coord.request_stop()
    coord.join(threads)
'''
# 7.3.3组合训练数据：batching
# 假设Example结构中i表示一个样例的特征向量，比如一张图像的像素矩阵，
#j表示该样例对应的标签。
example, label = features['i'], features['j']
#一个batch中样例的个数
batch_size = 3
#组合样例的队列中最多可以储存的样例个数。
#如果队列太大，需要占用很多内存；如果太小，则出队操作可能会因为没有数据而被阻碍（block），从而导致训练效率降低。
#一般来说这个队列的大小和每个batch的大小相关，下面给出了设置队列大小的一种方式。
capacity = 1000 + 3 * batch_size

#使用tf.train.batch函数来组合样例。
example_batch, label_batch = tf.train.batch(
    [example, label], batch_size=batch_size, capacity=capacity)

#使用tf.train.shuffle_batch函数来组合样例。参数类似于tf.train.batch，
# 但是min_after_dequeue参数是独有的。此参数限制了出队时队列中元素的最少个数。
#当队列中元素太少时，随机打乱样例顺序的作用就不大了。此参数保证随机打乱顺序的作用。
#当出队函数被调用但是队列中元素不够时，出队操作将等待更多的元素入队才会完成。
#如果min_after_dequeue被设定，capacity也应该相应调整来满足性能需求。
example_batch, label_batch = tf.train.shuffle_batch(
    [example, label], batch_size=batch_size, capacity=capacity,min_after_dequeue=30)

#tf.initialize_all_variables() is a shortcut to
# tf.initialize_variables(tf.all_variables()),
# which initializes variables in GraphKeys.VARIABLES collection;
#tf.initialize_local_variables() is a shortcut to
# tf.initialize_variables(tf.local_variables()),
# which initializes variables in GraphKeys.LOCAL_VARIABLE collection.
#Variables in GraphKeys.LOCAL_VARIABLES collection are variables
#  that are added to the graph, but not saved or restored.
#match_filenames_once函数最后一行可以看到使用了local Variable，
# 所以session里面要对local变量初始化，对全局变量初始化没用。
#https://www.zhihu.com/question/61834943/answer/335187308

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    tf.local_variables_initializer().run()
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    #获取并打印组合之后的样例。
    #在真实问题中，这个输出一般会作为神经网络的输入。
    for i in range(2):
        cur_example_batch, cur_label_batch = sess.run(
            [example_batch, label_batch])
        print(cur_example_batch, cur_label_batch)
    coord.request_stop()
    coord.join(threads)
'''
batch 输出：
[0 0 1] [0 1 0]
[1 0 0] [1 0 1]
shuffle_batch 输出是打乱顺序的：
[0 0 0] [0 1 1]
[0 1 1] [0 1 1]
'''