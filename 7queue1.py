__author__ = 'qcp'
# -*- coding:utf-8 -*-
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf

# 创建一个先进先出队列，指定队列中最多可以保存两个元素，并指定类型为整数
q = tf.FIFOQueue(2, 'int32')
# 使用enqueque_many函数来初始化队列中的元素。name=None
# 和初始化变量类似，在使用队列之前需要明确地调用这个初始化过程
init = q.enqueue_many(([0, 10],))
#使用dequeque函数将队列中的第一个元素出队列，存在变量x中
x = q.dequeue()
y = x + 1
#将+1后的值重新加入队列
q_inc = q.enqueue([y])

with tf.Session() as sess:
    init.run()
    for _ in range(5):
        v, _ = sess.run([x, q_inc])
        print(v)


#windows目录斜线导致uncode报错，前面加个r即可
r'''
文件名是queue.py，运行报错：

"C:\Program Files\Python36\python.exe" D:/PycharmProjects/learnTF/queue.py
Traceback (most recent call last):
  File "D:/PycharmProjects/learnTF/queue.py", line 7, in <module>
    import tensorflow as tf
  File "C:\Program Files\Python36\lib\site-packages\tensorflow\__init__.py", line 24, in <module>
    from tensorflow.python import *
  File "C:\Program Files\Python36\lib\site-packages\tensorflow\python\__init__.py", line 81, in <module>
    from tensorflow.python import keras
  File "C:\Program Files\Python36\lib\site-packages\tensorflow\python\keras\__init__.py", line 26, in <module>
    from tensorflow.python.keras import activations
  File "C:\Program Files\Python36\lib\site-packages\tensorflow\python\keras\activations\__init__.py", line 22, in <module>
    from tensorflow.python.keras._impl.keras.activations import elu
  File "C:\Program Files\Python36\lib\site-packages\tensorflow\python\keras\_impl\keras\__init__.py", line 21, in <module>
    from tensorflow.python.keras._impl.keras import activations
  File "C:\Program Files\Python36\lib\site-packages\tensorflow\python\keras\_impl\keras\activations.py", line 24, in <module>
    from tensorflow.python.keras._impl.keras.utils.generic_utils import deserialize_keras_object
  File "C:\Program Files\Python36\lib\site-packages\tensorflow\python\keras\_impl\keras\utils\__init__.py", line 21, in <module>
    from tensorflow.python.keras._impl.keras.utils.data_utils import GeneratorEnqueuer
  File "C:\Program Files\Python36\lib\site-packages\tensorflow\python\keras\_impl\keras\utils\data_utils.py", line 25, in <module>
    from multiprocessing.pool import ThreadPool
  File "C:\Program Files\Python36\lib\multiprocessing\pool.py", line 17, in <module>
    import queue
  File "D:\PycharmProjects\learnTF\queue.py", line 10, in <module>
    q = tf.FIFOQueue(2, 'int32')
AttributeError: module 'tensorflow' has no attribute 'FIFOQueue'

网上查不到原因，我以为可能是TensorFlow安装不完全导致没有TIFOQueue，但是写入tf.F时提示了有FIFOQueue，
又升级了TensorFlow：管理员运行powershell，pip install --upgrade tensorflow
成功后还是同样报错。我又试着运行其他程序，结果甚至helloworld都会报这个错。
索性删了这个queue.py试试，大家就当无事发生过，为省事重命名成queue1.py，运行，结果正常！
再打开queue1.py运行，结果也正常！谁tm能想到是因为文件名的原因！
命名与python中的queue包冲突导致报错，文件名不能和已有包中的相同。
'''