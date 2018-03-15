__author__ = 'qcp'
# -*- coding:utf-8 -*-
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np

# 简单循环神经网络的前向传播过程
'''
X = [1, 2]
state = [0.0, 0.0]

w_cell_state = np.asarray([[0.1, 0.2], [0.3, 0.4]])
w_cell_input = np.asarray([0.5, 0.6])
b_cell = np.asarray([0.1, -0.1])

w_output = np.asarray([[1.0], [2.0]])
b_output = 0.1

for i in range(len(X)):
    before_activation = np.dot(state, w_cell_state) + X[i] * w_cell_input + b_cell
    state = np.tanh(before_activation)
    final_output = np.dot(state, w_output) + b_output
    print('before_activation: ', before_activation)
    print('state: ', state)
    print('output: ', final_output)
'''
'''
before_activation:  [0.6 0.5]
state:  [0.53704957 0.46211716]
output:  [1.56128388]
before_activation:  [1.2923401  1.39225678]
state:  [0.85973818 0.88366641]
output:  [2.72707101]
'''

# 长短时记忆网络LSTM（long short term memory）结构的循环神经网络的前向传播过程
'''
# 定义一个LSTM结构。在TensorFlow中通过一句简单的命令就可以实现一个完整的LSTM结构。
lstm_hidden_size = 3
batch_size = 10

lstm = tf.contrib.rnn.BasicLSTMCell(lstm_hidden_size)
# 将LSTM中的状态初始化为全0数组
state = lstm.zero_state(batch_size, tf.float32)

loss = 0.0
num_steps = 10
#虽然理论上循环神经网络可以处理任意长度的序列，但是在训练时为了避免梯度消散问题，会规定一个最大的序列长度。
for i in range(num_steps):
    #在第一个时刻声明LSTM结构中使用的变量，在之后的时刻都需要复用之前定义好的变量。
    if i > 0: tf.get_variable_scope().reuse_variables()
    #每一步处理时间序列中的一个时刻。
    #将当前输入和前一时刻状态传入定义的LSTM结构可以得到当前LSTM结构的输出和更新后的状态
    lstm_output, state = lstm.call(current_input, state)
    #将当前时刻LSTM结构的输出传入一个全连接层得到最后的输出。
    final_output = fully_connected(lstm_output)
    #计算当前时刻输出的损失。
    loss += calc_loss(final_output, expected_output)
'''

# 双向循环神经网络
# tf.contrib.rnn.BidirctionalGridLSTMCell

# 深层循环神经网络deepRNN
'''
#定义一个基本的LSTM结构作为循环体的基础结构。
#深层循环神经网络也支持使用其他的循环体。
lstm = tf.contrib.rnn.BasicLSTMCell(lstm_hidden_size)
#通过MultiRNNCell类实现深层循环神经网络中每个时刻的前向传播过程。
stacked_lstm = tf.contrib.rnn.MultiRNNCell([lstm] * number_of_layers)
#其余代码类似
'''

# 深层循环神经网络deepRNN，带有dropout
'''
#定义一个基本的LSTM结构作为循环体的基础结构。
#深层循环神经网络也支持使用其他的循环体。
lstm = tf.contrib.rnn.BasicLSTMCell(lstm_hidden_size)
#使用DropoutWrapper类来实现dropout功能。
#该类通过两个参数来控制dropout的概率，
# input_keep_prob控制输入的dropout概率，
# output_keep_prob控制输出的dropout概率
dropout_lstm = tf.contrib.rnn.DropoutWrapper(lstm,output_keep_prob=0.5)
#通过MultiRNNCell类实现深层循环神经网络中每个时刻的前向传播过程。
stacked_lstm = tf.contrib.rnn.MultiRNNCell([dropout_lstm] * number_of_layers)
#其余代码类似
'''
# 简单例子
# from tensorflow.models.rnn.ptb import reader
# 因为没有这个模块，所以只好下载了reader.py放在当前目录
import reader

DATA_PATH = 'path/to/ptb/data'
HIDDEN_SIZE = 200  # 隐藏层规模
NUM_LAYERS = 2  # 深层循环神经网络中LSTM结构的层数
VOCAB_SIZE = 10000  # 词典规模，加上语句结束标识符和稀有单词标识符总共1w个单词
LEARNING_RATE = 1.0
TRAIN_BATCH_SIZE = 20
TRAIN_NUM_STEP = 35
EVAL_BATCH_SIZE = 1
EVAL_NUM_STEP = 1
NUM_EPOCH = 2  # 使用训练数据的轮数
KEEP_PROB = 0.5
MAX_GRID_NORM = 5  # 用于控制梯度膨胀的参数

# 通过一个PTBModel类来描述模型，这样方便维护循环神经网络中的状态
class PTBModel(object):
    def __init__(self, is_training, batch_size, num_steps):
        # 记录使用的batch大小和截断长度
        self.batch_size = batch_size
        self.num_steps = num_steps

        # 定义输入层。
        # 可以看到输入层的维度和batchs_size*num_steps，
        # 这和ptb_producer函数输出的训练数据batch是一致的。
        self.input_data = tf.placeholder(tf.int32, [batch_size, num_steps])
        # 定义预期输出。维度和ptb_producer函数输出的正确答案维度也是一样的
        self.targets = tf.placeholder(tf.int32, [batch_size, num_steps])

        # 定义使用LSTM结构为循环体结构且使用dropout的深层循环神经网络
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(HIDDEN_SIZE)
        if is_training:
            lstm_cell = tf.contrib.rnn.DropoutWrapper(
                lstm_cell, output_keep_prob=KEEP_PROB)
        cell = tf.contrib.rnn.MultiRNNCell([lstm_cell] * NUM_LAYERS)

        # 初始化最初的状态，也就是全0的向量。
        self.initial_state = cell.zero_state(batch_size, tf.float32)
        # 将单词ID转换成为单词向量。
        # 总共有VOCAB_SIZE个单词，每个单词向量的维度为HIDDEN_SIZE，
        # 所以embedding参数的维度为 *
        embedding = tf.get_variable('embedding', [VOCAB_SIZE, HIDDEN_SIZE])

        # 将原本batchs_size*num_steps个单词ID转化为单词向量，转化后的输入层维度为
        # batchs_size*num_steps*HIDDEN_SIZE
        # Looks up ids in a list of embedding tensors.
        inputs = tf.nn.embedding_lookup(embedding, self.input_data)

        # 只在训练时使用dropout
        if is_training: inputs = tf.nn.dropout(inputs, KEEP_PROB)

        # 定义输出列表。在这里先将不同时刻LSTM结构的输出收集起来，再通过一个全连接层得到最终输出。
        outputs = []
        # state存储不同batch中LSTM的状态，将其初始化为0
        state = self.initial_state
        with tf.variable_scope('RNN'):
            for time_step in range(num_steps):
                if time_step > 0: tf.get_variable_scope().reuse_variables()
                cell_output, state = cell(inputs[:, time_step, :], state)
                outputs.append(cell_output)

        # 把输出队列展开成[batch, hidden_size*num_steps]的形状，然后再reshape
        # 成[batch*num_steps, hidden_size]
        # 函数concat连接两个矩阵，axis=1，
        # 详见https://www.tensorflow.org/api_docs/python/tf/concat
        # If one component of shape is the special value -1, the size of that
        # dimension is computed so that the total size remains constant.
        # In particular, a shape of [-1] flattens into 1-D.
        # At most one component of shape can be -1.
        # https://www.tensorflow.org/api_docs/python/tf/reshape
        output = tf.reshape(tf.concat(outputs, 1), [-1, HIDDEN_SIZE])

        # 将从LSTM得到的输出再经过一个全连接层得到最后的预测结果，最终的预测结果在每一个时刻上
        # 都是一个长度为VOCAB_SIZE的数组，经过softmax层之后表示下一个位置是不同单词的概率。
        weight = tf.get_variable('weight', [HIDDEN_SIZE, VOCAB_SIZE])
        bias = tf.get_variable('bias', [VOCAB_SIZE])
        logits = tf.matmul(output, weight) + bias


        # TensorFlow提供了seqence_loss函数来计算一个序列的交叉熵的和。
        # 预测结果，正确答案(压缩成一维)，损失权重。损失权重都为1，即不同batch和时刻的重要程度是一样的。
        #https://www.tensorflow.org/api_docs/python/tf/contrib/seq2seq/sequence_loss
        #日他大爷的，应该用这个tf.contrib.legacy_seq2seq.sequence_loss_by_example，
        # TensorFlow为什么把函数名字和位置改来改去？烦的一比
        #https://www.tensorflow.org/api_docs/python/tf/contrib/legacy_seq2seq/sequence_loss_by_example
        loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
            [logits],
            [tf.reshape(self.targets,[-1])],
            [tf.ones([batch_size * num_steps], dtype=tf.float32)])

        # 计算得到每个batch的平均损失
        self.cost = tf.reduce_sum(loss) / batch_size
        self.final_state = state

        # 只在训练模型时定义反向传播操作
        if not is_training: return
        trainable_variables = tf.trainable_variables()
        # 通过tf.clip_by_global_norm控制梯度的大小，避免梯度膨胀的问题。
        # t * clip_norm / l2norm(t) This operation is typically used to clip
        # gradients before applying them with an optimizer.
        # 函数gradients(ys,xs),Returns: A list of sum(dy/dx) for each x in xs.
        grads, _ = tf.clip_by_global_norm(
            tf.gradients(self.cost, trainable_variables), MAX_GRID_NORM)
        # 定义优化方法
        optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE)
        # 定义训练步骤
        # re-zips the gradient and value lists back into an iterable of (gradient, value)
        #  tuples which is then passed to the optimizer.apply_gradients method.
        self.train_op = optimizer.apply_gradients(zip(grads, trainable_variables))


# 使用给定的模型model在数据data上运行train_op并返回在全部数据上是perplexity值。
def run_epoch(session, model, data_queue, train_op, output_log, epoch_size):
    total_costs = 0.0
    iters = 0
    state = session.run(model.initial_state)

    # 使用当前数据训练或测试模型
    for step in range(epoch_size):
        feed_dict = {}
        # x, y = session.run(reader.ptb_producer(data, model.batch_size, model.num_steps))
        # x,y=reader.ptb_producer(data, model.batch_size, model.num_steps)
        x, y = session.run(data_queue)
        feed_dict[model.input_data] = x
        feed_dict[model.targets] = y
        for i, (c, h) in enumerate(model.initial_state):
            feed_dict[c] = state[i].c
            feed_dict[h] = state[i].h
        cost, state, _ = session.run([model.cost, model.final_state, train_op], feed_dict=feed_dict)
        # 在当前batch上运行train_op并计算损失值。交叉熵损失函数计算的就是下一个单词为给定单词的概率。
        # 将不同时刻、不同batch的概率加起来就可以得到第二个perplexity公式等号右边的部分，
        # 再将这个和做指数运算就可以得到perplexity值
        total_costs += cost
        iters += model.num_steps


        # 只在训练时输出日志
        if output_log and step % 100 == 0:
            print('After %d steps, perplexity is %.3f' %
                  (step, np.exp(total_costs / iters)))
            #print(total_costs,iters)

    return np.exp(total_costs / iters)


def main(_):
    # 获取原始数据
    train_data, valid_data, test_data, _ = reader.ptb_raw_data(DATA_PATH)

    train_epoch_size = (len(train_data) // TRAIN_BATCH_SIZE) // TRAIN_NUM_STEP
    valid_epoch_size = (len(valid_data) // EVAL_BATCH_SIZE) // EVAL_NUM_STEP
    test_epoch_size = (len(test_data) // EVAL_BATCH_SIZE) // EVAL_NUM_STEP

    # 定义初始化函数
    initializer = tf.random_uniform_initializer(-0.05, 0.05)
    # 定义训练用的循环神经网络模型
    with tf.variable_scope('language_model', reuse=None, initializer=initializer):
        train_model = PTBModel(True, TRAIN_BATCH_SIZE, TRAIN_NUM_STEP)
    # 定义评测用的循环神经网络模型
    with tf.variable_scope('language_model', reuse=True, initializer=initializer):
        eval_model = PTBModel(False, EVAL_BATCH_SIZE, EVAL_NUM_STEP)

    #首先生成数据队列，必须要将数据队列的声明放在启动多线程之前，
    # 不然会出现队列一直等待出队的状态，
    train_queue = reader.ptb_producer(train_data, train_model.batch_size, train_model.num_steps)
    valid_queue = reader.ptb_producer(valid_data, eval_model.batch_size, eval_model.num_steps)
    test_queue = reader.ptb_producer(test_data, eval_model.batch_size, eval_model.num_steps)

    with tf.Session() as session:
        tf.global_variables_initializer().run()
        #tf.local_variables_initializer().run()

        # 开启多线程从而支持ptb_producer()使用tf.train.range_input_producer()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=session, coord=coord)
        # 使用训练数据训练模型
        for i in range(NUM_EPOCH):
            print('In iteration: %d' % (i + 1))
            # 训练
            run_epoch(session, train_model, train_queue, train_model.train_op, True, train_epoch_size)
            # 验证数据评测
            # tf.no_op(): Does nothing. Only useful as a placeholder for control edges.
            valid_perplexity = run_epoch(session, eval_model, valid_queue, tf.no_op(), False, valid_epoch_size)
            print('Epoch: %d Validation Perplexity: %.3f' % (i + 1, valid_perplexity))

        # 测试数据测试
        test_perplexity = run_epoch(session, eval_model, test_queue, tf.no_op(), False, test_epoch_size)
        print('Test Perplexity: %.3f' % test_perplexity)

        coord.request_stop()
        coord.join(threads)


if __name__ == '__main__':
    tf.app.run()

