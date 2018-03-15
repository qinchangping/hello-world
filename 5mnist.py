__author__ = 'qcp'
# -*- coding:utf-8 -*-
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

'''
mnist = input_data.read_data_sets('path/to/MNIST_data', one_hot=True)
print('training data size:', mnist.train.num_examples)
print('validation data size:', mnist.validation.num_examples)
print('testing data size:', mnist.test.num_examples)
# print('example training data:', mnist.train.images[0])
# print('example training data label:',mnist.train.labels[0])
batch_size = 100
xs, ys = mnist.train.next_batch(batch_size)
print('X shape:', xs.shape)
print('Y shape:', ys.shape)
'''

# 数据集相关参数
INPUT_NODE = 784  # 输入层节点数，对于MNIST数据集，=图片像素
OUTPUT_NODE = 10  # 输出层节点数，=类别数。区分0~9，共10类
# 神经网络相关参数
LAYER1_NODE = 500  # 隐藏层节点数。1层，500个节点
BATCH_SIZE = 100  # 一个训练batch中的训练数据个数。数字越小，越接近随机梯度下降，越大越接近梯度下降
LEARNING_RATE_BASE = 0.8  # 学习率
LEARNING_RATE_DECAY = 0.99  # 学习率的衰减率
REGULARIZATION_RATE = 0.0001  # 描述模型复杂度的正则化项在损失函数中的系数
TRAINING_STEPS = 8000  # 训练轮数
MOVING_AVERAGE_DECAY = 0.99  # 滑动平均衰减率


def inference(input_tensor, avg_class, weights1, biases1, weights2, biases2):
	'''辅助函数，给定社交网络的输入和所有参数，计算神经网络的前向传播结果。
	使用ReLu激活函数的三层全连接神经网络。
	也支持传入用于计算参数平均值的类，方便在测试时使用滑动平均模型。'''
	if avg_class == None:
		layer1 = tf.nn.relu(tf.matmul(input_tensor, weights1) + biases1)
		# 计算输出层前向传播结果。
		# 因为计算损失函数时会一并计算softmax函数，所以这里不需要加入激活函数。
		#而且不加入softmax不会影响预测结果，因为预测时使用的是不同类别对应节点输出值的相对大小，
		#有没有softmax层对最后分类结果的计算没有影响。
		#于是在计算整个神经网络的前向传播时可以不加入最后的softmax层。
		return tf.matmul(layer1, weights2) + biases2
	else:
		# 先使用avg_class.average函数计算得出变量的滑动平均值，再计算前向传播
		layer1 = tf.nn.relu(tf.matmul(input_tensor, avg_class.average(weights1)) + avg_class.average(biases1))
		return tf.matmul(layer1, avg_class.average(weights2)) + avg_class.average(biases2)


# 训练模型的过程
def train(mnist):
	# 784个输入节点
	x = tf.placeholder(dtype=tf.float32, shape=(None, INPUT_NODE), name='x-input')
	# 10个输出节点
	y_ = tf.placeholder(dtype=tf.float32, shape=(None, OUTPUT_NODE), name='y-input')

	# 生成隐藏层的参数
	weights1 = tf.Variable(tf.truncated_normal([INPUT_NODE, LAYER1_NODE], stddev=0.1))
	biases1 = tf.Variable(tf.constant(0.1, shape=[LAYER1_NODE]))

	#生成输出层的参数
	weights2 = tf.Variable(tf.truncated_normal([LAYER1_NODE, OUTPUT_NODE], stddev=0.1))
	biases2 = tf.Variable(tf.constant(0.1, shape=[OUTPUT_NODE]))

	#计算当前参数下神经网络前向传播，不使用滑动平均
	y = inference(x, None, weights1, biases1, weights2, biases2)

	#定义存储训练轮数的变量，
	#不需要计算滑动平均值，所以指定为不可训练量。
	#在使用TensorFlow训练神经网络时，一般会将代表训练轮数的变量指定为不可训练的参数
	global_step = tf.Variable(0, trainable=False)

	#给定滑动平均衰减率和训练轮数变量，初始化滑动平均类
	variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)

	#在所有代表神经网络参数的变量上使用滑动平均。
	#tf.trainable_variables返回图上集合GraphKeys.TRAINABLE_VARIABLES中的元素。
	#即所有没有指定trainable=False的参数。
	variable_averages_op = variable_averages.apply(tf.trainable_variables())

	#计算使用了滑动平均之后的前向传播结果。
	#滑动平均不会改变变量本身的取值，而是维护一个影子变量来记录其滑动平均值。
	#所以当需要使用这个滑动平均值时，需要明确调用average函数。
	average_y = inference(x, variable_averages, weights1, biases1, weights2, biases2)

	#计算交叉熵，作为刻画预测值和真实值之间的差距的损失函数。
	#这里使用了TensorFlow提供的tf.nn.sparse_softmax_cross_entropy_with_logits函数，
	#当分类问题只有一个正确答案时，可以使用这个函数来加速交叉熵的计算。
	#使用此函数要求写明“参数=”来传入
	#logits=神经网络不包括softmax层的前向传播结果，labels=正确答案。注意不包括softmax
	#标准答案是一个长度为10的一维数组，而该函数需要提供一个正确答案数字，
	# 所以需要使用tf.argmax函数来得到正确答案对应的类别编号，1表示在行方向上取值
	cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
	#计算在当前batch中所有样例的交叉熵平均值
	cross_entropy_mean = tf.reduce_mean(cross_entropy)

	#计算L2正则化损失函数
	regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
	#计算模型的正则化损失。一般只计算神经网络边上的权重的正则化损失，而不使用偏置项
	regularization = regularizer(weights1) + regularizer(weights2)
	#总损失等于交叉熵损失+正则化损失
	loss = cross_entropy_mean + regularization
	#设置指数衰减的学习率
	learning_rate = tf.train.exponential_decay(
		LEARNING_RATE_BASE,
		global_step,
		mnist.train.num_examples / BATCH_SIZE,
		LEARNING_RATE_DECAY)

	train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

	#在训练神经网络模型时，每过一遍数据既需要通过反向传播来更新神经网络的参数，
	#又要更新每一个参数的滑动平均值。为了一次完成多个操作，TensorFlow提供了两种机制：
	#tf.control_dependencies()和tf.group()两种机制。
	#with tf.control_dependencies([train_step,variable_averages_op]):
	#	train_op = tf.no_op(name='train')
	train_op = tf.group(train_step, variable_averages_op)

	#检验使用了滑动平均模型的神经网络前向传播结果是否正确。
	#tf.argmax(average_y, 1)计算每个样例的预测答案，
	# average_y是一个batch_size*10的二维数组，每一行表示一个样例的前向传播结果。
	#tf.argmax的第二个参数1表示选取最大值的操作仅在第一个维度进行，即只在行中选取最大值对应的下标，
	#于是得到一个长度为batch的一维数组，每个样例对应的数字识别结果。
	#tf.equal判断两个张量的每一维是否相等，相等返回True，否则返回False。
	correct_prediction = tf.equal(tf.argmax(average_y, 1), tf.argmax(y_, 1))
	#先将布尔型数值转换为实数型，然后计算平均值，即模型在这一组数据上的正确率
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, dtype=tf.float32))

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
				validate_acc = sess.run(accuracy, feed_dict=validate_feed)
				print('After %d training steps, validation accuracy using average model is %g' % (i, validate_acc))
			#产生这一轮使用的一个batch的训练数据，并运行训练过程
			xs, ys = mnist.train.next_batch(BATCH_SIZE)
			sess.run(train_op, feed_dict={x: xs, y_: ys})

		#在训练结束之后，在测试数据上检测神经网络模型的最终正确率
		test_acc = sess.run(accuracy, feed_dict=test_feed)
		print('After %d training steps, test accuracy using average model is %g' % (TRAINING_STEPS, test_acc))


# 主程序入口
def main(argv=None):
	mnist = input_data.read_data_sets('path/to/MNIST_data', one_hot=True)
	train(mnist)

#TensorFlow提供的一个主程序入口，tf.app.run()会调用上面定义的main函数
if __name__ == '__main__':
	tf.app.run()