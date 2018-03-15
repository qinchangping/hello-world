__author__ = 'qcp'
# -*- coding:utf-8 -*-
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import glob
# import os.path
import random
import tensorflow as tf
from tensorflow.python.platform import gfile
import numpy as np

# inception-v3模型瓶颈层节点个数
BOTTLENECK_TENSOR_SIZE = 2048

# inception-v3模型中代表瓶颈层结果的张量名称。
# 在谷歌提供的inception-v3模型中，这个张量名称就是'pool_3/_reshape:0'。
# 在训练模型时，可以通过tensor.name来获取张量的名称。
BOTTLENECK_TENSOR_NAME = 'pool_3/_reshape:0'

# 图像输入张量所对应的名称
JPEG_DATA_TENSOR_NAME = 'DecodeJpeg/contents:0'

# 下载的谷歌训练好的inception-v3模型文件目录
MODEL_DIR = 'path/to/model/'

# 下载的谷歌训练好的inception-v3模型文件名
#MODEL_FILE = 'classify_image_graph_def.pb'
MODEL_FILE = 'tensorflow_inception_graph.pb'

# 因为一个训练数据会被使用很多次，所以可以将
CACHE_DIR = 'tmp/bottleneck'

INPUT_DATA = 'path/to/flower_photos'

# 验证的数据百分比
VALIDATION_PERCENTAGE = 10
# 测试的数据百分比
TEST_PERCENTAGE = 10

# 定义神经网络的设置
LEARNING_RATE = 0.01
STEPS = 4000
BATCH = 100

# 这个函数从数据文件夹中读取所有的图片列表并按训练、验证、测试数据分开。
def create_image_lists(testing_percentage, validation_percentage):
    # 得到的所有图片都存在result这个字典里。这个字典的key为类别的名称。
    # value也是一个字典，字典里存储了所有的图片名称。
    result = {}
    sub_dirs = [x[0] for x in os.walk(INPUT_DATA)]
    #
    is_root_dir = True
    for sub_dir in sub_dirs:
        if is_root_dir:
            is_root_dir = False
            continue

        # 获取当前目录下所有的有效图片文件。
        extensions = ['jpg', 'jpeg', 'JPG', 'JPEG']
        file_list = []
        dir_name = os.path.basename(sub_dir)
        # print('sub_dir=', sub_dir)
        # print('dir_name=', dir_name)
        for extension in extensions:
            file_glob = os.path.join(INPUT_DATA, dir_name, '*.' + extension)
            file_list.extend(glob.glob(file_glob))
        if not file_list: continue

        # 通过目录名称获取类别名称
        label_name = dir_name.lower()
        # 初始化当前类别的训练数据集、测试~、验证~
        training_images = []
        testing_images = []
        validation_images = []
        for file_name in file_list:
            base_name = os.path.basename(file_name)
            # 随机将数据分到训练、测试、验证
            chance = np.random.randint(100)
            if chance < validation_percentage:
                validation_images.append(base_name)
            elif chance < (testing_percentage + validation_percentage):
                testing_images.append(base_name)
            else:
                training_images.append(base_name)
        # 将当前类别的数据放入结果字典
        result[label_name] = {'dir': dir_name,
                              'training': training_images,
                              'testing': testing_images,
                              'validation': validation_images}
    return result


# 这个函数通过类别名称、所属数据集和图片编号获取一张图片的地址。
# image_lists,所有图片信息
# image_dir,根目录。存放图片数据的根目录和存放图片特征向量的根目录地址不同。
# label_name,类别名称
# index,需要获取的图片的编号
# category,指定需要获取的图片是在训练数据集，还是测试/验证~
def get_image_path(image_lists, image_dir, label_name, index, category):
    # 获取给定类别中所有图片信息
    label_lists = image_lists[label_name]
    # 根据所属数据集的名称获取集合中的全部图片信息
    category_list = label_lists[category]
    mod_index = index % len(category_list)
    # 获取图片文件名
    base_name = category_list[mod_index]
    sub_dir = label_lists['dir']
    #
    full_path = os.path.join(image_dir, sub_dir, base_name)
    return full_path


# 这个函数通过类别名称、所属数据集和图片编号获取经过inception-v3模型处理之后的特征向量文件地址
def get_bottleneck_path(image_lists, label_name, index, category):
    return get_image_path(image_lists, CACHE_DIR, label_name, index, category) + '.txt'


# 这个函数使用加载的训练好的inception-v3模型处理一张图片，得到这个图片的特征向量
def run_bottleneck_on_image(sess, image_data, image_data_tensor, bottleneck_tensor):
    # 这个过程实际上就是将当前图片作为输入，计算瓶颈张量的值。
    # 这个值就是这张图片的新的特征向量。
    bottleneck_values = sess.run(bottleneck_tensor, {image_data_tensor: image_data})
    # 通过卷积神经网络处理的结果是一个四维数组，需要将这个结果压缩成一个特征向量（一维数组）。
    bottleneck_values = np.squeeze(bottleneck_values)
    return bottleneck_values


# 这个函数获取一张图片经过inception-v3模型处理之后的特征向量。
# 会先试图寻找已经计算且保存下来的特征向量，如果找不到则先计算，然后保存到文件。
def get_or_create_bottleneck(sess, image_lists, label_name, index, category, jpeg_data_tensor, bottleneck_tensor):
    # 获取一张图片对应的特征向量文件的路径
    label_lists = image_lists[label_name]
    sub_dir = label_lists['dir']
    sub_dir_path = os.path.join(CACHE_DIR, sub_dir)
    if not os.path.exists(sub_dir_path): os.makedirs(sub_dir_path)
    bottleneck_path = get_bottleneck_path(image_lists, label_name, index, category)

    # 如果特征向量不存在，则通过inception-v3模型来计算特征向量，并存入文件
    if not os.path.exists(bottleneck_path):
        # 获取原始图片的路径
        image_path = get_image_path(image_lists, INPUT_DATA, label_name, index, category)
        # 获取图片内容
        image_data = gfile.FastGFile(image_path, 'rb').read()
        bottleneck_values = run_bottleneck_on_image(sess, image_data, jpeg_data_tensor, bottleneck_tensor)
        bottleneck_string = ','.join(str(x) for x in bottleneck_values)
        with open(bottleneck_path, 'w') as bottleneck_file:
            bottleneck_file.write(bottleneck_string)
    else:
        with open(bottleneck_path, 'r') as bottleneck_file:
            bottleneck_string = bottleneck_file.read()
        bottleneck_values = [float(x) for x in bottleneck_string.split(',')]
    return bottleneck_values


# 这个函数随机获取一个batch的图片作为训练数据
def get_random_cached_bottlenecks(sess, n_class, image_lists, how_many, category, jpeg_data_tensor,
                                  bottleneck_tensor):
    bottlenecks = []
    ground_truths = []
    for _ in range(how_many):
        label_index = random.randrange(n_class)
        label_name = list(image_lists.keys())[label_index]
        image_index = random.randrange(65535)
        bottleneck = get_or_create_bottleneck(sess, image_lists, label_name, image_index, category, jpeg_data_tensor,
                                              bottleneck_tensor)
        ground_truth = np.zeros(n_class, dtype=np.float32)
        ground_truth[label_index] = 1.0
        bottlenecks.append(bottleneck)
        ground_truths.append(ground_truth)
    return bottlenecks, ground_truths


# 这个函数获取全部的测试数据。
# 在最终测试时需要在所有的测试数据上计算正确率
def get_test_bottlenecks(sess, image_lists, n_class, jpeg_data_tensor, bottleneck_tensor):
    bottlenecks = []
    ground_truths = []
    label_name_list = list(image_lists.keys())
    # 枚举所有的类别和每个类别中的测试图片
    for label_index, label_name in enumerate(label_name_list):
        category = 'testing'
        for index, unused_base_name in enumerate(image_lists[label_name][category]):
            bottleneck = get_or_create_bottleneck(sess, image_lists, label_name, index, category, jpeg_data_tensor,
                                                  bottleneck_tensor)
            ground_truth = np.zeros(n_class, dtype=np.float32)
            ground_truth[label_index] = 1.0
            bottlenecks.append(bottleneck)
            ground_truths.append(ground_truth)
    return bottlenecks, ground_truths


def main(_):
    # 读取所有图片
    image_lists = create_image_lists(TEST_PERCENTAGE, VALIDATION_PERCENTAGE)
    n_class = len(image_lists.keys())
    # 读取已经训练好的inception-v3模型
    with gfile.FastGFile(os.path.join(MODEL_DIR, MODEL_FILE), 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        # 加载读取的inception-v3模型，并返回输入所对应的张量以及计算瓶颈层结果所对应的张量
        bottleneck_tensor, jpeg_data_tensor = tf.import_graph_def(
            graph_def, return_elements=[BOTTLENECK_TENSOR_NAME, JPEG_DATA_TENSOR_NAME])

        # 定义新的神经网络输入
        # 这个输入就是新的图片经过inception-v3模型前向传播到达瓶颈层时的节点取值。
        bottleneck_input = tf.placeholder(tf.float32, [None, BOTTLENECK_TENSOR_SIZE],
                                          name='BottleneckInputPlaceholder')
        # 定义新的标准答案输入
        ground_truth_input = tf.placeholder(tf.float32, [None, n_class], name='GroundTruthInput')
        # 定义一层全连接层来解决新的图片分类问题。
        with tf.name_scope('final_training_ops'):
            weights = tf.Variable(tf.truncated_normal([BOTTLENECK_TENSOR_SIZE, n_class], stddev=0.001))
            biases = tf.Variable(tf.zeros([n_class]))
            logits = tf.matmul(bottleneck_input, weights) + biases
            final_tensor = tf.nn.softmax(logits)

        # 定义交叉熵损失函数
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=ground_truth_input, logits=logits)
        cross_entropy_mean = tf.reduce_mean(cross_entropy)
        train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cross_entropy_mean)

        # 计算正确率
        with tf.name_scope('evaluation'):
            correct_prediction = tf.equal(tf.argmax(final_tensor, 1), tf.argmax(ground_truth_input, 1))
            evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)

            # 训练过程
            for i in range(STEPS):
                train_bottlenecks, train_ground_truth = \
                    get_random_cached_bottlenecks(sess, n_class, image_lists, BATCH,
                                                  'training', jpeg_data_tensor,
                                                  bottleneck_tensor)
                sess.run(train_step, feed_dict={bottleneck_input: train_bottlenecks,
                                                ground_truth_input: train_ground_truth})

                # 在验证数据上测试正确率
                if i % 100 == 0 or i + 1 == STEPS:
                    validation_bottlenecks, validation_ground_truth = \
                        get_random_cached_bottlenecks(sess, n_class, image_lists, BATCH,
                                                      'validation', jpeg_data_tensor,
                                                      bottleneck_tensor)
                    validation_accuracy = sess.run(
                        evaluation_step, feed_dict={bottleneck_input: validation_bottlenecks,
                                                    ground_truth_input: validation_ground_truth})
                    print('Step %d: Validation accuracy on random sampled %d '
                          'examples = %.1f%%' % (i, BATCH, validation_accuracy * 100))

            # 在最后的测试数据上测试正确率
            test_bottlenecks, test_ground_truth = get_test_bottlenecks(
                sess, image_lists, n_class, jpeg_data_tensor, bottleneck_tensor)
            test_accuracy = sess.run(
                evaluation_step, feed_dict={bottleneck_input: test_bottlenecks,
                                            ground_truth_input: test_ground_truth})
            #%.1f%%是指明输出格式，.1就是0.1，0可以不写，表示输出只保留小数点后一位。%%是输出一个百分号，前面的%是格式符
            print('Final test accuracy = %.1f%%' % (test_accuracy * 100))


if __name__ == '__main__':
    tf.app.run()