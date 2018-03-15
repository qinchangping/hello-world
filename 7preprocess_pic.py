__author__ = 'qcp'
# -*- coding:utf-8 -*-
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# 给定一张图像，随机调整图像的色彩。因为调整疗毒、对比度、饱和度和色相的顺序会影响最后得到的结果，
# 所以可以定义多种不同的顺序。具体使用哪一种顺序可以在训练数据预处理时随机选择一种，
# 这样可以进一步降低无关因素对模型的影响。
#
def distort_color(image, color_ordering=0):
    if color_ordering == 0:
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
    elif color_ordering == 1:
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
    elif color_ordering == 2:
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
    # 其余略
    return tf.clip_by_value(image, 0.1, 1.0)


# 给定一张解码后的图像、目标图像的尺寸以及图像上的标注框，此函数可以对给出的图像进行预处理。
# 这个函数的输入图像是图像识别问题中原始的训练图像，
# 输出是神经网络模型的输入层。
# 只处理模型的训练数据，对于预测数据，一般不需要使用随机变换步骤。
def preprocess_for_train(image, height, width, bbox):
    # 如果没有提供标注框，则认为整个图像就是需要关注的部分。
    if bbox is None:
        bbox = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4])

    # 转换图像张量的类型
    if image.dtype != tf.float32:
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)

    #随机截取图像，减小需要关注的物体大小对图像识别算法的影响
    bbox_begin, bbox_size, _ = tf.image.sample_distorted_bounding_box(
        tf.shape(image), bounding_boxes=bbox,min_object_covered=0.1)
    distorted_image = tf.slice(image, bbox_begin, bbox_size)
    #将随机截取的图像调整为神经网络输入层的大小。大小调整的算法是随机选择的。
    distorted_image = tf.image.resize_images(
        distorted_image, [height, width], method=np.random.randint(4))
        #method也可以用这种方法指定：tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        #size是一个参数 [,1]
    #随机左右翻转图像
    distorted_image = tf.image.random_flip_left_right(distorted_image)
    #使用随机顺序调整图像色彩
    distorted_image = distort_color(distorted_image, np.random.randint(2))
    return distorted_image


image_raw_data = tf.gfile.FastGFile('path/to/picture.jpg', 'rb').read()
with tf.Session() as sess:
    img_data = tf.image.decode_jpeg(image_raw_data)
    boxes = tf.constant([[[0.05, 0.05, 0.9, 0.7], [0.35, 0.47, 0.5, 0.56]]])

    for i in range(6):
        result = preprocess_for_train(img_data,299,299,boxes)
        plt.imshow(result.eval())
        plt.show()

"""
image_raw_data = tf.gfile.FastGFile('path/to/picture.jpg', 'rb').read()
with tf.Session() as sess:
    # 解码得到三维矩阵
    # png函数：tf.image.decode_png
    img_data = tf.image.decode_jpeg(image_raw_data)
    '''
    #调整图像大小
    #第二和第三个参数为调整后图像的大小，method给出了调整图像的算法
    #method=0：双线性插值法Bilinear interpolation
    #method=1：最近邻居法Nearest neighbor interpolation
    #method=2：双三次插值法Bicubic interpolation
    #method=3：面积插值法Area interpolation
    resized_image = tf.image.resize_images(img_data, [1200, 1000], method=1)

    #
    #
    #s
    croped = tf.image.resize_image_with_crop_or_pad(img_data, 500, 500)
    padded = tf.image.resize_image_with_crop_or_pad(img_data, 1000, 1000)
    #按比例裁剪图像，调整比例(0,1]
    central_cropped = tf.image.central_crop(img_data, 0.5)
    #tf.image.crop_to_bounding_box()
    #tf.image.pad_to_bounding_box()

    #翻转:上下、左右、对角线
    flipped = tf.image.flip_up_down(img_data)
    flipped = tf.image.flip_left_right(img_data)
    transposed = tf.image.transpose_image(img_data)
    #以一定概率翻转
    flipped = tf.image.random_flip_up_down(img_data)
    flipped = tf.image.random_flip_left_right(img_data)

    #和图像翻转类似，调整图像的亮度、对比度、饱和度和色相在很多图像识别应用中都不会影响识别结果。
    #所以在训练神经网络模型时，可以随机调整训练图像的这些属性，使模型尽可能小地受无关因素的影响。

    #图像亮度调整
    adjusted = tf.image.adjust_brightness(img_data, -0.5)
    #在[-max_delta,max_delta]的范围随机调整图像的亮度
    adjusted = tf.image.random_brightness(img_data, max_delta=1.0)

    #对比度
    adjusted = tf.image.adjust_contrast(img_data, -5)
    adjusted = tf.image.random_contrast(img_data, lower=-5, upper=5)

    #色相
    adjusted = tf.image.adjust_hue(img_data,0.1)
    #在[-max_delta,max_delta]的范围随机调整图像的色相，max_delta在[0,0.5]
    adjusted = tf.image.random_hue(img_data,max_delta)

    #饱和度
    adjusted = tf.image.adjust_saturation(img_data,-5)
    adjusted = tf.image.random_saturation(img_data,lower,upper)
    '''
    #标准化，将图像上的亮度均值变为0，方差变为1
    adjusted = tf.image.per_image_standardization(img_data)
    print(sess.run(adjusted))

    #处理标注框

    #print(img_data.eval())
    #plt.imshow(adjusted.eval())
    #plt.imshow(img_data.eval())
    #plt.imshow(padded.eval())
    #plt.show()
'''
    img_data = tf.image.convert_image_dtype(img_data, dtype=tf.uint8)

    encoded_image = tf.image.encode_jpeg(img_data)
    with tf.gfile.GFile('path/to/output.jpg', 'wb') as f:
        f.write(encoded_image.eval())'''
#http://www.tensorfly.cn/tfdoc/api_docs/python/image.html
"""