import numpy as np
import random
import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
from GoogLeNet import GoogLeNet
import cv2 as cv
from tqdm import tqdm
import math
import os

# 显卡选择
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

batch_size = 5
# 最大训练次数
max_iterations = 9000
# 保存文件路径和图像文件路径
directory = ''
directory_img = 'fr1/'
dataset = 'as.txt'
file_1 = open('evaluation.txt', 'w')
file_2 = open('pose_x.txt', 'w')


# 采用类的方式保存数据
class datasource(object):
    def __init__(self, images, poses):
        self.images = images
        self.poses = poses


# 对图像进行中心化操作（论文中所提到的图像预处理）
def centeredCrop(img, output_side_length):
    height, width, depth = img.shape
    new_height = output_side_length
    new_width = output_side_length
    if height > width:
        new_height = output_side_length * height / width
    else:
        new_width = output_side_length * width / height
    height_offset = (new_height - output_side_length) / 2
    width_offset = (new_width - output_side_length) / 2
    # 整型变量-注意  否则可能出现的bug
    # TypeError: slice indices must be integers or None or have an __index__ method
    height_offset = int(height_offset)
    width_offset = int(width_offset)
    cropped_img = img[height_offset:height_offset + output_side_length,
                  width_offset:width_offset + output_side_length]
    return cropped_img


def preprocess(images):
    images_out = []  ##保存最后变换完结果的图像
    # 调整图像大小和中心化操作
    images_cropped = []
    for i in tqdm(range(len(images))):
        X = cv.imread(images[i])
        X = cv.resize(X, (455, 256))
        X = centeredCrop(X, 224)
        images_cropped.append(X)
    # 计算图像的均值
    N = 0
    mean = np.zeros((1, 3, 224, 224))
    for X in tqdm(images_cropped):
        X = np.transpose(X, (2, 0, 1))
        mean[0][0] += X[0, :, :]
        mean[0][1] += X[1, :, :]
        mean[0][2] += X[2, :, :]
        N += 1
    mean[0] /= N
    # 去均值操作，这样使得训练更为稳定
    for X in tqdm(images_cropped):
        X = np.transpose(X, (2, 0, 1))
        X = X - mean
        X = np.squeeze(X)
        X = np.transpose(X, (1, 2, 0))
        images_out.append(X)
    return images_out


def get_data():
    poses = []
    images = []
    # 读取文件加载配准的数据
    with open(directory + dataset) as f:
        test_num = 0
        for line in f:
            # 每15个抽取一个作为交叉验证集
            if test_num % 15 != 0:
                pass
            else:
                # 读取图像文件名和位姿
                timestp1, fname, timestp2, p0, p1, p2, p3, p4, p5, p6 = line.split()
                p0 = float(p0)
                p1 = float(p1)
                p2 = float(p2)
                p3 = float(p3)
                p4 = float(p4)
                p5 = float(p5)
                p6 = float(p6)
                poses.append((p0, p1, p2, p3, p4, p5, p6))
                images.append(directory_img + fname)
            test_num = test_num + 1
    images = preprocess(images)
    return datasource(images, poses)


# 随机抽取样本数据
def gen_data(source):
    while True:
        indices = list(
            range(len(source.images)))  # 错误 TypeError: 'range' object does not support item assignment 需要list变量
        random.shuffle(indices)
        for i in indices:
            image = source.images[i]
            pose_x = source.poses[i][0:3]
            pose_q = source.poses[i][3:7]
            yield image, pose_x, pose_q


# 抽取一个batch数据
def gen_data_batch(source):
    data_gen = gen_data(source)
    while True:
        image_batch = []
        pose_x_batch = []
        pose_q_batch = []
        for _ in range(batch_size):
            image, pose_x, pose_q = next(data_gen)
            image_batch.append(image)
            pose_x_batch.append(pose_x)
            pose_q_batch.append(pose_q)
        yield np.array(image_batch), np.array(pose_x_batch), np.array(pose_q_batch)


def main():
    # 图像的tensor维度为224*224*3
    image = tf.placeholder(tf.float32, [1, 224, 224, 3])
    datasource = get_data()
    results = np.zeros((len(datasource.images), 2))
    # 加载Goolenet网络
    net = GoogLeNet({'data': image})
    # 3层的预测网络输出
    p3_x = net.layers['cls3_fc_pose_xyz']
    p3_q = net.layers['cls3_fc_pose_wpqr']

    # 加载模型文件位置
    init = tf.initialize_all_variables()
    outputFile = "/home/wangsj/slam/PoseNet.ckpt"

    saver = tf.train.Saver()

    with tf.Session() as sess:
        # 加载数据和训练好的模型
        sess.run(init)
        saver.restore(sess, outputFile)

        data_gen = gen_data_batch(datasource)
        # 加载训练或测试集的图像
        for i in range(len(datasource.images)):
            np_image = datasource.images[i]
            np_image = np_image[np.newaxis, :]
            feed = {image: np_image}

            pose_q = np.asarray(datasource.poses[i][3:7])
            pose_x = np.asarray(datasource.poses[i][0:3])
            predicted_x, predicted_q = sess.run([p3_x, p3_q], feed_dict=feed)
            # 将数据的维度降低 方便操作
            pose_q = np.squeeze(pose_q)
            pose_x = np.squeeze(pose_x)
            predicted_q = np.squeeze(predicted_q)
            predicted_x = np.squeeze(predicted_x)
            # 保存预测得到的位姿数据
            file_2 = open('pose_x.txt', 'a')
            file_2.write('iteration: ' + str(i) + ' ' + 'pose_x: ' + str(predicted_x[0]) + ' ' +
                         str(predicted_x[1]) + ' ' + str(predicted_x[2]))
            file_2.write('\n')
            file_2.close()

            # 计算单个样本误差并且保存误差
            q1 = pose_q / np.linalg.norm(pose_q)
            q2 = predicted_q / np.linalg.norm(predicted_q)
            d = abs(np.sum(np.multiply(q1, q2)))
            theta = 2 * np.arccos(d) * 180 / math.pi
            error_x = np.linalg.norm(pose_x - predicted_x)
            results[i, :] = [error_x, theta]
            print('Iteration:  ', i, '  Error XYZ (m):  ', error_x, '  Error Q (degrees):  ', theta)
            file_1 = open('evaluation.txt', 'a')
            file_1.write('iteration: ' + str(i) + ' ' + 'error_x: ' + str(error_x) + ' ' + 'error_q: ' + str(theta))
            file_1.write('\n')
            file_1.close()

    # 计算平均误差
    median_result = np.median(results, axis=0)
    print('Median error ', median_result[0], 'm  and ', median_result[1], 'degrees.')


if __name__ == '__main__':
    main()
