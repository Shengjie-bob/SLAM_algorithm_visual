import numpy as np
import random
import tensorflow as tf
from GoogLeNet import GoogLeNet
import cv2 as cv
from tqdm import tqdm
import os

#显卡选择
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

batch_size = 5
#最大训练次数
max_iterations = 9000
# 设置保存路径
directory = ''
directory_img = 'fr1/'
#配准好的图像和位姿文件
dataset = 'as.txt'
#保存loss值
file= open('loss.txt','w')


#采用类的方式保存数据
class datasource(object):
	def __init__(self, images, poses):
		self.images = images
		self.poses = poses

#对图像进行中心化操作（论文中所提到的图像预处理）
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
	#整型变量-注意  否则可能出现的bug
	#TypeError: slice indices must be integers or None or have an __index__ method
	height_offset=int(height_offset)
	width_offset=int(width_offset)
	cropped_img = img[height_offset:height_offset + output_side_length,
						width_offset:width_offset + output_side_length]
	return cropped_img

def preprocess(images):
	images_out = [] #保存最后变换完结果的图像
	#调整图像大小和中心化操作
	images_cropped = []
	for i in tqdm(range(len(images))):
		X = cv.imread(images[i])
		X = cv.resize(X, (455, 256))
		X = centeredCrop(X, 224)
		images_cropped.append(X)
	#计算图像的均值
	N = 0
	mean = np.zeros((1, 3, 224, 224))
	for X in tqdm(images_cropped):
			X = np.transpose(X,(2,0,1))
			mean[0][0] += X[0,:,:]
			mean[0][1] += X[1,:,:]
			mean[0][2] += X[2,:,:]
			N += 1
	mean[0] /= N
	#去均值操作，这样使得训练更为稳定
	for X in tqdm(images_cropped):
			X = np.transpose(X,(2,0,1))
			X = X - mean
			X = np.squeeze(X)
			X = np.transpose(X, (1,2,0))
			images_out.append(X)
	return images_out

def get_data():
	poses = []
	images = []
	#读取文件加载配准的数据
	with open(directory+dataset) as f:
		test_num = 0
		for line in f:
    		#每15个抽取一个作为交叉验证集
			if test_num %15 ==0:
				pass
			else:
    			#读取图像文件名和位姿
				timestp1,fname,timestp2,p0,p1,p2,p3,p4,p5,p6 = line.split()
				p0 = float(p0)
				p1 = float(p1)
				p2 = float(p2)
				p3 = float(p3)
				p4 = float(p4)
				p5 = float(p5)
				p6 = float(p6)
				poses.append((p0,p1,p2,p3,p4,p5,p6))
				images.append(directory_img+fname)
			test_num = test_num+1
	images = preprocess(images)
	return datasource(images, poses)

#随机抽取样本数据
def gen_data(source):
	while True:
		indices = list(range(len(source.images)))   #错误 TypeError: 'range' object does not support item assignment 需要list变量
		random.shuffle(indices)
		for i in indices:
			image = source.images[i]
			pose_x = source.poses[i][0:3]
			pose_q = source.poses[i][3:7]
			yield image, pose_x, pose_q

#抽取一个batch数据
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
    #图像的tensor维度为224*224*3
	images = tf.placeholder(tf.float32, [batch_size, 224, 224, 3])
	poses_x = tf.placeholder(tf.float32, [batch_size, 3])
	poses_q = tf.placeholder(tf.float32, [batch_size, 4])
	datasource = get_data()
	#加载Goolenet网络
	net = GoogLeNet({'data': images})
	#1层和2层和3层的预测网络输出
	p1_x = net.layers['cls1_fc_pose_xyz']
	p1_q = net.layers['cls1_fc_pose_wpqr']
	p2_x = net.layers['cls2_fc_pose_xyz']
	p2_q = net.layers['cls2_fc_pose_wpqr']
	p3_x = net.layers['cls3_fc_pose_xyz']
	p3_q = net.layers['cls3_fc_pose_wpqr']

	#各层的训练误差 尺度因子选为3
	l1_x = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(p1_x, poses_x)))) * 50
	l1_q = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(p1_q, poses_q)))) * 150
	l2_x = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(p2_x, poses_x)))) * 50
	l2_q = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(p2_q, poses_q)))) * 150
	l3_x = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(p3_x, poses_x)))) * 150
	l3_q = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(p3_q, poses_q)))) * 300

	#总体的loss值
	loss = l1_x + l1_q + l2_x + l2_q + l3_x + l3_q
	opt = tf.train.AdamOptimizer(learning_rate=0.0001, beta1=0.9, beta2=0.999, epsilon=0.00000001, use_locking=False, name='Adam').minimize(loss)
	#初始化
	init = tf.global_variables_initializer()
	saver = tf.train.Saver()
	#保存模型的位置
	outputFile = "/home/wangsj/slam/PoseNet.ckpt"   #将outputFile的绝对路径加上即可。ValueError: Parent directory of PoseNet.ckpt doesn't exist, can't save.

	with tf.Session() as sess:
		# 加载数据
		sess.run(init)
		net.load('weights/posenet.npy', sess)
		data_gen = gen_data_batch(datasource)
		#训练
		for i in range(max_iterations):
			np_images, np_poses_x, np_poses_q = next(data_gen)
			feed = {images: np_images, poses_x: np_poses_x, poses_q: np_poses_q}
			print(i)
			sess.run(opt, feed_dict=feed)
			np_loss = sess.run(loss, feed_dict=feed)
			#每20次保存loss值
			if i % 20 == 0:
				print("iteration: " + str(i) + "\n\t" + "Loss is: " + str(np_loss))
				file= open('loss.txt','a')
				file.write('iteration: '+str(i)+' '+'loss: '+str(np_loss))
				file.write('\n')
				file.close()
			#每1000次保存模型
			if i % 1000 == 0:
				saver.save(sess, outputFile)
				print("Intermediate file saved at: " + outputFile)
		#保存模型
		saver.save(sess, outputFile)
		print("Intermediate file saved at: " + outputFile)
		file.close()

if __name__ == '__main__':
	main()
