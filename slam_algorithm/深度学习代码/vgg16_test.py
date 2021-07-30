import tensorflow as tf
import os
import numpy as np
import math
from search_dara import *
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'  #隐藏提示警告
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

#设置超参数
learning_rate_init=0.00001
keep_prob=1  #可能是这个值太大了 调小减少训练负担
training_epochs=1
batch_size = 1
display_step=20
##训练次数
num_example_per_epoch_for_train=1000
num_example_per_epoch_for_eval=500
#数据集中的输入图像参数
image_size=224
image_channel=3
#位姿向量的大小
p_x_classes =3
p_q_classes =4
#配准文件和保存loss的文件
dataset = 'as.txt'
file_1= open('evaluation_vgg.txt','w')
file_2= open('pose_x_vgg.txt','w')
#生成加训练数据用于模型训练过程

#假数据用来验证模型使用
def get_faked_train_batch(batch_size):
    images=tf.Variable(tf.random_normal(shape=[batch_size,image_size,image_size,image_channel],
                       mean=0.0, stddev=1.0,dtype=tf.float32))
    pose_x =tf.Variable(tf.random_uniform(shape=[batch_size,3],minval=-0.5,maxval=0.5,dtype=tf.float32))
    pose_q =tf.Variable(tf.random_uniform(shape=[batch_size,4],minval=-0.5,maxval=0.5,dtype=tf.float32))
    return images, pose_x, pose_q

def get_faked_test_batch(batch_size):
    images=tf.Variable(tf.random_normal(shape=[batch_size,image_size,image_size,image_channel],
                       mean=0.0, stddev=1.0,dtype=tf.float32))
    pose_x =tf.Variable(tf.random_uniform(shape=[batch_size,3],minval=-0.5,maxval=0.5,dtype=tf.float32))
    pose_q =tf.Variable(tf.random_uniform(shape=[batch_size,4],minval=-0.5,maxval=0.5,dtype=tf.float32))
    return images, pose_x, pose_q




#二维卷积层
def Conv2d_Op(input_op,name,kh,kw,n_out,dh,dw,
              activation_func=tf.nn.relu,activation_name='relu'):
    with tf.name_scope(name) as scope:
        n_in=input_op.get_shape()[-1].value
        kernels=tf.get_variable(scope+'weight',shape=[kh,kw,n_in,n_out],dtype=tf.float32,
                                initializer=tf.contrib.layers.xavier_initializer_conv2d())
        conv=tf.nn.conv2d(input_op,kernels,strides=(1,dh,dw,1),padding='SAME')
        bias_init_val=tf.constant(0.0,shape=[n_out],dtype=tf.float32)
        biases=tf.Variable(bias_init_val,trainable=True,name='bias')
        z=tf.nn.bias_add(conv,biases)
        activation=activation_func(z,name=activation_name)
        return activation
#pool层
def Pool2d_Op(input_op,name,kh=2,kw=2,dh=2,dw=2,padding='SAME',pool_func=tf.nn.max_pool):
    with tf.name_scope(name) as scope:
        return pool_func(input_op,ksize=[1,kh,kw,1],strides=[1,dh,dw,1],
                         padding=padding,name=name)
#全连接层
def FullyConnected_Op(input_op,name,n_out,activation_func=tf.nn.relu,activation_name='relu'):
    with tf.name_scope(name) as scope:
        n_in = input_op.get_shape()[-1].value
        kernels = tf.get_variable(scope + 'weight', shape=[n_in, n_out], dtype=tf.float32,
                                  initializer=tf.contrib.layers.xavier_initializer_conv2d())
        bias_init_val = tf.constant(0.0, shape=[n_out], dtype=tf.float32)
        biases = tf.Variable(bias_init_val, trainable=True, name='bias')
        z=tf.add(tf.matmul(input_op,kernels),biases)
        activation=activation_func(z,name=activation_name)
        return activation

#保存每个节点处的参数分布
def AddActivationSummary(x):
    tf.summary.histogram('/activations',x)
    tf.summary.scalar('/sparsity',tf.nn.zero_fraction(x))

#保存loss值
def AddLossesSummary(losses):
    loss_averages=tf.train.ExponentialMovingAverage(0.9,name='avg')
    loss_averages_op=loss_averages.apply(losses)

    for loss in losses:
        tf.summary.scalar(loss.op.name+'(raw)',loss)
        tf.summary.scalar(loss.op.name+'(avg)',loss_averages.average(loss))
    return loss_averages_op
#打印每个层的shape大小信息
def print_activations(t):
    print(t.op.name,' ',t.get_shape().as_list())

#Vgg-16网络的主体结构
def Inference(images_holder,keep_prob=keep_prob):
    conv1_1=Conv2d_Op(images_holder,name='conv1_1',kh=3,kw=3,n_out=64,dh=1,dw=1)
    pool1=Pool2d_Op(conv1_1,name='pool1',kh=2,kw=2,dh=2,dw=2,padding='SAME')
    print_activations(pool1)
    AddActivationSummary(conv1_1)

    conv2_1 = Conv2d_Op(pool1, name='conv2_1', kh=3, kw=3, n_out=128, dh=1, dw=1)
    pool2 = Pool2d_Op(conv1_1, name='pool2', kh=2, kw=2, dh=2, dw=2, padding='SAME')
    print_activations(pool2)
    AddActivationSummary(conv2_1)

    conv3_1 = Conv2d_Op(pool2, name='conv3_1', kh=3, kw=3, n_out=256, dh=1, dw=1)
    conv3_2 = Conv2d_Op(conv3_1, name='conv3_2', kh=3, kw=3, n_out=256, dh=1, dw=1)
    pool3 = Pool2d_Op(conv3_2, name='pool3', kh=2, kw=2, dh=2, dw=2, padding='SAME')
    print_activations(pool3)
    AddActivationSummary(conv3_2)

    conv4_1 = Conv2d_Op(pool3, name='conv4_1', kh=3, kw=3, n_out=512, dh=1, dw=1)
    conv4_2 = Conv2d_Op(conv4_1, name='conv4_2', kh=3, kw=3, n_out=512, dh=1, dw=1)
    pool4 = Pool2d_Op(conv4_2, name='pool4', kh=2, kw=2, dh=2, dw=2, padding='SAME')
    print_activations(pool4)
    AddActivationSummary(conv4_2)

    conv5_1 = Conv2d_Op(pool4, name='conv5_1', kh=3, kw=3, n_out=512, dh=1, dw=1)
    conv5_2 = Conv2d_Op(conv5_1, name='conv5_2', kh=3, kw=3, n_out=512, dh=1, dw=1)
    pool5 = Pool2d_Op(conv5_2, name='pool5', kh=2, kw=2, dh=2, dw=2, padding='SAME')
    print_activations(pool5)
    AddActivationSummary(conv5_2)

#将二维的特征图转换为一维
    with tf.name_scope('FeatsReshape'):
        features=tf.reshape(pool5,[batch_size,-1])
        feats_dim=features.get_shape()[1].value

    fc1_out=FullyConnected_Op(features,name='fc1',n_out=2000,
                              activation_func=tf.nn.relu,activation_name='relu')
    print_activations(fc1_out)
    
    #位姿中x输出
    with tf.name_scope('dropout_1'):
        fc1_dropout=tf.nn.dropout(fc1_out,keep_prob=keep_prob)

    pos_x = FullyConnected_Op(fc1_dropout, name='fc2', n_out=p_x_classes,
                                activation_func=tf.identity, activation_name='identity')
    print_activations(pos_x)

    #位姿中q输出
    with tf.name_scope('dropout_2'):
        fc2_dropout = tf.nn.dropout(fc1_out, keep_prob=keep_prob)

    pos_q=FullyConnected_Op(fc2_dropout,name='fc3',n_out=p_q_classes,
                             activation_func=tf.identity,activation_name='identity')
    print_activations(pos_q)
    return pos_x,pos_q

#输入的tensor
with tf.name_scope('Inputs'):
    images_holder = tf.placeholder(tf.float32, [batch_size, image_size,
                                                image_size, image_channel],
                                   name='images')
    poses_x_holder = tf.placeholder(tf.float32, [batch_size,3], name='poses_x')
    poses_q_holder = tf.placeholder(tf.float32, [batch_size,4], name='poses_q')

with tf.name_scope('Inference'):
    keep_prob_holder = tf.placeholder(tf.float32, name='KeepProb')
    pos_x, pos_q = Inference(images_holder, keep_prob_holder)
#loss值计算
with tf.name_scope('Loss'):
    cross_entropy_x=tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(pos_x, poses_x_holder))))*200
    cross_entropy_q=tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(pos_q, poses_q_holder))))*300
    total_loss_op = cross_entropy_x+cross_entropy_q
    average_loss_op=AddLossesSummary([total_loss_op])
#训练
with tf.name_scope('Train'):
    learning_rate = tf.placeholder(tf.float32, name='LearningRate')
    global_step = tf.Variable(0, name='global_step', trainable=False, dtype=tf.int64)
    optimizer = tf.train.AdamOptimizer(learning_rate=0.0001, beta1=0.9, beta2=0.999, epsilon=0.00000001, use_locking=False, name='Adam')
    train_op = optimizer.minimize(total_loss_op, global_step=global_step)

#得到训练的batch数据
with tf.name_scope('GetTrainBatch'):
    #images_train, poses_x_train, poses_q_train= get_faked_train_batch(batch_size=batch_size)
    images_train, poses_x_train, poses_q_train= search_data(data_gen)
    tf.summary.image('images',images_train,max_outputs=8)
#得到测试的batch数据
with tf.name_scope('GetTestBatch'):
    images_test, poses_x_test, poses_q_test= search_data(data_gen)
    tf.summary.image('images',images_test,max_outputs=8)
merged_summaries=tf.summary.merge_all()

#初始化操作
init_op = tf.global_variables_initializer()


#读取模型位置
outputFile = "/home/wangsj/slam/vgg/vgg.ckpt"
#真实数据
datasource = get_data()
data_gen = gen_data_batch(datasource)


with tf.Session() as sess:
    #初始化
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())


    # #训练指定的轮数，每一轮的训练样本总数为 ：num_examples_per_epoch_for_train
    # # 创建一个协调器，管理线程
    coord = tf.train.Coordinator()
    # 启动QueueRunner, 此时文件名队列已经进队
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    #加载模型
    saver = tf.train.Saver()
    saver.restore(sess, outputFile)
    results = np.zeros((len(datasource.images),2))


    print("==>>>>>>>>>>==start to test==<<<<<<<<<<===")

    ## 加载训练或测试集的图像
    for i in range(len(datasource.images)):
        np_image = datasource.images[i]
        np_image = np_image[np.newaxis,:]

        pose_q= np.asarray(datasource.poses[i][3:7])
        pose_x= np.asarray(datasource.poses[i][0:3])
        #获得预测的位姿
        pre_x,pre_q=sess.run([pos_x,pos_q],
                             feed_dict={images_holder:np_image,
                                        keep_prob_holder:keep_prob,
                                        learning_rate:learning_rate_init})
        pre_x = np.squeeze(pre_x)
        #保存位置量
        file_2= open('pose_x_vgg.txt','a')
        file_2.write('iteration: '+str(i)+' '+'pose_x: '+str(pre_x[0])+' '+
                        str(pre_x[1])+' '+str(pre_x[2]))
        file_2.write('\n')
        file_2.close()
        #计算误差
        q1 = pose_q / np.linalg.norm(pose_q)
        q2 = pre_q / np.linalg.norm(pre_q)
        d = abs(np.sum(np.multiply(q1,q2)))
        theta = 2 * np.arccos(d) * 180/math.pi
        error_x = np.linalg.norm(pose_x-pre_x)
        results[i,:] = [error_x,theta]
        #保存误差值
        print('Iteration:  ', i, '  Error XYZ (m):  ', error_x, '  Error Q (degrees):  ', theta)
        file_1= open('evaluation_vgg.txt','a')
        file_1.write('iteration: '+str(i)+' '+'error_x: '+str(error_x)+' '+'error_q: '+str(theta))
        file_1.write('\n')
        file_1.close()
    #求误差的平均值
    median_result = np.median(results,axis=0)
    print ('Median error ', median_result[0], 'm  and ', median_result[1], 'degrees.')





