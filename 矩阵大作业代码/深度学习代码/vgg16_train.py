import tensorflow as tf
import os
import numpy as np
from search_dara import *
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'  #隐藏提示警告
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

#设置超参数
learning_rate_init=0.00001
keep_prob=1  #可能是这个值太大了 调小减少训练负担
training_epochs=5
batch_size = 5
display_step=20
#训练次数
num_example_per_epoch_for_train=20000
num_example_per_epoch_for_eval=500
#数据集中的输入图像参数
image_size=224
image_channel=3
#位姿向量的大小
p_x_classes =3
p_q_classes =4
#配准文件和保存loss的文件
dataset = 'as.txt'
file= open('loss_vgg.txt','w')
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


#真实数据
datasource = get_data()
data_gen = gen_data_batch(datasource)

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
    cross_entropy_x=tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(pos_x, poses_x_holder))))*150
    cross_entropy_q=tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(pos_q, poses_q_holder))))*300
    total_loss_op = cross_entropy_x+cross_entropy_q
    average_loss_op=AddLossesSummary([total_loss_op])
#训练
with tf.name_scope('Train'):
    learning_rate = tf.placeholder(tf.float32, name='LearningRate')
    global_step = tf.Variable(0, name='global_step', trainable=False, dtype=tf.int64)
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
    train_op = optimizer.minimize(total_loss_op, global_step=global_step)

#得到训练的batch数据
with tf.name_scope('GetTrainBatch'):
    #images_train, poses_x_train, poses_q_train= get_faked_train_batch(batch_size=batch_size)
    images_train, poses_x_train, poses_q_train= search_data(data_gen)
    tf.summary.image('images',images_train,max_outputs=8)
#得到测试的batch数据
with tf.name_scope('GetTestBatch'):
    images_test, poses_x_test, poses_q_test= get_faked_test_batch(batch_size=batch_size)
    tf.summary.image('images',images_test,max_outputs=8)
merged_summaries=tf.summary.merge_all()

#初始化操作
init_op = tf.global_variables_initializer()


print('please write down the Graph to the event and you can check them in tensorboard')

#保存模型的log文件
summary_writer = tf.summary.FileWriter(logdir='logs/vggnet_11')
summary_writer.add_graph(graph=tf.get_default_graph())
summary_writer.flush()

#将评估结果保存在文件中
results_list=list()
results_list.append(['learning_rate',learning_rate_init,
                     'training_epochs',training_epochs,
                     'batch_size',batch_size,
                     'display_step',display_step])
results_list.append(['train_step','train_loss'])

#保存模型位置
outputFile = "/home/wangsj/slam/vgg/vgg.ckpt"

with tf.Session() as sess:
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())

    print('==>>>>>>>>>>>===Start to train the model in the testing==<<<<<<<<<<=====')
    num_example_per_epoch=int(num_example_per_epoch_for_train/batch_size)
    print('Per batch Size: ',batch_size)
    print('Training sample count per epoch: ',num_example_per_epoch_for_train)
    print('Total batch count per epoch: ',num_example_per_epoch)

    #训练的模型次数
    training_step=0

    # #训练指定的轮数，每一轮的训练样本总数为 ：num_examples_per_epoch_for_train
    # # 创建一个协调器，管理线程
    coord = tf.train.Coordinator()
    # 启动QueueRunner, 此时文件名队列已经进队
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    saver = tf.train.Saver()

    for epoch in range(training_epochs):
        for batch_idx in range(num_example_per_epoch):
            #这部分是假数据需要的内容 并不需要用session获得
            # images_batch,poses_x_batch,poses_q_batch=sess.run([images_train,poses_x_train,poses_q_train])
            images_batch,poses_x_batch,poses_q_batch=[images_train,poses_x_train,poses_q_train]
            #运行优化器的训练节点
            _,loss_value,avg_loss_value=sess.run([train_op,total_loss_op,average_loss_op],
                                                 feed_dict={images_holder:images_batch,
                                                            poses_x_holder:poses_x_batch,
                                                            poses_q_holder:poses_q_batch,
                                                            keep_prob_holder:keep_prob,
                                                            learning_rate:learning_rate_init})

            #每调用一次训练节点 training——step加一
            training_step=sess.run(global_step)
            #每20次保存一次loss值
            if training_step % display_step ==0 :


                print('Training epoch: '+str(epoch)+
                      ",Training Step: "+str(training_step)+
                      ", Training Loss= "+"{:.6f}".format(loss_value))

                results_list.append([training_step,loss_value])
                
                #保存loss值
                file= open('loss_vgg.txt','a')
                file.write('iteration: '+str(training_step)+' '+'loss: '+str(loss_value))
                file.write('\n')
                file.close()

                #运行汇总的节点
                summaries_str=sess.run(merged_summaries,
                                       feed_dict={images_holder:images_batch,
                                                  poses_x_holder:poses_x_batch,
                                                  poses_q_holder:poses_q_batch,
                                                  keep_prob_holder: keep_prob})
                summary_writer.add_summary(summary=summaries_str,global_step=training_step)
                summary_writer.flush()
    #训练结束
    summary_writer.close()
    print("training over")
    saver.save(sess, outputFile)
    print("Intermediate file saved at: " + outputFile)
    file.close()

