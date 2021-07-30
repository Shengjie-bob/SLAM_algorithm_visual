import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import pandas as pd
from mpl_toolkits import mplot3d



#得到曲线图
f =open('as_1.txt')
lines = f.readlines()
pos=[];orient=[]
test_num = 0
for l in lines:
    if test_num % 15 == 0:
        pass
    else:
        time_stp1,img,time_stp2,x,y,z,q1,q2,q3,q4 =l.split()
        postion = list([x,y,z])
        orientation = list([q1,q2,q3,q4])
        pos.append(postion)
        orient.append(orientation)
    test_num = test_num + 1
pos = np.array(pos,dtype=np.float)  #一定注意不要numpy类型套numpy 否则后面写入文件会存在问题
orient=np.array(orient,dtype=np.float)

f2 =open('pose_x.txt')
lines = f2.readlines()
pos_pre =[]

for l in lines:
    _,iteration,pose,x_pre,y_pre,z_pre =l.split()
    postion_pre =list([x_pre,y_pre,z_pre])
    pos_pre.append(postion_pre)

pos_pre =np.array(pos_pre,dtype=np.float)


# ax = plt.axes(projection='3d')
# ax.plot(pos[:,0], pos[:,1], pos[:,2], c='g')
# ax.plot(pos_pre[:,0], pos_pre[:,1], pos_pre[:,2], c='b')
# plt.legend(['ground_truth','prediction'])
# plt.xlabel("X coordinate", fontsize=11)
# plt.ylabel("Y coordinate", fontsize=11)
# ax.set_zlabel("Z coordinate", fontsize=11)
# plt.title('Figure')
# plt.show()


#x轴坐标
t= np.arange(0,len(pos))
fig= plt.figure()
plt.plot(t,pos[:,0],'r-',linewidth='3') #画图出现直线问题 是变量没有定义为float变量
plt.plot(t, pos_pre[:,0], 'b-')
plt.legend(['groundtruth','prediction'])

plt.xlabel("Number of pictures",fontsize=15)
plt.ylabel("X coordinate(m)",fontsize=15)
plt.show()

#y轴坐标

t= np.arange(0,len(pos))
fig= plt.figure()
plt.plot(t,pos[:,1],'r-',linewidth='3') #画图出现直线问题 是变量没有定义为float变量
plt.plot(t, pos_pre[:,1], 'b-')
plt.legend(['groundtruth','prediction'])

plt.xlabel("Number of pictures",fontsize=15)
plt.ylabel("Y coordinate(m)",fontsize=15)
plt.show()

#z轴坐标
t= np.arange(0,len(pos))
fig= plt.figure()
plt.plot(t,pos[:,2],'r-',linewidth='3') #画图出现直线问题 是变量没有定义为float变量
plt.plot(t, pos_pre[:,2], 'b-')
plt.legend(['groundtruth','prediction'])

plt.xlabel("Number of pictures",fontsize=15)
plt.ylabel("Z coordinate(m)",fontsize=15)
plt.show()


#读取loss值
f =open('loss_posenet.txt')
lines = f.readlines()
iter_list=[];error_list=[]
#读取loss
for l in lines:
    time_stp,iteration,loss,error =l.split()
    iter_list.append(iteration)
    error_list.append(error)

iter_array = np.array(iter_list,dtype=np.float)  #一定注意不要numpy类型套numpy 否则后面写入文件会存在问题
error_array=np.array(error_list,dtype=np.float)

#画图 loss值
fig= plt.figure()
plt.plot(iter_array,error_array,'b-',linewidth='2') #画图出现直线问题 是变量没有定义为float变量
# fig.tight_layout(pad=0.4, w_pad=3.0, h_pad=3.0)
plt.xlabel("Iteration",fontsize=15)
plt.ylabel("Loss",fontsize=15)
plt.show()


#读取训练误差值
f =open('evaluation_posenet.txt')
lines = f.readlines()
iter_list=[];error_x_list=[];error_q_list=[]
#读取loss
for l in lines:
    time_stp,iteration,error_x,e_x,error_q,e_q =l.split()
    error_x_list.append(e_x)
    error_q_list.append(e_q)
    iter_list.append(iteration)

error_x_array = np.array(error_x_list,dtype=np.float)  #一定注意不要numpy类型套numpy 否则后面写入文件会存在问题
error_q_array=np.array(error_q_list,dtype=np.float)
iter_array =np.array(iter_list,dtype=np.float)

#画图loss值
fig= plt.figure()
plt.plot(iter_array,error_x_array,'g-',linewidth='2') #画图出现直线问题 是变量没有定义为float变量
# fig.tight_layout(pad=0.4, w_pad=3.0, h_pad=3.0)
plt.xlabel("Number of pictures",fontsize=15)
plt.ylabel("Error XYZ(m)",fontsize=15)
plt.show()

fig= plt.figure()
plt.plot(iter_array,error_q_array,'b-',linewidth='2') #画图出现直线问题 是变量没有定义为float变量
# fig.tight_layout(pad=0.4, w_pad=3.0, h_pad=3.0)
plt.xlabel("Number of pictures",fontsize=15)
plt.ylabel("Error Q(degree)",fontsize=15)
plt.show()


#读取测试误差值
f =open('test_posenet.txt')
lines = f.readlines()
iter_list=[];error_x_list=[];error_q_list=[]
#读取loss
for l in lines:
    time_stp,iteration,error_x,e_x,error_q,e_q =l.split()
    error_x_list.append(e_x)
    error_q_list.append(e_q)
    iter_list.append(iteration)

error_x_array = np.array(error_x_list,dtype=np.float)  #一定注意不要numpy类型套numpy 否则后面写入文件会存在问题
error_q_array=np.array(error_q_list,dtype=np.float)
iter_array =np.array(iter_list,dtype=np.float)

#画图loss值
fig= plt.figure()
plt.plot(iter_array,error_x_array,'g-',linewidth='2') #画图出现直线问题 是变量没有定义为float变量
# fig.tight_layout(pad=0.4, w_pad=3.0, h_pad=3.0)
plt.xlabel("Number of pictures",fontsize=15)
plt.ylabel("Error XYZ(m)",fontsize=15)
plt.show()

fig= plt.figure()
plt.plot(iter_array,error_q_array,'b-',linewidth='2') #画图出现直线问题 是变量没有定义为float变量
# fig.tight_layout(pad=0.4, w_pad=3.0, h_pad=3.0)
plt.xlabel("Number of pictures",fontsize=15)
plt.ylabel("Error Q(degree)",fontsize=15)
plt.show()