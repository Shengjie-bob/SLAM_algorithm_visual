import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import pandas as pd
from mpl_toolkits import mplot3d





#读取loss值
f =open('loss_vgg.txt')
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
f =open('evaluation_vgg.txt')
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

