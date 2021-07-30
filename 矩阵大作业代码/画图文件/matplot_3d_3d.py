import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import pandas as pd
from mpl_toolkits import mplot3d


#打开文件
f =open('loss_3d_3d.txt')
lines = f.readlines()
p2_list=[];pre_p2_list=[]

#读取文件中点的3d坐标
for l in lines:
    time_stp,iteration,x2,y2,z2,x2_2,y2_2,z2_2=l.split()
    p2 =list([x2,y2,z2])
    pre_p2 =list([x2_2,y2_2,z2_2])
    p2_list.append(p2)
    pre_p2_list.append(pre_p2)

p2_array = np.array(p2_list,dtype=np.float)  #一定注意不要numpy类型套numpy 否则后面写入文件会存在问题
pre_p2_array=np.array(pre_p2_list,dtype=np.float)

#计算重投影误差
error_list = [];iter_list =[]
for i in range(len(p2_array)):
    error = np.linalg.norm(p2_array[i,:]-pre_p2_array[i,:])
    error_list .append(error)
    iter_list.append(i)

iter_array= np.array(iter_list,dtype=np.float)
error_array=np.array(error_list,dtype=np.float)

#画图
fig= plt.figure()
plt.plot(iter_array,error_array,'r^-',linewidth='3') #画图出现直线问题 是变量没有定义为float变量
plt.xlabel("Keypoints",fontsize=15)
plt.ylabel("The norm of errors",fontsize=15)
plt.show()





