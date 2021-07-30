import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import pandas as pd
from mpl_toolkits import mplot3d


#打开文件
f =open('loss_3d_2d.txt')
lines = f.readlines()
p2_list=[];pre_p2_list=[]

#读取文件中点的坐标
for l in lines:
    time_stp,iteration,x2,y2,x2_2,y2_2 =l.split()
    p2 =list([x2,y2])
    pre_p2 =list([x2_2,y2_2])
    p2_list.append(p2)
    pre_p2_list.append(p2)

p2_array = np.array(p2_list,dtype=np.float)  #一定注意不要numpy类型套numpy 否则后面写入文件会存在问题
pre_p2_array=np.array(pre_p2_list,dtype=np.float)

#画图
fig= plt.figure()
plt.scatter(p2_array[:,0],pre_p2_array[:,0],c='g') #画图出现直线问题 是变量没有定义为float变量
# fig.tight_layout(pad=0.4, w_pad=3.0, h_pad=3.0)
plt.xlabel("X coordinate of keypoints",fontsize=15)
plt.ylabel("X coordinate of reprojection",fontsize=15)
plt.show()

fig_2= plt.figure()
plt.scatter(p2_array[:,1],pre_p2_array[:,1],c='g') #画图出现直线问题 是变量没有定义为float变量
# fig.tight_layout(pad=0.4, w_pad=3.0, h_pad=3.0)
plt.xlabel("Y coordinate of keypoints",fontsize=15)
plt.ylabel("Y coordinate of reprojection",fontsize=15)
plt.show()


