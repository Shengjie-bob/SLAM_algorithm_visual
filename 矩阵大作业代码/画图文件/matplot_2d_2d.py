import numpy as np
import cv2 as cv
import pandas as pd
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt



#打开文件
f =open('loss_2d_2d.txt')
lines = f.readlines()
iter_list=[];error_list=[]

#读取对极几何约束限制
for l in lines:
    time_stp,iteration,error =l.split()
    iter_list.append(iteration)
    error_list.append(error)

iter_array = np.array(iter_list,dtype=np.float)  #一定注意不要numpy类型套numpy 否则后面写入文件会存在问题
error_array=np.array(error_list,dtype=np.float)

#画图
fig= plt.figure()
plt.plot(iter_array,error_array,'r^-',linewidth='3') #画图出现直线问题 是变量没有定义为float变量
plt.legend(['constraint'])
# fig.tight_layout(pad=0.4, w_pad=3.0, h_pad=3.0)
plt.xlabel("Keypoints",fontsize=15)
plt.ylabel("Error",fontsize=15)
plt.show()




