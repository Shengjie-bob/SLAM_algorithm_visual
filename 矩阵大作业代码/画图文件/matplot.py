import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import pandas as pd
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

# #学习代码范例
# x=[0,1,2,3]
# y=[0,1,1,3]
# plt.plot(x,y,'ro-',linewidth='3')
# plt.axis([-3,3,-3,3])
# plt.xlabel("money",color="r",fontsize=20)
# plt.ylabel("consume",color="r",fontsize=20)
# plt.title('Figure')
# plt.text(2.5,100,'Text')
# #箭头指示
# #指定文字,箭头指向的坐标,文字显示的坐标,箭头的属性
# plt.annotate('max value', xy=(20, 400), xytext=(12.5, 400),
#              arrowprops=dict(facecolor='black', shrink=0.05),
#              )
#
# x=np.arange(0,10,0.01)
# line1,line2=plt.plot(x,np.sin(x),'-',x,np.cos(x),'--') #line1得到2D Lines
# plt.setp(line1,color='r',linewidth='11.0') #设置曲线的宽度
# plt.show()
#
# plt.gird(True)
# plt.show()
#
#
# #分布图
# mu, sigma = 0,1
# x = np.random.normal(mu,sigma,10000)
# n, bins, patches = plt.hist(x,bins=100,facecolor='g', alpha=0.75)
# plt.text(-3, 250, r'$\mu=0,\ \sigma=1$')
# plt.grid(True)
# plt.show()
#
# #散点图
# x = np.random.normal(0, 1, 1000)  # 1000个点的x坐标
# y = np.random.normal(0, 1, 1000) # 1000个点的y坐标
# c = np.random.rand(1000) #1000个颜色
# s = np.random.rand(100)*100 #100种大小
# plt.scatter(x, y, c=c, s=s,alpha=0.5)
# plt.grid(True)
# plt.show()


# img1 = cv.imread('1.png',cv.IMREAD_GRAYSCALE)
# print(img1.shape)
# x =np.arange(0,640)
# y =np.arange(0,480)
# #等高线图
# # x = np.arange(-5, 5, 0.1)
# # y = np.arange(-5, 5, 0.1)
# # xx, yy = np.meshgrid(x, y, sparse=True)
# # z = np.sin(xx**2 + yy**2) / (xx**2 + yy**2)
# plt.contour(img1)
# plt.show()
#
#
# #bar图
# x=np.random.randint(1,10,8)
# y=np.random.randint(1,10,8)
# z=np.random.randint(1,10,8)
# data=pd.DataFrame([x,y,z],index=['X','Y','Z'],columns=list('abcdefgh'))
# data.plot.bar()
# plt.show()
# data.transpose().plot.bar() #data.transpose()转置
# plt.show()

# #得到曲线图
# f =open('as_1.txt')
# lines = f.readlines()
# pos=[];orient=[]
# test_num = 0
# for l in lines:
#     if test_num % 15 == 0:
#         pass
#     else:
#         time_stp1,img,time_stp2,x,y,z,q1,q2,q3,q4 =l.split()
#         postion = list([x,y,z])
#         orientation = list([q1,q2,q3,q4])
#         pos.append(postion)
#         orient.append(orientation)
#     test_num = test_num + 1
# pos = np.array(pos,dtype=np.float)  #一定注意不要numpy类型套numpy 否则后面写入文件会存在问题
# orient=np.array(orient,dtype=np.float)
#
#
# ax = plt.axes(projection='3d')
# ax.plot(pos[:,0], pos[:,1], pos[:,2], c='g')
# # ax.scatter(pos[:,0], pos[:,1], pos[:,2], c='g')
# plt.legend(['ground_truth'])
# plt.xlabel("money",color="r",fontsize=20)
# plt.ylabel("consume",color="r",fontsize=20)
# ax.set_zlabel("consume",color="r",fontsize=20)
# plt.title('Figure')
#
#
#
# pos = pos.transpose()
# fig= plt.figure()
# plt.plot(pos[0,:],pos[1,:],'r^-',linewidth='3') #画图出现直线问题 是变量没有定义为float变量
# plt.legend(['plot'])
# fig.tight_layout(pad=0.4, w_pad=3.0, h_pad=3.0)
# plt.xlabel("money",color="r",fontsize=20)
# plt.ylabel("consume",color="r",fontsize=20)
# plt.title('Figure')
# plt.show()

#
# #保存曲线图
#
# file = open('lg.txt', 'w')
# file.write('position: ' + str(pos[0,0]) + ' ' + str(pos[1,0])+ ' ' + str(pos[2,0]))
# file.write('\n')
# file.close()


#直接法所耗时间
time = [0.013967037200927734/2,0.038452863693237305/2,0.12596392631530762/3,
        0.3111090660095215/4,0.36940503120422363/4,0.8689162731170654/6,
        3.116727828979492/10,2.036206007003784/3,6.165853023529053/9]

#点的数量
points = [10,20,50,80,100,200,500,800,1000]

#位置误差量 norm（matlab计算摘抄）
error =[0.2364,0.3796,0.3309,0.1522,0.0892, 0.1662,0.0694,0.1128, 0.1061]

time =np.array(time)
points =np.array(points)
error=np.array(error)

#画图
fig= plt.figure()
plt.plot(points,time,'r^-',linewidth='3')
plt.plot(points,error,'b^-',linewidth='3')
#画图出现直线问题 是变量没有定义为float变量
plt.legend(['time(s)','error'])
# fig.tight_layout(pad=0.4, w_pad=3.0, h_pad=3.0)
plt.xlabel("Number of points",fontsize=15)
plt.title('Figure')
plt.show()