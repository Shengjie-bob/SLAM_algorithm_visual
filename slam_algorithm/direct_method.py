import cv2 as cv
import numpy as np
import math
import time
import os

np.random.seed(1)

#相机位姿
camera_pos=np.array([0,0,0,1,0,0,0])
camera_pos = camera_pos[:,None]

global fx,fy,cx,cy
#数据集所用相机参数
fx = 718.856; fy = 718.856;  cx = 607.1928; cy = 185.2157
baseline = 0.573

# #纹理低图像相机参数
# fx = 520.9; fy = 521.0;  cx = 325.1; cy = 249.7
# baseline=1000

#读取文件
file_1='./left.png'
file_2='./disparity.png'
directory_name='./image'
#读取无纹理图片
file_3='./rgb_notexture.png'
file_4='./depth_notexture.png'

directory_name_2='./image_notexture'
#
f1=os.listdir(directory_name)
# #无纹理图片使用
# f1=os.listdir(directory_name_2)
f1= f1[1:]
f1.sort()

#计算∂e/∂ξ=-(∂I_2)/∂u *∂u/∂q *∂q/∂δξ
class JacobianAccumulator():
    def __init__( self,img1_,img2_,px_ref_,depth_ref_,T21_):
        #图片
        self.img1=img1_
        self.img2=img2_
        #像素点坐标
        self.px_ref=px_ref_
        #初始化的深度
        self.depth_ref =depth_ref_
        #转移矩阵
        self.T21=T21_
        #投影点
        self.projection=np.zeros([len(px_ref_),2])
        #hessian矩阵
        self.H = np.zeros([6,6])
        #偏移量
        self.b =np.zeros([6,1])
        self.cost=0

    def reset(self,T21):
        self.H = np.zeros([6,6])
        self.b =np.zeros([6,1])
        self.cost =0
        self.T21 =T21
    def projected_points(self):
        return self.projection
    def bias(self):
        return self.b
    def hessian(self):
        return self.H
    #返回损失值
    def cost_func(self):
        return self.cost
    def accumated_jacobian(self,Range):
        half_patch_size = 1
        cnt_good=0
        hessian = np.zeros([6,6])
        bias = np.zeros([6,1])
        cost_tmp =0
        for i in Range:
            point = np.array([(self.px_ref[i][0]-cx)/fx,(self.px_ref[i][1]-cy)/fy,1])
            point = point[:,np.newaxis]   #一维变量transpose无效果
            #得到三维坐标点
            point_ref = self.depth_ref[i]*point
            a =self.T21[0:3, 3]
            #旋转后的假想三维点
            point_cur = np.dot(self.T21[0:3,0:3],point_ref)+a

            point_cur = point_cur[:,0]    #注意维度问题

            if point_cur[2] < 0:
                continue
            #得到像素坐标
            u = fx * point_cur[0]/point_cur[2]+cx
            v = fy*point_cur[1]/point_cur[2]+cy
            #保证投影后像素坐标还在图像里
            if (u <half_patch_size or u > self.img2.shape[1]-half_patch_size or
                v < half_patch_size or v >self.img2.shape[0]-half_patch_size):
                continue
            self.projection[i,:]=np.squeeze(np.array([u, v]))
            X = point_cur[0]; Y=point_cur[1];Z=point_cur[2]
            #以下均为算法中计算雅可比的方法
            Z2 =Z *Z; Z_inv =1.0/Z; Z2_inv = Z_inv*Z_inv
            cnt_good+=1
            for x in range(-half_patch_size,half_patch_size,1):
                for y in range(-half_patch_size,half_patch_size,1):
                    #灰度值误差
                    error = GetPixelValue(self.img1,self.px_ref[i][0]+x,self.px_ref[i][1]+y)-\
                            GetPixelValue(self.img2,u+x,v+y)
                    J_pixel_xi = np.zeros([2,6])
                    J_img_pixel = np.zeros([2,1])

                    #像素点对位姿的雅可比矩阵
                    J_pixel_xi[0,0] = fx*Z_inv
                    J_pixel_xi[0,1] = 0
                    J_pixel_xi[0,2] = -fx*X*Z2_inv
                    J_pixel_xi[0,3] = -fx*X*Y*Z2_inv
                    J_pixel_xi[0,4] = fx+fx*X*X*Z2_inv
                    J_pixel_xi[0,5] = -fx*Y*Z_inv

                    J_pixel_xi[1,0]=0
                    J_pixel_xi[1,1] =fy*Z_inv
                    J_pixel_xi[1,2] = -fy*Y*Z2_inv
                    J_pixel_xi[1,3] = -fy-fy*Y*Y*Z2_inv
                    J_pixel_xi[1,4] = fy*X*Y*Z2_inv
                    J_pixel_xi[1,5] = fy*X*Z_inv

                    #计算第二幅图像中u点的像素梯度
                    J1 =0.5*(GetPixelValue(self.img2,u+1+x,v+y)-GetPixelValue(self.img2,u-1+x,v+y))
                    J2=0.5*(GetPixelValue(self.img2,u+x,v+1+y)-GetPixelValue(self.img2,u+x,v-1+y))
                    J_img_pixel = np.array([J1,J2])
                    J_img_pixel=J_img_pixel[:,np.newaxis].transpose()

                    J = -1.0 *(J_img_pixel.dot(J_pixel_xi))
                    #求解hessian
                    hessian += J.transpose().dot(J)
                    bias += -error*J.transpose()
                    cost_tmp +=error*error
            
        if cnt_good:
            self.H +=hessian
            self.b +=bias
            self.cost +=cost_tmp/cnt_good

#主函数
def main():
    #读取图像
    left_img = cv.imread(file_1,cv.IMREAD_GRAYSCALE)
    disparity_img = cv.imread(file_2,cv.IMREAD_GRAYSCALE)
    #读取低纹理图像
    # left_img = cv.imread(file_3,cv.IMREAD_GRAYSCALE)
    # disparity_img = cv.imread(file_4,cv.IMREAD_GRAYSCALE)
    #随机生成的点数数量
    nPoints =100 #80
    #边界
    boarder = 60
    pixels_ref =[]
    depth_ref = []
    #在边界内随机选取点
    for i in range(nPoints):
        x =np.random.randint(boarder,left_img.shape[1]-boarder)    #代表图像的cols宽度
        y = np.random.randint(boarder,left_img.shape[0]-boarder)   #代表图像的rows高度
        disparity = disparity_img[y,x]  #此处需要注意坐标的一致性 是行还是列
        depth = fx*baseline/disparity
        #低纹理图片使用
        # depth = disparity/5000
        depth_ref.append(depth)
        pixels_ref.append(np.array([[x],[y]]))

    T_cur_ref =np.identity(4)
    #读取下一帧图像
    for f in f1:
        c=directory_name +'/' + f
        #低纹理使用
        # c = directory_name_2 + '/' + f
        img = cv.imread(c,cv.IMREAD_GRAYSCALE)
        pose_new = DirectPoseEstimationSingleLayer(left_img,img,pixels_ref,depth_ref,T_cur_ref)
        camera_pos =pose_new
        #DirectPoseEstimationMultiLayer(left_img,img,pixels_ref,depth_ref,T_cur_ref)
        


#得到图像坐标的像素值，采用了双线性插值来逼近非整数点的像素
def GetPixelValue(img,x,y):
    if x < 0: x=0
    if y < 0: y=0
    if x>=img.shape[1] or np.isnan(x): x =img.shape[1]-1
    if y>=img.shape[0]or np.isnan(y): y =img.shape[0]-1
    xx = x -math.floor(x)
    yy = y -math.floor(y)
    value = (1-xx)*(1-yy)*img[int(y),int(x)]+\
            xx*(1-yy)*img[int(y),int(x)+1]+\
            (1-xx)*yy*img[int(y)+1,int(x)]+\
            xx*yy*img[int(y)+1,int(x)+1]
    return np.float(value)

#单个层的直接法
def DirectPoseEstimationSingleLayer(img1,img2,px_ref,depth_ref,T21):
    iteration=10
    cost= 0
    last_cost=0
    #初始时间
    t1 =time.time()
    jaco_accu = JacobianAccumulator(img1,img2,px_ref,depth_ref,T21)

    for iter in range(iteration):
        jaco_accu.reset(T21)
        Range = range(0,len(px_ref))
        #计算得到hessian和bias
        jaco_accu.accumated_jacobian(Range)
        H = jaco_accu.hessian()
        b = jaco_accu.bias()
        #求解方程H∆x=-b
        update = np.linalg.solve(H,b)
        update = transform_se3_to_T(update)
        T21 = update.dot(T21)   #注意此处没有使用李代数的方式更新
        cost = jaco_accu.cost_func()

        if update[0]== 'nan':
            print('update is nan')
            break
        #如果迭代中loss值增加就停止迭代
        if iter >0 and cost > last_cost:
            print('cost increased:'+str(cost)+','+str(last_cost))
            break
        if np.linalg.norm(update)<1e-3:
            break
            
        last_cost =cost
        print('iteration:'+str(iter)+',cost:'+str(cost))

    print('T21='+str(T21))
    #位姿更新
    pos_new=transform_T_to_q(T21,camera_pos)


    print('camera pos: '+str(pos_new))
    t2 = time.time()
    t_used =t2 -t1
    #计时统计
    print('direct method for single layer:'+str(t_used))

    #画出跟踪效果 可以看出移动的范围
    img2_show = cv.cvtColor(img2,cv.COLOR_GRAY2BGR)
    projection =jaco_accu.projected_points()
    for i in range(len(px_ref)):
        p_ref =px_ref[i]
        p_cur = projection[i]
        if p_cur[0] >0 and p_cur[1]>0:
            cv.circle(img2_show,(int(p_cur[0]),int(p_cur[1])),2,(0,250,0),2)
            cv.line(img2_show,(int(p_ref[0]),int(p_ref[1])),(int(p_cur[0]),int(p_cur[1])),(255,0,0))

    cv.imshow('current',img2_show)
    cv.waitKey()
    return  pos_new

#构建图像金字塔来实现位姿估计 暂时功能不完善
def DirectPoseEstimationMultiLayer(img1, img2, px_ref, depth_ref, T21,):
    pyramids =4
    pyramid_scale =0.5
    scale=[1.0,0.5,0.25,0.125]

    pyr1=[];pyr2=[]

    for i in range(pyramids):
        if i == 0:
            pyr1.append(img1)
            pyr2.append(img2)
        else:
            img1_pyr = cv.resize(pyr1[i-1],(int(pyr1[i-1].shape[1]*pyramid_scale),int(pyr1[i-1].shape[0]*pyramid_scale)))
            img2_pyr = cv.resize(pyr2[i-1],(int(pyr2[i-1].shape[1]*pyramid_scale),int(pyr2[i-1].shape[0]*pyramid_scale)))
            pyr1.append(img1_pyr)
            pyr2.append(img2_pyr)

    global fx ,fy ,cx ,cy
    fxG = fx ; fyG=fy ; cxG=cx; cyG=cy
    for level in range(pyramids-1,-1,-1):   #range函数stop的位置是上一个数
        px_ref_pyr =[]
        for px in px_ref:
            px_ref_pyr.append(scale[level]*px)

        fx = fxG*scale[level]
        fy = fyG*scale[level]
        cx = cxG*scale[level]
        cy = cyG*scale[level]

        DirectPoseEstimationSingleLayer(pyr1[level],pyr2[level],px_ref_pyr,depth_ref,T21)

#将李代数形式转换为坐标变换矩阵
def transform_se3_to_T(se3):   #se3是个6*1向量
    theta = np.linalg.norm(se3[3:6])
    a = se3[3:6]/theta
    a_hat =np.array(([0, -a[2],a[1]],
                    [a[2],0,-a[0]],
                    [-a[1],a[0],0]),dtype=np.float)
    R = np.cos(theta)*np.eye(3)+(1-np.cos(theta))*a.dot(a.transpose())+np.sin(theta)*a_hat
    J = np.sin(theta)/theta*np.eye(3)+(1-np.sin(theta)/theta)*a.dot(a.transpose())+(1-np.cos(theta))/theta*a_hat
    t = J.dot(se3[0:3])
    T2= np.array([0,0,0,1])
    T1 = np.hstack((R,t))
    T = np.vstack((T1,T2))

    return T

#得到位姿
def get_position(rotM,tvec,camera_postion):
    camera_postion=np.dot(rotM,camera_postion)+tvec
    return camera_postion

#坐标转换矩阵到四元数转换
def transform_T_to_q(T,camera_pos):
    R=T[0:3, 0:3]
    t =T[0:3,3]
    pos = np.dot(R,camera_pos[0:3])+t[:,None]

    R_last = transform_R_to_q(camera_pos[3:7])
    R_new =R.dot(R_last)
    R_new = np.squeeze(R_new)
    q0 = 0.5*np.sqrt(np.trace(R_new)+1)
    q1 = (R_new[1,2]-R_new[2,1])/(4*q0)
    q2 = (R_new[2,0]-R_new[0,2])/(4*q0)
    q3 = (R_new[0,1]-R_new[1,0])/(4*q0)
    q_new = np.array([q0,q1,q2,q3],dtype=np.float)
    q_new = np.squeeze(q_new)
    q_new = q_new[:, None]
    pos_new = np.vstack((pos,q_new))

    return  pos_new

def transform_R_to_q(cam_pose_q):
    
    q0 = cam_pose_q[0];q1 = cam_pose_q[1];q2 = cam_pose_q[2];q3 = cam_pose_q[3]

    R=np.array([[1-2*q2*q2-2*q3*q3,2*q1*q2+2*q0*q3,2*q1*q3-2*q0*q2],
    [2*q1*q2-2*q0*q3,1-2*q1*q1-2*q3*q3,2*q2*q3+2*q0*q1],
    [2*q1*q3+2*q0*q2,2*q2*q3-2*q0*q1,1-2*q1*q1-2*q2*q2]],dtype=np.float)
    return R
    



if __name__ == "__main__":
    main()
    