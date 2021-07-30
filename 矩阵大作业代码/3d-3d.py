import cv2 as cv
import numpy as np
from numpy import linalg as la
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import time
#相机的内参矩阵
camera_matrix = np.array(([520.9, 0, 325.1],
                         [0, 521.0, 249.7],
                         [0, 0, 1.0]), dtype=np.double)
#相机位姿
camera_pos =np.array([0,0,0,1,0,0,0])
#保存txt文件
file = open('loss_3d_3d.txt', 'w')

#主函数
def main (camera_pos=camera_pos):
    # 读取图像
    img_1= cv.imread('1.png',cv.IMREAD_COLOR)
    img_2= cv.imread('2.png',cv.IMREAD_COLOR)
    #img_3=np.copy(img_2)
    print(img_2.shape)
    # ORb特征匹配
    kp1,kp2 =find_feature_matches(img_1,img_2)
    #读取深度信息
    depth_1 =cv.imread('1_depth.png',cv.IMREAD_UNCHANGED)
    depth_2 = cv.imread('2_depth.png',cv.IMREAD_UNCHANGED)

    #得到特征点的相机坐标系坐标
    pt1_3d=[]
    pt2_3d=[]
    for i in range(0,len(kp1)):
        p1 = kp1[i]
        p2 = kp2[i]
        d1 = depth_1[int(p1[1]), int(p1[0])]
        d2 = depth_2[int(p2[1]), int(p2[0])]
        #去除深度为0的点
        if d1 == 0 or d2 ==0 :
            continue
        pt1 = pixel2cam(kp1[i], camera_matrix)
        pt2 = pixel2cam(kp2[i], camera_matrix)
        dd1 =float(d1/5000)
        dd2 = float(d2/5000)
        pt1_3d.append([pt1[0]*dd1,pt1[1]*dd1,dd1])
        pt2_3d.append([pt2[0]*dd2, pt2[1]*dd2, dd2])

    pt1_3d=np.array(pt1_3d)
    pt2_3d=np.array(pt2_3d)
    print(pt1_3d.shape)

    R = np.identity(3,dtype=np.float32)
    t = np.zeros([3,1])
    # 位姿估计
    R_new,t_new =pose_estimation_3d3d(pt1_3d,pt2_3d,R,t)

    #将位姿转换四元数形式
    pos_new = transform_T_to_q(R_new, t_new, camera_pos)

    camera_pos = pos_new
    print(camera_pos)

    #验证p1 = p2 * R+t

    pt1_new_list = []
    for i in range(len(pt1_3d)):
        print('p1:'+str(pt1_3d[i]))
        print('p2:'+str(pt2_3d[i]))
        pt_2=pt2_3d[i].transpose()[:,np.newaxis]  #如果想要保证相乘维度（3，）需要转换为（3，1）
        pt1_new = np.dot(R_new,pt_2)+t_new
        pt1_new = np.squeeze(pt1_new)
        pt1_new_list.append([pt1_new[0], pt1_new[1],pt1_new[2]])
        print('R*p2+t:'+str(pt1_new))
        #保存重投影点和p2点
        file = open('loss_3d_3d.txt', 'a')
        file.write('point: ' +str(i)+' '+ str(pt1_3d[i][0]) + ' ' + str(pt1_3d[i][1]) + ' ' +str(pt1_3d[i][2]) + ' '
                   + str(pt1_new[0])+' '+str(pt1_new[1])+' '+str(pt1_new[2]))
        file.write('\n')
        file.close()

    pt1_new_array = np.array(pt1_new_list)
    #画出上述重投影点和p2点
    ax = plt.axes(projection='3d')
    ax.scatter(pt1_3d[:, 0], pt1_3d[:, 1], pt1_3d[:, 2], c='g')
    ax.scatter(pt1_new_array[:, 0], pt1_new_array[:, 1], pt1_new_array[:, 2], c='r')
    plt.legend(['Keypoints','prediction'])
    plt.xlabel("X coordinate", fontsize=11)
    plt.ylabel("Y coordinate", fontsize=11)
    ax.set_zlabel("Z coordinate", fontsize=11)
    plt.show()

    return 0

#ORb特征匹配
def find_feature_matches(img_1,img_2):

    # orb特征
    orb=cv.ORB_create()
    # 初始时间
    t1 = time.time()
    kp1,des1=orb.detectAndCompute(img_1,None)
    kp2,des2=orb.detectAndCompute(img_2,None)
    img_3=np.copy(img_2)
    img_3 = img_2.copy()
    cv.drawKeypoints(img_2, kp2, outImage=img_3, color=(0, 0, 255))
    # 计算hamming距离找其中前k个
    bf=cv.BFMatcher(cv.NORM_HAMMING,crossCheck=True)
    matches=bf.match(des1,des2)
    matches = sorted(matches,key = lambda x:x.distance)
    print("Find total " + str(len(matches)) + " matches.")

    t2 = time.time()
    t_used =t2 -t1
    #计时统计
    print('feature method:'+str(t_used))

    goodMatches = matches[:60]   #最简单的排序找20个反而最好
    # 画出匹配点
    img3 = cv.drawMatches(img_1,kp1,img_2,kp2,goodMatches,img_2,flags=2)

    cv.imshow('img3',img3)
    cv.waitKey()
    pts_1 = []
    pts_2 = []
    # 把关键点保存
    for i in range(0, len(goodMatches)):
        p = kp1[goodMatches[i].queryIdx].pt
        p_2= kp2[goodMatches[i].trainIdx].pt
        pts_1.append(p)
        pts_2.append(p_2)
    # pts_1 =np.array(pts_1)
    # pts_2 = np.array(pts_2)
    return pts_1,pts_2

#位姿估计
def pose_estimation_3d3d(pt1_3d,pt2_3d,R,t):
    kp1_3d =pt1_3d
    kp2_3d =pt2_3d
    p1 = np.zeros([1,3])
    p2 = np.zeros([1, 3])
    N = np.size(kp1_3d,0)
    #计算p1和p2的平均值
    for i in range(N):
        p1 = p1 + kp1_3d[i]
        p2 = p2 + kp2_3d[i]
    p1 = p1 /N
    p2 = p2 /N
    q1 = np.zeros([N,3])
    q2 = np.zeros([N,3])
    #获得去中心化的q1和q2
    for i in range(N):
        q1[i]=kp1_3d[i]-p1
        q2[i]=kp2_3d[i]-p2
    #W=q1*q2^T
    W = np.zeros([3,3])
    for i in range(N):
        W = W + np.dot(q1.transpose(),q2)
    print('W: '+str(W))

    #svd分解
    U, sigma, VT = la.svd(W)

    R_ = np.dot(U,VT)
    #判断正定性
    #计算特征值
    B = np.linalg.eigvals(R_)
    if np.all(B>0):
        R =R_
    else:
        R =-R_
    #反解t矩阵
    t_ = p1.transpose() - np.dot(R,p2.transpose())   #注意观察p1和p2

    t_new =t_
    R_new =R

    print(str(R))
    return R_new,t_new

#像素坐标转相机坐标
def pixel2cam(kp,camera_matrix):
    u= (kp[0]-camera_matrix[0,2])/ camera_matrix[0,0]
    v=(kp[1]-camera_matrix[1,2])/ camera_matrix[1,1]
    kp_cam = [u,v]
    return kp_cam

#以下是将旋转矩阵转换为四元数的方法
def transform_T_to_q(rotM, tvec,camera_pos_):
    R = rotM.transpose()
    t = -rotM.transpose().dot(tvec)
    print(R)
    print(t)
    pos = np.dot(R, camera_pos_[0:3]) + t
    pos = pos[:,0]
    pos = pos[:,None]

    R_last = transform_R_to_q(camera_pos_[3:7])
    R_new = R.dot(R_last)
    R_new = np.squeeze(R_new)
    q0 = 0.5 * np.sqrt(np.trace(R_new) + 1)
    q1 = (R_new[1, 2] - R_new[2, 1]) / (4 * q0)
    q2 = (R_new[2, 0] - R_new[0, 2]) / (4 * q0)
    q3 = (R_new[0, 1] - R_new[1, 0]) / (4 * q0)
    q_new = np.array([q0, q1, q2, q3], dtype=np.float)
    q_new = np.squeeze(q_new)
    q_new = q_new[:, None]
    pos_new = np.vstack((pos, q_new))

    return pos_new


def transform_R_to_q(cam_pose_q):
    q0 = cam_pose_q[0];q1 = cam_pose_q[1];q2 = cam_pose_q[2];q3 = cam_pose_q[3]

    R = np.array([[1 - 2 * q2 * q2 - 2 * q3 * q3, 2 * q1 * q2 + 2 * q0 * q3, 2 * q1 * q3 - 2 * q0 * q2],
                  [2 * q1 * q2 - 2 * q0 * q3, 1 - 2 * q1 * q1 - 2 * q3 * q3, 2 * q2 * q3 + 2 * q0 * q1],
                  [2 * q1 * q3 + 2 * q0 * q2, 2 * q2 * q3 - 2 * q0 * q1, 1 - 2 * q1 * q1 - 2 * q2 * q2]],
                 dtype=np.float)
    return R


if __name__ == "__main__":
    main()