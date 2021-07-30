import cv2 as cv
import numpy as np

#相机的内参矩阵
camera_matrix = np.array(([520.9, 0, 325.1],
                         [0, 521.0, 249.7],
                         [0, 0, 1.0]), dtype=np.float)
#相机位姿
camera_pos =np.array([0,0,0,1,0,0,0])

#保存txt文件
file = open('loss_2d_2d.txt', 'w')

#主函数
def main (camera_pos=camera_pos):
    #读取图像
    img_1= cv.imread('1.png',cv.IMREAD_COLOR)
    img_2= cv.imread('2.png',cv.IMREAD_COLOR)
    #img_3=np.copy(img_2)
    print(img_2.shape)
    #ORb特征匹配
    kp1,kp2 =find_feature_matches(img_1,img_2)
    R = np.identity(3,dtype=np.double)
    t = np.zeros([3,1])
    #位姿估计
    R_new,t_new =pose_estimation_2d2d(kp1,kp2,R,t)

    pos_new = transform_T_to_q(R_new, t_new, camera_pos)

    camera_pos = pos_new
    print(camera_pos)

    #验证E=t^R*scale
    t_x = np.array([[0,-t_new[2,0],t_new[1,0]],
                    [t_new[2,0],0,-t_new[0,0]],
                    [-t_new[1,0],t_new[0,0],0]])
    print('t^R='+str(np.dot(t_x,R_new)))
    #计算重投影误差并保存文件
    for i in range(len(kp1)):
        pt1 = pixel2cam(kp1[i],camera_matrix)
        pt2 = pixel2cam(kp2[i],camera_matrix)

        d = pt2.transpose().dot(t_x).dot(R_new).dot(pt1)

        print('epipolar constraint='+str(d))

        file = open('loss_2d_2d.txt', 'a')
        file.write('constraint: ' + str(i) + ' ' + str(d))
        file.write('\n')
        file.close()

    return 0

#ORb特征匹配
def find_feature_matches(img_1,img_2):

    #orb特征
    orb=cv.ORB_create()
    kp1,des1=orb.detectAndCompute(img_1,None)
    kp2,des2=orb.detectAndCompute(img_2,None)
    img_3=np.copy(img_2)
    img_3 = img_2.copy()
    cv.drawKeypoints(img_2, kp2, outImage=img_3, color=(0, 255, 0))
    cv.imshow('img_3', img_3)
    cv.waitKey()
    #计算hamming距离找其中前k个
    bf=cv.BFMatcher(cv.NORM_HAMMING,crossCheck=True)
    matches=bf.match(des1,des2)
    matches = sorted(matches,key = lambda x:x.distance)
    print("Find total " + str(len(matches)) + " matches.")


    goodMatches = matches[:80]   #最简单的排序找20个反而最好

    #画出匹配点
    img3 = cv.drawMatches(img_1,kp1,img_2,kp2,goodMatches,img_2,flags=2)

    cv.imshow('img3',img3)
    cv.waitKey()
    pts_1 = []
    pts_2 = []
    #把关键点保存
    for i in range(0, len(goodMatches)):
        p = kp1[goodMatches[i].queryIdx].pt
        p_2= kp2[goodMatches[i].trainIdx].pt
        pts_1.append(p)
        pts_2.append(p_2)
    pts_1 =np.array(pts_1)
    pts_2 = np.array(pts_2)
    return pts_1,pts_2

#位姿估计
def pose_estimation_2d2d(kp1,kp2,R_1,t_1):
    kp1 = np.array(kp1)
    kp2 = np.array(kp2)
    #计算基础矩阵
    fundamental_matrix =cv.findFundamentalMat(kp1,kp2,cv.FM_8POINT)

    # print(str(fundamental_matrix[0]))
    #计算本质矩阵
    #相机的光心

    essential_matrix=cv.findEssentialMat(kp1,kp2,camera_matrix,cv.RANSAC)
    print('E='+str(essential_matrix[0]))

    #计算单应矩阵
    '''    0 - 利用所有点的常规方法
    RANSAC - RANSAC-基于RANSAC的鲁棒算法
    LMEDS - 最小中值鲁棒算法
    RHO - PROSAC-基于PROSAC的鲁棒算法
'''
    homography_matrix = cv.findHomography(kp1,kp2,cv.RANSAC,3)
    # print(str(homography_matrix[0]))

    #从本质矩阵直接恢复旋转与平移信息
    R=None
    t=None
    mask=None
    retval, R_new, t_new, mask_new=cv.recoverPose(essential_matrix[0], kp1, kp2,camera_matrix,R,t,mask)
    print(R_new)
    #R_new,t_new=cv.recoverPose(essential_matrix,kp1,kp2,camera_matrix,R,t)
    return R_new,t_new

#从像素坐标到相机坐标转换
def pixel2cam(kp,camera_matrix):
    u= (kp[0]-camera_matrix[0,2])/ camera_matrix[0,0]
    v=(kp[1]-camera_matrix[1,2])/ camera_matrix[1,1]
    kp_cam = np.array([u,v,1],dtype=np.double)
    kp_cam= kp_cam.transpose()
    return kp_cam

#以下是将旋转矩阵转换为四元数的方法
def transform_T_to_q(rotM, tvec,camera_pos_):
    R = rotM
    t = tvec
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
    q_new = np.array([q0, q1, q2, q3], dtype=np.double)
    q_new = np.squeeze(q_new)
    q_new = q_new[:, None]
    pos_new = np.vstack((pos, q_new))

    return pos_new


def transform_R_to_q(cam_pose_q):
    q0 = cam_pose_q[0];q1 = cam_pose_q[1];q2 = cam_pose_q[2];q3 = cam_pose_q[3]

    R = np.array([[1 - 2 * q2 * q2 - 2 * q3 * q3, 2 * q1 * q2 + 2 * q0 * q3, 2 * q1 * q3 - 2 * q0 * q2],
                  [2 * q1 * q2 - 2 * q0 * q3, 1 - 2 * q1 * q1 - 2 * q3 * q3, 2 * q2 * q3 + 2 * q0 * q1],
                  [2 * q1 * q3 + 2 * q0 * q2, 2 * q2 * q3 - 2 * q0 * q1, 1 - 2 * q1 * q1 - 2 * q2 * q2]],
                 dtype=np.double)
    return R

if __name__ == "__main__":
    main()