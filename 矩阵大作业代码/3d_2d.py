import cv2 as cv
import numpy as np
from numpy import linalg as la

#相机的内参矩阵
camera_matrix = np.array(([520.9, 0, 325.1],
                         [0, 521.0, 249.7],
                         [0, 0, 1.0]), dtype=np.double)
#相机位姿
camera_pos=np.array([0,0,0,1,0,0,0])
#用与pnp求解的深度图像参数
dist_coefs = np.array([0, 0, 0, 0, 0], dtype=np.double)
#保存文件位置
file = open('loss_3d_2d.txt', 'w')

#主函数
def main (camera_pos=camera_pos):
    # 读取图像
    img_1= cv.imread('1.png',cv.IMREAD_COLOR)
    img_2= cv.imread('2.png',cv.IMREAD_COLOR)
    #img_3=np.copy(img_2)
    print(img_2.shape)
    # ORb特征匹配
    kp1,kp2 =find_feature_matches(img_1,img_2)
    # 读取深度信息
    depth_1 =cv.imread('1_depth.png',cv.IMREAD_UNCHANGED)

    # 得到第一帧特征点的相机坐标系3D坐标和第二针特征点的2d坐标
    pt1_3d=[]
    pt2_2d=[]
    for i in range(0,kp1.shape[0]):
        p1 = kp1[i]
        d1 = depth_1[int(p1[1]), int(p1[0])]
        if d1 == 0  :
            continue
        pt1 = pixel2cam(kp1[i], camera_matrix)
        pt2 =kp2[i]
        dd1 =float(d1/5000)
        pt1_3d.append([pt1[0]*dd1,pt1[1]*dd1,dd1])
        pt2_2d.append([pt2[0],pt2[1]])

    pt1_3d = np.array(pt1_3d,dtype=np.float32)
    pt2_2d = np.array(pt2_2d,dtype=np.float32)

    R = np.identity(3,dtype=np.float32)
    t = np.zeros([3,1])

    #使用pnp求解得到旋转向量和平移向量
    retval, rvec, tvec, inliers = cv.solvePnPRansac(pt1_3d, pt2_2d, camera_matrix, dist_coefs,flags=cv.SOLVEPNP_EPNP,useExtrinsicGuess = False, iterationsCount = 100, reprojectionError = 1.76)
    rotM = cv.Rodrigues(rvec)[0]

    #获得新的位姿
    pos_new=transform_T_to_q(rotM,rvec,camera_pos)
    
    camera_pos =pos_new
    print(camera_pos)

    #验证p1 = p2 * R+t
    R_new = rotM
    t_new = tvec
    pt1_new_list =[]
    for i in range(len(pt1_3d)):
        print('p1:'+str(pt1_3d[i]))
        print('p2:'+str(pt2_2d[i]))
        pt_1=pt1_3d[i].transpose()[:,np.newaxis]  #如果想要保证相乘维度（3，）需要转换为（3，1）
        pt1_new = np.dot(R_new,pt_1)+t_new
        pt1_new = camera_matrix.dot(pt1_new)/(pt1_new[2])
        pt1_new = np.squeeze(pt1_new)
        pt1_new_list.append([pt1_new[0],pt1_new[1]])
        print('R*p1+t:'+str(pt1_new))
        #保存信息
        file = open('loss_3d_2d.txt', 'a')
        file.write('point: ' +str(i)+' '+ str(pt2_2d[i][0]) + ' ' + str(pt2_2d[i][1]) + ' '
                   + str(pt1_new[0])+' '+str(pt1_new[1]))
        file.write('\n')
        file.close()
    pt1_new_array =np.array(pt1_new_list)

    #画出重投影的位置和原本图像中特征点
    img_4 = cv.imread('2.png',cv.IMREAD_COLOR)
    for i in range(len(pt2_2d)):
        p_ref = pt2_2d[i]
        p_cur = pt1_new_array[i]
        if p_cur[0] >0 and p_cur[1]>0:
            cv.circle(img_4,(int(p_ref[0]),int(p_ref[1])),2,(0,250,0),2)
            cv.circle(img_4, (int(p_cur[0]), int(p_cur[1])), 2, (0, 0, 255), 2)
    cv.imshow('current', img_4)
    cv.waitKey()

    return 0

#ORb特征匹配
def find_feature_matches(img_1,img_2):
    # orb特征
    orb=cv.ORB_create()
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

    goodMatches = matches[:80]   #最简单的排序找20个反而最好
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
    pts_1 =np.array(pts_1)
    pts_2 = np.array(pts_2)
    return pts_1,pts_2

#像素坐标转相机坐标
def pixel2cam(kp,camera_matrix):
    u= (kp[0]-camera_matrix[0,2])/ camera_matrix[0,0]
    v=(kp[1]-camera_matrix[1,2])/ camera_matrix[1,1]
    kp_cam = np.array([[u],[v]],dtype=np.float)
    return kp_cam
#获得新的相机位姿
def get_position(rotM,tvec,camera_postion):
    camera_postion=np.dot(rotM,camera_postion)+tvec
    return camera_postion

#位姿转换
def transform_T_to_q(rotM, tvec,camera_pos_):
    R = rotM
    t = tvec
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