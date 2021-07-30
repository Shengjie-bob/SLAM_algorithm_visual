import  numpy as np
import  math
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from scipy.sparse import coo_matrix


#主函数
def main():
    #加载原始的地图模型
    file = open("./data/killian-v.dat")
    var_list = file.readlines()
    vmeans = []
    #读取xy和旋转sieta坐标
    for st in var_list:
        st = st.split()
        # print(st)
        vmeans.append(st[2:5])
        # print(vmeans)
    vmeans =np.array(vmeans,dtype=np.float).transpose()
    print(vmeans.shape)
    #加载观测量
    file= open("data/killian-e.dat")
    var_list =file.readlines()
    eids=[]
    emeans =[]
    einfs=[]
    #读取观测量和权重信息
    for st in var_list:
        st=st.split()
        eids.append(st[1:3])
        emeans.append(st[3:6])
        einf = np.array(([st[6],st[7],st[10]],
                         [st[7],st[8],st[11]],
                         [st[10],st[11],st[9]]))
        einfs.append(einf)
    eids = np.array(eids,dtype=np.int).transpose()
    einfs = np.array(einfs,dtype=np.float).transpose()
    emeans = np.array(emeans,dtype=np.float).transpose()
    #绘制初始地图

    plt.plot(vmeans[0,:],vmeans[1,:],linewidth='2')
    plt.xlabel("X coordinate", fontsize=15)
    plt.ylabel("Y coordinate", fontsize=15)
    plt.show()
    #后端优化
    new_v,correct_list=ls_slam(vmeans,eids,emeans,einfs,5)
    #绘制更新地图
    correct_array =np.array(correct_list,dtype=np.float)
    print(correct_array.shape)
    plt.figure()
    plt.plot(new_v[0,:],new_v[1,:],linewidth='2')
    plt.xlabel("X coordinate", fontsize=15)
    plt.ylabel("Y coordinate", fontsize=15)
    plt.show()

    #画每次迭代更新图
    fig =plt.figure()
    points =np.arange(0,1941)
    colors =['orange','purple','hotpink','yellow','blue']
    for i in range(5):
        plt.plot(points,correct_array[i,:],c=colors[i],linewidth='2',label=str(i+1)+' '+'iteration')
    plt.legend()
    plt.xlabel("Number of mapping points", fontsize=15)
    plt.ylabel("Correction", fontsize=15)
    plt.show()

    print(new_v.shape)

#图优化主函数
def ls_slam(vmeans,eids,emeans,einfs,n):
    correct_list=[]
    for i in range(n):
        vmeans_2 = linearize_and_solve(vmeans,eids,emeans,einfs)
        correct=np.linalg.norm(vmeans_2-vmeans,axis=0)
        vmeans=vmeans_2
        correct_list.append(correct)
    newmeans = vmeans
    return newmeans,correct_list
#将x，y，角度转换为矩阵
def v2t(v):
    c = np.cos(v[2])
    s = np.sin(v[2])
    A =np.array(([c,-s,v[0]],
                  [s,c,v[1]],
                 [0,0,1]))
    return A
#将矩阵转换为x，y，角度
def t2v(A):
    v = np.zeros([3,1])
    v[0:2,0] =A[0:2,2]
    v[2,0] = math.atan2(A[1,0],A[0,0])
    return v

#构建误差函数e和计算雅可比矩阵A，B
#k值一定的注意 实际从0开始
def linear_factors(vmeans,eids,emeans,k):
    id_i = eids[0,k]
    id_j = eids[1,k]
    v_i = vmeans[:,int(id_i)]  #由于得到的id非int变量 不识别需要强制转换
    v_j = vmeans[:,int(id_j)]
    z_ij =emeans[:,k]

    zt_ij = v2t(z_ij)
    vt_i = v2t(v_i)
    vt_j = v2t(v_j)

    f_ij= np.linalg.inv(vt_i).dot(vt_j)

    theta_i = v_i[2]
    ti = v_i[0:2]
    tj =v_j[0:2]
    dt_ij = tj-ti

    si = np.sin(theta_i)
    ci = np.cos(theta_i)
    A =np.array(([-ci,-si, np.array([-si, ci]).dot(dt_ij)],
                 [si, -ci, np.array([-ci,-si]).dot(dt_ij)],
                 [0,0,-1]))
    B = np.array(([ci,si,0],
                  [-si,ci,0],
                  [0,0,1]))
    ztinv = np.linalg.inv(zt_ij)
    e = t2v(ztinv.dot(f_ij))    #误差
    ztinv[0:2,2]= 0
    A = ztinv.dot(A)
    B =ztinv.dot(B)
    return  e,A,B

#求解H*delta_x=-b
def linearize_and_solve(vmeans,eids,emeans,einfs):
    print('allocating workspace\n')
    H = np.zeros([np.size(vmeans,1)*3,np.size(vmeans,1)*3])
    b = np.zeros([np.size(vmeans,1)*3,1])

    print('linearizing\n')
    F_e =0

    for k  in range(np.size(eids,1)):
        id_i = eids[0,k]
        id_j = eids[1,k]
        e,A,B = linear_factors(vmeans,eids,emeans,k)
        omega = einfs[:,:,k]
        #计算H矩阵和b矩阵
        b_i = -A.transpose().dot(omega).dot(e)
        b_j = -B.transpose().dot(omega).dot(e)
        H_ii = A.transpose().dot(omega).dot(A)
        H_ij = A.transpose().dot(omega).dot(B)
        H_jj = B.transpose().dot(omega).dot(B)

        H[id_i*3:(id_i+1)*3,id_i*3:(id_i+1)*3]=\
                H[id_i*3:(id_i+1)*3,id_i*3:(id_i+1)*3]+H_ii
        H[id_j*3:(id_j+1)*3,id_j*3:(id_j+1)*3]=\
                H[id_j*3:(id_j+1)*3,id_j*3:(id_j+1)*3]+H_jj
        H[id_i*3:(id_i+1)*3,id_j*3:(id_j+1)*3]=\
                H[id_i*3:(id_i+1)*3,id_j*3:(id_j+1)*3]+H_ij
        H[id_j*3:(id_j+1)*3,id_i*3:(id_i+1)*3]=\
                H[id_j*3:(id_j+1)*3,id_i*3:(id_i+1)*3]+H_ij.transpose()


        b[id_i*3:(id_i+1)*3] =  b[id_i*3:(id_i+1)*3]+b_i
        b[id_j*3:(id_j+1)*3] =  b[id_j*3:(id_j+1)*3]+b_j

        e_sum = e.transpose().dot(omega).dot(e)
        F_e =F_e+e_sum

    print('loss: '+str(F_e))
    print('done\n')

    H[0:3,0:3] = H[0:3,0:3]+np.eye(3) 
    #SH = coo_matrix(H)   #这部分存在问题，稀疏矩阵表示问题
    print('System size: '+str(np.size(H)))
    print('solving !') 
    deltax = np.linalg.solve(H,b)
    #deltax = np.linalg.pinv(H).dot(b)

    deltax=np.reshape(deltax, (np.size(vmeans, 1),3))
    newmeans = vmeans +deltax.transpose()

    print('Normalizing the angle\n')

    for i in range(np.size(newmeans,1)):
        s = np.sin(newmeans[2,i])
        c = np.cos(newmeans[2,i])
        newmeans[2,i] = math.atan2(s,c)

    print('done\n')

    newmeans = np.array(newmeans)
    return newmeans


if __name__ == "__main__":
    main()

        