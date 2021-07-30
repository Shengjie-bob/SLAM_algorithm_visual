import numpy as np
import cv2 as cv
import time
def main():
    ar = 1.0
    br = 2.0
    cr = 1.0
    
    ae = 2.0
    be = -1.0
    ce = 5.0
    N = 100
    w_sigma = 1.0
    inv_sigma =1 /w_sigma

    x_data =[]
    y_data = []
    for i in range(N):
        x = i/100
        x_data.append(x)
        y_data.append(np.exp(ar*x*x+br*x+cr))

    y_data=y_data+np.random.normal(0, w_sigma, N)


    #开始G_N迭代
    interation = 100
    cost =0
    lastCost = 0
    t1 = time.time()

    for iter in range(interation):
        H = np.zeros((3,3))
        b = np.zeros((3,1))
        cost = 0
        for i in range(N):
            xi = x_data[i]
            yi = y_data[i]
            error = yi - np.exp(ae*xi*xi+be*xi+ce)
            J=np.zeros((3,1))
            J[0]=-xi*xi*np.exp(ae*xi*xi+be*xi+ce)
            J[1]=-xi*np.exp(ae*xi*xi+be*xi+ce)
            J[2]=-np.exp(ae*xi*xi+be*xi+ce)

            H=H+inv_sigma*inv_sigma*J*J.transpose()
            b =b+(-inv_sigma*inv_sigma*error*J)

            cost = cost+error*error

        dx = np.linalg.solve(H, b)

        if np.isnan(dx[0]):
            print("result is nan!")

        if (iter >0 and cost >= lastCost):
            print("cost: %f"  % (cost))

        ae = ae+dx[0]
        be = be+dx[1]
        ce = ce+dx[2]

        lastCost =cost

        print("ae:",str(ae),"be:",str(be),"ce:",str(ce))


    t2 =time.time()
    used_time =t2-t1
    print('using time:'+str(used_time))

if __name__ == "__main__":
    main()
            
 
    