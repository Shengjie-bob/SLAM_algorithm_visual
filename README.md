# V_SLAM_algorithm

1.仿真程序由python3.6编写 使用opencv3.4库函数、tensorflow1.8库函数、numpy和matplot

2.特征法2d-2d：2d_2d.py

3.特征法3d-2d：3d_2d.py

4.特征法3d-3d：3d_3d.py

5.直接法计算：direct_method.py

6.基于图优化的地图：graph_slam.py

7.深度学习的部分：/深度学习代码/Goo_train.py      -----训练GooLeNet
		/深度学习代码/Goo_test.py      -----测试GooLeNet
		/深度学习代码/vgg16_train.py      ----训练VGG
		/深度学习代码/vgg16_train.py      ----测试VGG
		其余代码为模型构建文件和读取文件脚本

8.画图部分：/画图文件/matplot_2d_2d.py     ------2d-2d的对极约束图
	/画图文件/matplot_3d_2d.py     ------3d-2d的坐标误差图
	/画图文件/matplot_3d_3d.py     ------3d-3d的坐标误差图
	/画图文件/matplot_posenet.py     ------深度学习GooleNet的文件
	/画图文件/matplot_vgg.py        ------深度学习VGG的文件
	/画图文件/matplot.py     ------其他图片
