# adjust

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd  #导入pandas库

df = pd.read_excel("../1.xlsx")
Z = np.array(df)  #方法三
xx = np.arange(3, 4.1, step=0.2)    # X轴的坐标
yy = [0.2, 0.22, 0.24, 0.26, 0.28, 0.30, 0.32, 0.34, 0.36, 0.38, 0.40]         # Y轴的坐标

print(xx)
print(yy)
print(Z)

X, Y = np.meshgrid(xx, yy)#网格化坐标
# # X, Y=xx.ravel(), yy.ravel()#矩阵扁平化
#
fig = plt.figure()
ax3 = plt.axes(projection='3d')

ax3.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='rainbow')
# ax3.contour(X, Y, Z, zdim='z', offset=0.079, cmap='rainbow')   #等高线图，要设置offset，为Z的最小值

#坐标轴设置
new_ticks = np.linspace(0.20, 0.40, 11)
print(new_ticks)
plt.yticks(new_ticks)
ax3.set_ylim(0.4, 0.20)
font2 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size' : 20, }
plt.tick_params(axis='both', which='major', labelsize=16)
ax3.set_xlabel(' ', font2)
ax3.set_ylabel(' ', font2)
ax3.set_zlabel(' ', font2)
plt.show()

# bottom=np.zeros_like(X)#设置柱状图的底端位值
# Z=Z.ravel()#扁平化矩阵
#
# width = 0.1 #每一个柱子的长和宽
# height = 0.01
#
# #绘图设置
# fig=plt.figure()
# ax=fig.gca(projection='3d')#三维坐标轴
# ax.bar3d(X, Y, bottom, width, height, Z, shade=True)#

# plt.show()
