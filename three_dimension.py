import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

df = pd.read_excel("../1.xlsx")
Z = np.array(df)
xx = np.arange(3, 4.1, step=0.2)    # the coordinates of the X axis
yy = [0.2, 0.22, 0.24, 0.26, 0.28, 0.30, 0.32, 0.34, 0.36, 0.38, 0.40]         # the coordinates of the Y axis

print(xx)
print(yy)
print(Z)

X, Y = np.meshgrid(xx, yy)          # Grid coordinates
# # X, Y=xx.ravel(), yy.ravel()     # Matrix flattening
fig = plt.figure()
ax3 = plt.axes(projection='3d')

ax3.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='rainbow')
# ax3.contour(X, Y, Z, zdim='z', offset=0.079, cmap='rainbow')   # Contour map, assign the minimum value of Z to argument offset 

# Set axis
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

# bottom=np.zeros_like(X)   # Set the bottom level value of the histogram
# Z=Z.ravel()               # Matrix flattening
#
# width = 0.1       # Width and height of every column
# height = 0.01
#
# Drawing settings
# fig=plt.figure()
# ax=fig.gca(projection='3d')   # 3D axis
# ax.bar3d(X, Y, bottom, width, height, Z, shade=True)

# plt.show()
