# adjust

import time
import numpy as np
import os
from scipy.spatial.distance import pdist
import glob
from natsort import natsorted
from PIL import Image, ImageStat
import shutil
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from matplotlib import pyplot as plt

# 35
def hierarchy_cluster(data, method='average', threshold=68.0):
    '''层次聚类

    Arguments:
        data [[0, float, ...], [float, 0, ...]] -- 文档 i 和文档 j 的距离

    Keyword Arguments:
        method {str} -- [linkage的方式： single、complete、average、centroid、median、ward] (default: {'average'})
        threshold {float} -- 聚类簇之间的距离
    Return:
        cluster_number int -- 聚类个数
        cluster [[idx1, idx2,..], [idx3]] -- 每一类下的索引
    '''
    data = np.array(data)

    Z = linkage(data, method=method)
    cluster_assignments = fcluster(Z, threshold, criterion='distance')
    # print(type(cluster_assignments))
    num_clusters = cluster_assignments.max()
    indices = get_cluster_indices(cluster_assignments)

    dendrogram(Z)
    plt.xlabel("serial number of global optical flow image   number of frame", fontsize=28)
    # 设置y轴的文本，用于描述y轴代表的是什么
    plt.ylabel("distance sum of histograms", fontsize=28)
    plt.title('Process of Hierarchical Clustering', fontsize=48)  # 设置图表标题
    plt.tick_params(labelsize=8)

    plt.show()

    return num_clusters, indices


def get_cluster_indices(cluster_assignments):
    '''映射每一类至原数据索引

    Arguments:
        cluster_assignments 层次聚类后的结果

    Returns:
        [[idx1, idx2,..], [idx3]] -- 每一类下的索引
    '''
    n = cluster_assignments.max()
    indices = []
    for cluster_number in range(1, n + 1):
        indices.append(np.where(cluster_assignments == cluster_number)[0])

    return indices


time_start = time.time()

dir = '../data/origin_img/test'
save_dir = os.path.join(dir, 'hockey')     # 在这儿更改保存位置

event_list = natsorted(os.listdir(save_dir))
o1 = int(os.path.basename(event_list[-1]))
print(o1)

dx_b = {}
dx_g = {}
dx_r = {}
d_hist0 = {}
d_hist1 = {}
d_hist2 = {}
arr = []
for i in range(0, o1+1):
    img_list = natsorted(glob.glob(save_dir + '//' + '{}'.format(i) + '/*.jpg'))
    img_num = len(img_list)
    hist0 = []
    hist1 = []
    hist2 = []
    for index in range(img_num):
        img_path = img_list[index]
        im = Image.open(img_path)
        stat = ImageStat.Stat(im)
        hist0.append(stat.mean[0])
        hist1.append(stat.mean[1])
        hist2.append(stat.mean[2])
    dx_b['{}'.format(i)] = max(hist0) - min(hist0)
    dx_g['{}'.format(i)] = max(hist1) - min(hist1)
    dx_r['{}'.format(i)] = max(hist2) - min(hist2)

    average_hist0 = np.mean(hist0)
    average_hist1 = np.mean(hist1)
    average_hist2 = np.mean(hist2)
    hist = np.vstack((average_hist0, average_hist1, average_hist2))
    arr.append(hist)

    d_hist0['{}'.format(i)] = average_hist0
    d_hist1['{}'.format(i)] = average_hist1
    d_hist2['{}'.format(i)] = average_hist2

arr = np.array(arr)
arr = np.squeeze(arr)

# 把二维矩阵变成对角矩阵
# c, r = arr.shape
# for i in range(r):
#     for j in range(i, c):
#         if arr[i][j] != arr[j][i]:
#             arr[i][j] = arr[j][i]
# for i in range(r):
#     for j in range(i, c):
#         if arr[i][j] != arr[j][i]:
#             print(arr[i][j], arr[j][i])

num_clusters, indices = hierarchy_cluster(arr)

print("%d clusters" % num_clusters)
for k, ind in enumerate(indices):
    print("cluster", k + 1, "is", ind)

ind_img_num = {}
ind_num_list = []

for ind in enumerate(indices):
    ind_num_list.append(len(ind))
    ind_img_num['{}'.format(len(ind))] = ind
max_ind = ind_img_num['{}'.format(max(ind_num_list))]

print(type(max_ind[1]))
print(max_ind[1])

d_img_num = {}
img_num_list = []
for i in max_ind[1]:
    img_list = natsorted(glob.glob(save_dir + '//' + '{}'.format(i) + '/*.jpg'))
    print(type(i))
    print(i)
    img_num = len(img_list)
    if dx_b['{}'.format(i)] > 100 and dx_g['{}'.format(i)] > 100 and dx_r['{}'.format(i)] > 100:      # 防止切割错误导致文件数量过大被认为是最大数量文件夹
        print('分割错误数量过大：{}'.format(i))
    else:
        img_num_list.append(img_num)
        d_img_num['{}'.format(img_num)] = i

m = max(img_num_list)
n = d_img_num['{}'.format(m)]
print(n, m)

# first_end = [0, o1]       # 单独计算首尾文件夹RGB均值
# for i in first_end:
#     img_list = natsorted(glob.glob(save_dir + '//' + '{}'.format(i) + '/*.jpg'))
#     img_num = len(img_list)
#     hist0 = []
#     hist1 = []
#     hist2 = []
#     for index in range(0, img_num):
#         img_path = img_list[index]
#         im = Image.open(img_path)
#         stat = ImageStat.Stat(im)
#         hist0.append(stat.mean[0])
#         hist1.append(stat.mean[1])
#         hist2.append(stat.mean[2])
#     average_hist0 = np.mean(hist0)
#     average_hist1 = np.mean(hist1)
#     average_hist2 = np.mean(hist2)
#
#     d_hist0['{}'.format(i)] = average_hist0
#     d_hist1['{}'.format(i)] = average_hist1
#     d_hist2['{}'.format(i)] = average_hist2
#
average0 = d_hist0['{}'.format(n)]
average1 = d_hist1['{}'.format(n)]
average2 = d_hist2['{}'.format(n)]

time_list = natsorted(os.listdir(save_dir))
for time in time_list:
    a = d_hist0['{}'.format(time)]
    b = d_hist1['{}'.format(time)]
    c = d_hist2['{}'.format(time)]
    a1 = 100 * abs(a - average0) / a
    b1 = 100 * abs(b - average1) / b
    c1 = 100 * abs(c - average2) / c
    print(time, a1, b1, c1)
    if a1 > 20 or b1 > 15 or c1 > 15:
        path = save_dir + '//' + '{}'.format(time)
        shutil.rmtree(path)

time_end = time.time()
print('Time cost = %fs' % (time_end - time_start))
