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

def hierarchy_cluster(data, method='average', threshold=68.0):
    '''Hierarchical Clustering

    Arguments:
        data [[0, float, ...], [float, 0, ...]] -- the distance between i and j

    Keyword Arguments:
        method {str} -- [linkage methods: single、complete、average、centroid、median、ward] (default: {'average'})
        threshold {float} -- the distance between the clusters
    Return:
        cluster_number int
        cluster [[idx1, idx2,..], [idx3]] -- index of every cluster
    '''
    data = np.array(data)

    Z = linkage(data, method=method)
    cluster_assignments = fcluster(Z, threshold, criterion='distance')
    # print(type(cluster_assignments))
    num_clusters = cluster_assignments.max()
    indices = get_cluster_indices(cluster_assignments)

    dendrogram(Z)
    plt.xlabel("serial number of global optical flow image   number of frame", fontsize=28)
    plt.ylabel("distance sum of histograms", fontsize=28)
    plt.title('Process of Hierarchical Clustering', fontsize=48)
    plt.tick_params(labelsize=8)

    plt.show()

    return num_clusters, indices


def get_cluster_indices(cluster_assignments):
    '''Map each class to the original data index

    Arguments:
        cluster_assignments -- the result after hierarchical clustering

    Returns:
        [[idx1, idx2,..], [idx3]] -- index of every cluster
    '''
    n = cluster_assignments.max()
    indices = []
    for cluster_number in range(1, n + 1):
        indices.append(np.where(cluster_assignments == cluster_number)[0])

    return indices


time_start = time.time()

dir = '../data/origin_img/test'
save_dir = os.path.join(dir, 'hockey')     # video frame save path

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

# Convert two-dimensional matrix into diagonal matrix
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
    if dx_b['{}'.format(i)] > 100 and dx_g['{}'.format(i)] > 100 and dx_r['{}'.format(i)] > 100:      # Prevent the folder with excessive files caused by segmentation mistake to be considered as the folder with the maximum number of files
        print('segmentation mistake: {}'.format(i))
    else:
        img_num_list.append(img_num)
        d_img_num['{}'.format(img_num)] = i

m = max(img_num_list)
n = d_img_num['{}'.format(m)]
print(n, m)

# first_end = [0, o1]       # Separately calculate the RGB mean value of the first and last folder
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
