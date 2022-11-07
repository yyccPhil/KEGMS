import cv2
import numpy as np
import os
from scipy.spatial.distance import pdist
import glob
from natsort import natsorted
from PIL import Image, ImageStat
import shutil
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from matplotlib import pyplot as plt

np.seterr(divide='ignore', invalid='ignore')
def diffimage(src_1,src_2):
    X = np.vstack([src_1, src_2])
    d2 = pdist(X)
    return d2
def convertImage(src):
    color_histgrams2 = [cv2.calcHist([src], [c], None, [16], [0, 256]) \
                        for c in range(3)]
    color_histgrams = np.array([chist / float(sum(chist)) for chist in color_histgrams2])
    hist = color_histgrams.flatten()
    return hist
def barMinus(src_1,src_2):
    src_1 = convertImage(src_1)
    src_2 = convertImage(src_2)
    return np.sum(diffimage(src_1,src_2))
def crop(src):
    box_list = []
    for i in range(int(h / cols)):
        for j in range(int(w / rows)):
            box = src[j * rows:(j + 1) * rows, i * cols:(i + 1) * cols]
            box_list.append(box)
    return box_list
def barMinus_crop(src_1,src_2):
    b=0
    for i in range(0,a):
        src_11 = convertImage(src_1[i])
        src_22 = convertImage(src_2[i])
        if np.sum(diffimage(src_11,src_22))<=threshold:
            b+=1
    return b


def hierarchy_cluster(data, method='average', threshold=35.0):
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
    plt.xlabel("serial number of global optical flow image   number of frame", fontsize=8)
    # 设置y轴的文本，用于描述y轴代表的是什么
    plt.ylabel("distance sum of histograms", fontsize=28)
    plt.title('Process of Hierarchical Clustering', fontsize=48)  # 设置图表标题
    plt.tick_params(labelsize=25)

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


cap = cv2.VideoCapture('Untitled1.wmv')    # 将这个改为手动选择文件
ret, lastframe3 = cap.read()
ret, lastframe2 = cap.read()
ret, lastframe1 = cap.read()
ret, lastframe = cap.read()
ret, frame = cap.read()
ret, frame1 = cap.read()
ret, frame2 = cap.read()
o = 2
o1 = 0
size = frame.shape
h = size[0]
w = size[1]
cols = 50
rows = 50
a = int(h / cols) * int(w / rows)
threshold = 0.15

dir = 'E:/project/capstone/data/origin_img/test'
os.makedirs(os.path.join(dir, 'nfl', str(o1)));   # 在这儿更改保存位置
save_dir = os.path.join(dir, 'nfl')     # 在这儿更改保存位置
# delete_dir = os.path.join(dir, 'delete_test')
# isExists = os.path.exists(delete_dir)
# if not isExists:
#     os.makedirs(os.path.join(dir, 'delete_test'));

while(cap.isOpened()):
    ret, frame3 = cap.read()
    o+=1
    if not ret:
        break
    if barMinus(lastframe, frame)>= threshold:
        if  barMinus(lastframe3, frame3)>= 0.16:
            # img = cv2.addWeighted(lastframe,0.5,frame1,0.5,0)
            # if barMinus(frame, img)<= 0.3:
            #    cv2.imwrite(str(o) + " 1" + ".jpg", frame)
            # elif barMinus(lastframe, frame)>= 0.4:
            #    cv2.imwrite(str(o) + " 2" + ".jpg", frame)
            # else:
            #    cv2.imwrite(str(o) + " 3" + ".jpg", frame)
            img1 = crop(lastframe)
            img2 = crop(frame)
            if barMinus_crop(img1, img2)<=30:
                # cv2.imwrite('keyframe' + '//' + str(o) + ".jpg", lastframe)
                if barMinus(frame, frame1)<= threshold:
                    o1+=1
                    os.mkdir(save_dir + '//' + str(o1));
            else:
                cv2.imwrite(save_dir + '//' + str(o1) + '//' + str(o) + ".jpg", lastframe)
        else:
            cv2.imwrite(save_dir + '//' + str(o1) + '//' + str(o) + ".jpg", lastframe)
    else:
        cv2.imwrite(save_dir + '//' + str(o1) + '//' + str(o) + ".jpg", lastframe)
    lastframe3 = lastframe2
    lastframe2 = lastframe1
    lastframe1 = lastframe
    lastframe = frame
    frame = frame1
    frame1 = frame2
    frame2 = frame3
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print(o1)

cap.release()
cv2.destroyAllWindows()

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

d_img_num = {}
img_num_list = []
for i in max_ind:
    img_list = natsorted(glob.glob(save_dir + '//' + '{}'.format(i) + '/*.jpg'))
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
    if a1 > 15 or b1 > 15 or c1 > 15:
        path = save_dir + '//' + '{}'.format(time)
        shutil.rmtree(path)

# 再优化的话我想到的办法就是转换成HSV颜色空间，先转换图像的空间，再计算均值比对，让一些恰巧在阈值内的图片与比赛图片区分开
