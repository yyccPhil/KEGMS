# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
import glob
from natsort import natsorted
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from matplotlib import pyplot as plt
import shutil


def hierarchy_cluster(data, method='average', threshold=40000.0):
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


def rgb2hsi(img_rgb):
    rows = int(img_rgb.shape[0])
    cols = int(img_rgb.shape[1])
    B, G, R = cv2.split(img_rgb)
    # 归一化到[0,1]
    B = B / 255.0
    G = G / 255.0
    R = R / 255.0
    img_hsi = img_rgb.copy()
    H, S, I = cv2.split(img_hsi)
    for i in range(rows):
        for j in range(cols):
            num = 0.5 * ((R[i, j] - G[i, j]) + (R[i, j] - B[i, j]))
            den = np.sqrt((R[i, j] - G[i, j]) ** 2 + (R[i, j] - B[i, j]) * (G[i, j] - B[i, j]))
            theta = float(np.arccos(num / den))

            if den == 0:
                H = 0
            elif B[i, j] <= G[i, j]:
                H = theta
            else:
                H = 2 * np.pi - theta

            min_RGB = min(min(B[i, j], G[i, j]), R[i, j])
            sum = B[i, j] + G[i, j] + R[i, j]
            if sum == 0:
                S = 0
            else:
                S = 1 - 3 * min_RGB / sum

            H = H / (2 * np.pi)
            I = sum / 3.0

            # H = H * 360

            # 为了便于理解，常常对结果做扩充，即 [0°，360°],[0,100],[0,255]
            img_hsi[i, j, 0] = H * 360
            img_hsi[i, j, 1] = S * 100
            img_hsi[i, j, 2] = I * 255

            # 或者为了便于计算直方图，都扩充为0~255（同RGB）
            # img_hsi[i, j, 0] = H * 255
            # img_hsi[i, j, 1] = S * 255
            # img_hsi[i, j, 2] = I * 255
    return img_hsi


class Hisogram(object):
    # def create_rgb_hist(self, image, color_type=1):
    #     """
    #     获取彩色空间直方图
    #     """
    #     h, w, c = image.shape
    #     rgHist = np.zeros([16 * 16 * 16, 1], np.float32)  # 必须是float型
    #     print(rgHist)
    #     hsize = 256 / 16
    #     for row in range(0, h, 1):
    #         for col in range(0, w, 1):
    #             b = image[row, col, 0]
    #             g = image[row, col, 1]
    #             r = image[row, col, 2]
    #             index = np.int(b / hsize) * 16 * 16 + np.int(g / hsize) * 16 + np.int(r / hsize)
    #             rgHist[np.int(index), 0] = rgHist[np.int(index), 0] + 1
    #     return rgHist


    def hist_compare(self, hist1, hist2):
        """
        比较两个直方图
        """
        # hist1 = self.create_rgb_hist(image1)
        # hist2 = self.create_rgb_hist(image2)
        match1 = cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)
        match2 = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        match3 = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CHISQR)
        match4 = cv2.compareHist(hist1, hist2, cv2.HISTCMP_INTERSECT)
        # print("巴氏距离：%20s,   相关性：%20s,   卡方：%s,  HISTCMP_INTERSECT：%s " % (match1, match2, match3, match4))

        return match1, match2, match3, match4


    # def hist_image(self, image):
    #     color = ("Hue", "Saturity", "Intensity")
    #     for i, color in enumerate(color):
    #         hist = cv2.calcHist([image], [i], None, [256], [0, 256])  # 计算rgb的直方图
    #     #            hist = cv.calcHist([image], [0,1], None, [180,256], [0,180,0,256])  #计算H-S直方图
    #     print(hist)


if __name__ == '__main__':

    # origin_dir = 'E:/project/data/origin_img'
    # base_dir = 'E:/project/data/key_data/key_global_3'
    # new_dir = 'E:/project/data/key_data/key_global_new'
    # key_dir = 'E:/project/data/key_data/key'
    time = 'E:/project/capstone/data/13'
    # time = 'E:/project/capstone/main/final/pytorch-pwc-final/data/key_data/nba/key_global/nba/30/27'

    img_list = natsorted(glob.glob(time + '/*.png'))

    if len(img_list) > 2:
        arr = []
        for img in img_list:
            img_rgb1 = cv2.imread(img)
            # img_hsi1 = rgb2hsi(img_rgb1)
            hist_h1 = cv2.calcHist([img_rgb1], [0], None, [256], [0, 255])
            hist_h2 = cv2.calcHist([img_rgb1], [1], None, [256], [0, 255])
            hist_h3 = cv2.calcHist([img_rgb1], [2], None, [256], [0, 255])
            hist = np.vstack((hist_h1, hist_h2, hist_h3))
            arr.append(hist)

        arr = np.array(arr)
        arr = np.squeeze(arr)

        # 把二维矩阵变成对角矩阵
        c, r = arr.shape
        for i in range(r):
            for j in range(i, c):
                if arr[i][j] != arr[j][i]:
                    arr[i][j] = arr[j][i]
        for i in range(r):
            for j in range(i, c):
                if arr[i][j] != arr[j][i]:
                    print(arr[i][j], arr[j][i])

        num_clusters, indices = hierarchy_cluster(arr)

        print("%d clusters" % num_clusters)
        for k, ind in enumerate(indices):
            print("cluster", k + 1, "is", ind)
        print(indices[0])
