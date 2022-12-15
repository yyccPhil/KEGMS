# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
import glob
from natsort import natsorted
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from matplotlib import pyplot as plt
import shutil
import time


def hierarchy_cluster(data, method='average', threshold=6000.0):
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


def rgb2hsi(img_rgb):
    rows = int(img_rgb.shape[0])
    cols = int(img_rgb.shape[1])
    B, G, R = cv2.split(img_rgb)
    # Normalize to [0,1]
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

            # For ease of understanding, the results are often extended, that is, [0°, 360°], [0,100], [0,255]
            img_hsi[i, j, 0] = H * 360
            img_hsi[i, j, 1] = S * 100
            img_hsi[i, j, 2] = I * 255

            # Or in order to facilitate the calculation of the histogram, expanded to 0~255 (same as RGB)
            # img_hsi[i, j, 0] = H * 255
            # img_hsi[i, j, 1] = S * 255
            # img_hsi[i, j, 2] = I * 255
    return img_hsi


class Hisogram(object):
    # def create_rgb_hist(self, image, color_type=1):
    #     """
    #     Get color space histogram
    #     """
    #     h, w, c = image.shape
    #     rgHist = np.zeros([16 * 16 * 16, 1], np.float32)  # must be float type
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
        compare two histograms
        """
        # hist1 = self.create_rgb_hist(image1)
        # hist2 = self.create_rgb_hist(image2)
        match1 = cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)
        match2 = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        match3 = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CHISQR)
        match4 = cv2.compareHist(hist1, hist2, cv2.HISTCMP_INTERSECT)
        # print("Bhattacharyya distance：%20s,  Correlation：%20s,  Chi-square：%s,   HISTCMP_INTERSECT：%s " % (match1, match2, match3, match4))

        return match1, match2, match3, match4


    # def hist_image(self, image):
    #     color = ("Hue", "Saturity", "Intensity")
    #     for i, color in enumerate(color):
    #         hist = cv2.calcHist([image], [i], None, [256], [0, 256])  # Calculate the histogram of RGB
    #     #            hist = cv.calcHist([image], [0,1], None, [180,256], [0,180,0,256])  # Calculate the histogram of H-S
    #     print(hist)


if __name__ == '__main__':

    origin_dir = '../data/data_1/origin_img/nba'
    # shutil.copytree(origin_dir, 'E:/project/capstone/data/test/nba')
    base_dir = '../data/test'

    time_start = time.time()

    game_list = os.listdir(base_dir)
    for game in game_list:
        event_list = natsorted(os.listdir(os.path.join(base_dir, game)))
        for event in event_list:
            print(event)
            img_list = natsorted(glob.glob(os.path.join(base_dir, game, event) + '/*.jpg'))

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

                # print("%d clusters" % num_clusters)
                # for k, ind in enumerate(indices):
                #     print("cluster", k + 1, "is", ind)

                for ind in range(num_clusters):
                    if len(indices[ind]) > 2:
                        for front in indices[ind][1:-1]:
                            num = int(front)
                            os.unlink(img_list[num])
                    print("%d clusters" % num_clusters)
                    for k, ind in enumerate(indices):
                        print("cluster", k + 1, "is", ind)

    time_end = time.time()
    print('Time cost = %fs' % (time_end - time_start))
