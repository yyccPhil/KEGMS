# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
import glob
from natsort import natsorted
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from matplotlib import pyplot as plt
import shutil


def hierarchy_cluster(data, method='average', threshold=90000.0):
    '''Hierarchical Clustering

    Arguments:
        data [[0, float, ...], [float, 0, ...]] -- the distance between i and j

    Keyword Arguments:
        method {str} -- [linkage methods： single、complete、average、centroid、median、ward] (default: {'average'})
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

    origin_dir = '../data/origin_img'
    base_dir = '../data/key_data/key_global_3'
    new_dir = '../data/key_data/key_global_new'
    key_dir = '../data/key_data/key'

    shutil.copytree(base_dir, '../data/key_data/key_global')

    game_list = os.listdir(base_dir)
    for game in game_list:
        event_list = natsorted(os.listdir(os.path.join(base_dir, game)))
        for event in event_list:
            # print(event)
            time_list = natsorted(os.listdir(os.path.join(base_dir, game, event)))
            for time in time_list:
                print(event, '--', time)
                img_list = natsorted(glob.glob(os.path.join(base_dir, game, event, time) + '/*.png'))
                img_num = len(img_list)
                end = img_num - 1

                front_list = []
                img2_path = ''
                new_path = os.path.join(new_dir, game, event, time)
                isExists_0 = os.path.exists(new_path)
                if not isExists_0:
                    os.makedirs(new_path)

                if img_num == 1:
                    shutil.copy(img_list[0], new_path)
                else:
                    for index in range(0, end):
                        img1_path = img_list[index]
                        img_rgb1 = cv2.imread(img1_path)
                        # img_hsi1 = rgb2hsi(img_rgb1)
                        hsv1 = cv2.cvtColor(img_rgb1, cv2.COLOR_RGB2HSV)
                        hist_h1 = cv2.calcHist([hsv1], [0], None, [360], [0, 359])

                        img2_path = img_list[index + 1]
                        img_rgb2 = cv2.imread(img2_path)
                        # img_hsi2 = rgb2hsi(img_rgb2)
                        hsv2 = cv2.cvtColor(img_rgb2, cv2.COLOR_RGB2HSV)
                        hist_h2 = cv2.calcHist([hsv2], [0], None, [360], [0, 359])

                        myHist = Hisogram()
                        match1, match2, match3, match4 = myHist.hist_compare(hist_h1, hist_h2)

                        front_list.append(img1_path)

                        if match1 > 0.17:  # Bhattacharyya distance, the threshold for extraction
                            # if match2 < 0.9:
                            if len(front_list) <= 2:
                                for front in front_list:
                                    shutil.copy(front, new_path)
                            elif 2 < len(front_list):
                                img1_name = os.path.basename(img1_path)
                                print((img1_name.replace(".png", "")))
                                shutil.copy(front_list[1], new_path)
                                shutil.copy(front_list[-1], new_path)
                            # elif len(front_list) >= 6:
                            #     img1_name = os.path.basename(img1_path)
                            #     print((img1_name.replace(".png", "")))
                            #     for front in front_list[1:-2]:
                            #         shutil.move(front, del_path)
                            front_list = []

                    front_list.append(img2_path)
                    if len(front_list) <= 2:
                        for front in front_list:
                            shutil.copy(front, new_path)
                    elif 2 < len(front_list):
                        shutil.copy(front_list[1], new_path)
                        shutil.copy(front_list[-1], new_path)

    game_list = os.listdir(base_dir)
    for game in game_list:
        event_list = natsorted(os.listdir(os.path.join(base_dir, game)))
        for event in event_list:
            # print(event)
            time_list = natsorted(os.listdir(os.path.join(base_dir, game, event)))
            for time in time_list:
                print(event, '--', time)
                img_list = natsorted(glob.glob(os.path.join(base_dir, game, event, time) + '/*.png'))

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

                new_list = natsorted(glob.glob(os.path.join(base_dir, game, event, time) + '/*.png'))
                new_path = os.path.join(new_dir, game, event, time)
                for new in new_list:
                    isExists_new = os.path.exists(new)
                    if not isExists_new:
                        shutil.move(new, new_path)

    shutil.rmtree(base_dir)
    base_dir = new_dir

    game_list = os.listdir(base_dir)
    for game in game_list:
        pic_list = natsorted(os.listdir(os.path.join(base_dir, game)))
        for pic in pic_list:
            flo_list = natsorted(os.listdir(os.path.join(base_dir, game, pic)))
            for index in range(len(flo_list) - 1):
                img_list = natsorted(glob.glob(os.path.join(base_dir, game, pic, flo_list[index]) + '/*.png'))
                next_path = os.path.join(base_dir, game, pic, flo_list[index + 1])
                shutil.move(img_list[-1], next_path)

    # game_list = os.listdir(base_dir)
    # for game in game_list:
    #     pic_list = natsorted(os.listdir(os.path.join(base_dir, game)))
    #     for pic in pic_list:
    #         flo_list = natsorted(os.listdir(os.path.join(base_dir, game, pic)))
    #         for event in flo_list:
    #             img_list = natsorted(glob.glob(os.path.join(base_dir, game, pic, event) + '/*.png'))
    #             img_num = len(img_list)
    #             if img_num <= 1:
    #                 print(os.path.join(base_dir, game, pic, event))
    #                 path_del_1 = os.path.join(base_dir, game, pic, event)
    #                 shutil.rmtree(path_del_1)

    game_list = os.listdir(base_dir)
    for game in game_list:
        pic_list = natsorted(os.listdir(os.path.join(base_dir, game)))
        for pic in pic_list:
            flo_list = natsorted(os.listdir(os.path.join(base_dir, game, pic)))
            for event in flo_list:
                img_list = natsorted(glob.glob(os.path.join(base_dir, game, pic, event) + '/*.png'))
                for img in img_list:
                    global_name = os.path.basename(img)
                    origin_name0 = global_name.replace("global_", "")
                    origin_name = origin_name0.replace("png", "jpg")
                    origin_path = os.path.join(origin_dir, game, pic, origin_name)

                    key_path = os.path.join(key_dir, game, pic, event)
                    isExists_key = os.path.exists(key_path)
                    if not isExists_key:
                        os.makedirs(key_path)

                    shutil.copy(origin_path, key_path)

    shutil.rmtree(new_dir)
    os.rename('../data/key_data/key_global', '../data/key_data/key_global_3')
