"""
Use the least squares fitting method to estimate the global motion, and eliminate the interference of local motion through continuous iteration
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import cv2
import glob
from natsort import natsorted
import pickle
from scipy import optimize
import random
import time

import xlwt
import shutil
import math

# hyper parameter of the threshold function, the bigger the hyper the less point
# (less points will be considered as local motion)
hyper = 1.0

def fitting_func(x, a, b, c):
    return a*x**2 + b*x + c


def fitting_func2(x, a, b):
    return a*x + b


def LeastSquareFit(flo_data, index_list):
    v_y = index_list
    v_z = flo_data
    fita_list = []

    fita, fitb = optimize.curve_fit(fitting_func, np.array(v_y), np.array(v_z))
    fita_list.append(fita)
    fita_avg = np.mean(np.array(fita_list), axis=0)
    return fita_avg


def aux(data):
    i, lst = data[0], data[1]
    if i not in lst:
        return i[1]


def return_path(base_path, depth, index_list):
    sub_path = base_path
    for i in range(depth):
        file_list = sorted(os.listdir(sub_path))
        sub_path = os.path.join(sub_path, file_list[index_list[i]])

    return sub_path


def param_estimate_x(flo_data, index_list, outlier_mat, iteration, thres):
    outline_index_list = []
    x_mat = None
    thres_auto = 0.0
    for iter in range(iteration):
        # x_axis_list (x-coordinate values of all points)
        # data (the motion value corresponding to each point, which has been normalized to [0, 255])
        x_axis_list = [i[1] for i in index_list if i not in outline_index_list]
        data = [flo_data[i[0], i[1], 0] for i in index_list if i not in outline_index_list]

        # fita_avg (the fitted parameter values)
        fita_avg = LeastSquareFit(data, x_axis_list)

        data_new = np.arange(0, len(flo_data[0]))
        data_com = fita_avg[0] * data_new.reshape(1, -1) ** 2 + fita_avg[1] * data_new.reshape(1, -1) + fita_avg[2]

        if iter == 0:
            thres_auto = abs(data_com[0][-1] - data_com[0][0]) * hyper
        # print('x channel iter:{}, auto_thres:{}'.format(iter, thres_auto))

        x_mat = np.repeat(data_com, flo_data.shape[0], axis=0)
        m_diff = np.abs(flo_data[:, :, 0] - x_mat)

        # mask (a matrix with the same size as the original optical flow field, and all of the values are true or false. Used to indicate which are outliers)
        mask = np.where(m_diff > thres_auto, True, False)

        outlier_mat = outlier_mat | mask
        outline_index_list = np.argwhere(outlier_mat).tolist()

        index_set = set(index_list)
        index_set.difference_update(set(map(tuple, outline_index_list)))
        index_list = list(index_set)

    return x_mat, outlier_mat, thres_auto


def param_estimate_y(flo_data, index_list, outlier_mat, iteration, thres):
    outline_index_list = []
    y_mat = None
    thres_auto = 0.0
    for iter in range(iteration):
        # y_axis_list
        # data
        y_axis_list = [i[0] for i in index_list if i not in outline_index_list]
        data = [flo_data[i[0], i[1], 1] for i in index_list if i not in outline_index_list]

        # fita_avg
        fita_avg = LeastSquareFit(data, y_axis_list)

        data_new = np.arange(0, len(flo_data))
        data_com = fita_avg[0]*data_new.reshape(-1, 1)**2 + fita_avg[1]*data_new.reshape(-1, 1) + fita_avg[2]

        if iter == 0:
            thres_auto = abs(data_com[0][0] - data_com[-1][0]) * hyper

        # print('y channel iter:{}, auto_thres:{}'.format(iter, thres_auto))

        y_mat = np.repeat(data_com, flo_data.shape[1], axis=1)

        m_diff = np.abs(flo_data[:, :, 1] - y_mat)

        # mask
        mask = np.where(m_diff > thres_auto, True, False)
        outlier_mat = outlier_mat | mask
        outline_index_list = np.argwhere(outlier_mat).tolist()
        index_set = set(index_list)
        index_set.difference_update(set(map(tuple, outline_index_list)))
        index_list = list(index_set)
    return y_mat, outlier_mat, thres_auto


def global_motion_estimation(flo_data, w=490, h=360):

    # mask = np.where(np.abs(flo_data) < 0.8, 0, 1)
    # mask = mask[:, :, 0] & mask[:, :, 1]
    # mask = np.expand_dims(mask, axis=2)
    # mask = np.repeat(mask, 2, axis=2)

    displace = 15
    iteration = 1
    thres = (np.max(flo_data) - np.min(flo_data)) * 0.08
    # print(thres)

    u = flo_data[:, :, 0]
    v = flo_data[:, :, 1]
    rad = np.sqrt(u ** 2 + v ** 2)  # the direction of optical flow field
    maxrad = max(-1, np.max(rad))

    flo_data = np.clip(flo_data, -displace, displace)
    index_list = [(i, j) for i in range(0, flo_data.shape[0]) for j in range(0, flo_data.shape[1])]
    outlier_mat = np.full((flo_data.shape[0], flo_data.shape[1]), False)

    x_ch, outlier_x, thres_auto_x = param_estimate_x(flo_data, index_list, outlier_mat, iteration=iteration, thres=thres)
    y_ch, outlier_y, thres_auto_y = param_estimate_y(flo_data, index_list, outlier_mat, iteration=iteration, thres=thres)

    outlier_merge = np.expand_dims(outlier_x | outlier_y, axis=2)
    outlier_merge = np.repeat(outlier_merge, 2, axis=2)
    outlier_merge = cv2.resize(outlier_merge.astype(np.float32), dsize=(w, h), interpolation=cv2.INTER_LINEAR)

    flo_x = x_ch[:, :, np.newaxis]
    flo_y = y_ch[:, :, np.newaxis]
    flo_global = np.concatenate((flo_x, flo_y), axis=2)

    flo_global = cv2.resize(flo_global, dsize=(w, h), interpolation=cv2.INTER_LINEAR)

    # flo_local = flo_data - flo_global
    #
    # flo_global_color = fl2.flow_to_image(flo_global, maxrad)
    # flo_local_color = fl2.flow_to_image(flo_local, maxrad)

    return flo_global, maxrad, outlier_merge


if __name__ == '__main__':

    base_dir = '../data/flo_data/flo_nfl'       # "mixed" optical flow
    global_pic_dir = '../data/flo_img'          # global optical flow

    key_global_dir = '../data/key_data/key_global_dir'
    save_dir = '../data/key_data/key_global_3'
    # key_global_middle_dir = '../data/key_data/key_global_middle'

    game_list = os.listdir(base_dir)
    for game in game_list:
        pic_list = natsorted(os.listdir(os.path.join(base_dir, game)))
        time_start = time.time()  # Timing begins
        for pic in pic_list:
            print(pic)
            flo_list = natsorted(glob.glob(os.path.join(base_dir, game, pic) + '/*.npy'))
            flo_num = len(flo_list)
            end = flo_num - 1

            d_x = {}
            d_y = {}
            d_x2 = {}       # Used to extract "steal" frames

            first_count = int(os.path.basename(flo_list[0]).replace(".npy", "")) + 1    # used to take the local minimum value
            last_count = int(os.path.basename(flo_list[-1]).replace(".npy", ""))  # the frame number of the last picture

            save_count = 1

            for index in flo_list:

                # global_motion
                data_org = np.load(index).astype(np.float32)

                resize_w = 32
                resize_h = 24

                data = cv2.resize(data_org, dsize=(resize_w, resize_h), interpolation=cv2.INTER_LINEAR)
                global_motion, maxrad, outlier_set = global_motion_estimation(data, w=data_org.shape[1],
                                                                              h=data_org.shape[0])

                x_global_motion = global_motion[0:1, :, 0]
                y_global_motion = global_motion[:, 0:1, 1]

                x1 = (x_global_motion[0, -1] - x_global_motion[0, 0]) / 2
                x2 = x_global_motion[0, -1] - x1
                x3 = abs(abs(x_global_motion[0, -1]) - abs(x1))
                x4 = abs(x2)

                y1 = (y_global_motion[-1, 0] - y_global_motion[0, 0]) / 2
                y2 = y_global_motion[-1, 0] - y1

                # print("zooming vector: (%f, %f)" % (x1, y1))
                # print("translation vector: (%f, %f)" % (x2, y2))

                if (x1 * y1) <= 0:
                    a = 0
                    b = math.sqrt(x2 * x2 + y2 * y2)
                else:
                    a = math.sqrt(x1 * x1 + y1 * y1)
                    b = math.sqrt(x2 * x2 + y2 * y2)

                flo_count = int(os.path.basename(index).replace(".npy", ""))

                d_x['{}'.format(flo_count)] = a
                d_y['{}'.format(flo_count)] = b
                d_x2['{}'.format(flo_count)] = x2
                # print(a, b, '\n')

                save_count += 1

            time_end = time.time()
            print('Time cost = %f s , %f' % (time_end - time_start, save_count))
            print(first_count, last_count)
            print('Extraction begins')

            key_global_path = os.path.join(key_global_dir, game, pic)
            isExists_global = os.path.exists(key_global_path)
            if not isExists_global:
                os.makedirs(key_global_path)

            global_img_list = natsorted(glob.glob(os.path.join(global_pic_dir, game, pic) + '/*.png'))

            # Extract the first and last frames, which is equivalent to extracting the first and last frames after the shots are segmented, where there is only 1 picture in the folder
            path_first = global_img_list[0]
            shutil.copy(path_first, key_global_path)

            path_last = global_img_list[-1]
            shutil.copy(path_last, key_global_path)

            # key_global_path_middle = os.path.join(key_global_middle_dir, game, pic)
            # isExists_middle = os.path.exists(key_global_path_middle)
            # if not isExists_middle:
            #     os.makedirs(key_global_path_middle)

            extre_list = [first_count - 1]
            extre_count = 1

            delta_middle = 53

            for extre in range(first_count, last_count):
                a0 = d_x['{}'.format(extre - 1)]
                a1 = d_x['{}'.format(extre)]
                a2 = d_x['{}'.format(extre + 1)]
                b0 = d_y['{}'.format(extre - 1)]
                b1 = d_y['{}'.format(extre)]
                b2 = d_y['{}'.format(extre + 1)]

                c = a1 > a0 and a1 > a2
                d = b1 > b0 and b1 > b2

                if c and d:
                    path_1 = os.path.join(global_pic_dir, game, pic) + '/global_{}.png'.format(extre)
                    shutil.copy(path_1, key_global_path)

                    extre_list.append(extre)
                    extre_count += 1

                    key0 = extre_list[extre_count - 2]
                    key1 = extre_list[extre_count - 1]
                    del_key = key1 - key0
                    if del_key >= delta_middle:  # When the difference between the numbers of the front and back keyframes is >= delta_middle, supplement frame
                        path_del_1 = os.path.join(global_pic_dir, game, pic) + '/global_{}.png'.format(
                            key0 + del_key // 2 + 1)
                        shutil.copy(path_del_1, key_global_path)
                        # shutil.copy(path_del_1, key_global_path_middle)

                        print('{}'.format(key0 + del_key // 2 + 1))

            extre_list.append(last_count)
            del_key = extre_list[-1] - extre_list[-2]
            if del_key >= delta_middle:  # >= delta_middle, supplement frame
                path_del_1 = os.path.join(global_pic_dir, game, pic) + '/global_{}.png'.format(
                    extre_list[-2] + del_key // 2 + 1)
                shutil.copy(path_del_1, key_global_path)
                # shutil.copy(path_del_1, key_global_path_middle)

                print('{}'.format(extre_list[-2] + del_key // 2 + 1))

            img_list = natsorted(glob.glob(os.path.join(key_global_dir, game, pic) + '/*.png'))
            img_num = len(img_list)
            end = img_num - 1

            save_count = 0  # folder number
            key_list = [0]

            for index in range(0, end, 1):

                img1_path = img_list[index]
                filename_1 = os.path.basename(img1_path).replace("global_", "")
                d1 = d_x2['{}'.format(int(filename_1.replace(".png", "")))]

                img2_path = img_list[index + 1]
                filename_2 = os.path.basename(img2_path).replace("global_", "")
                d2 = d_x2['{}'.format(int(filename_2.replace(".png", "")))]

                if d1 * d2 < 0:
                    key_list.append(index)
                    img2_name = os.path.basename(img2_path)
                    print((img2_name.replace(".png", "")), end='    ')

                    save_path = os.path.join(save_dir, game, pic, '{}'.format(save_count))
                    isExists_0 = os.path.exists(save_path)
                    if not isExists_0:
                        os.makedirs(save_path)

                    for i in range(key_list[save_count], key_list[save_count + 1] + 1):
                        path = img_list[i]
                        shutil.move(path, save_path)

                    del key_list[-1]
                    key_list.append(index + 1)
                    save_count += 1

                if index == end - 1:
                    path_new = os.path.join(key_global_dir, game,
                                            '{}_'.format(save_count))
                    os.rename(os.path.join(key_global_dir, game, pic), path_new)
                    shutil.move(path_new, os.path.join(save_dir, game, pic))

            if save_count == 0:
                mis_list = natsorted(glob.glob(os.path.join(save_dir, game, pic) + '/*.png'))
                mis_path = os.path.join(save_dir, game, pic, '0')
                os.makedirs(mis_path)
                for i in mis_list:
                    shutil.move(i, mis_path)
                print('unsegmented folder: {}'.format(pic))

        os.rmdir(os.path.join(key_global_dir, game))  # can only remove empty directories
    os.rmdir(key_global_dir)

    # base_dir = '../key_data/key_global_3'
    #
    # m = 3
    #
    # game_list = os.listdir(base_dir)
    # for game in game_list:
    #     pic_list = natsorted(os.listdir(os.path.join(base_dir, game)))
    #     for pic in pic_list:
    #         flo_list = natsorted(os.listdir(os.path.join(base_dir, game, pic)))
    #         for event in flo_list:
    #             img_list = natsorted(glob.glob(os.path.join(base_dir, game, pic, event) + '/*.png'))
    #             img_num = len(img_list)
    #             if img_num <= m:
    #                 path_del_1 = os.path.join(base_dir, game, pic, event)
    #                 shutil.rmtree(path_del_1)
