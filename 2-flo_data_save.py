"""
Use the least squares fitting method to estimate the global motion, and eliminate the interference of local motion through continuous iteration
"""

from lib import flowlib_v2 as fl2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import cv2
from lib import flowlib as fl
import glob
from natsort import natsorted
import pickle
from scipy import optimize
import random
import time
import math
import shutil

import run_many as run

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
        print('x channel iter:{}, auto_thres:{}'.format(iter, thres_auto))

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
        # y_axis_list (y-coordinate values of all points)
        # data (the motion value corresponding to each point, which has been normalized to [0, 255])
        y_axis_list = [i[0] for i in index_list if i not in outline_index_list]
        data = [flo_data[i[0], i[1], 1] for i in index_list if i not in outline_index_list]

        # fita_avg (the fitted parameter values)
        fita_avg = LeastSquareFit(data, y_axis_list)

        data_new = np.arange(0, len(flo_data))
        data_com = fita_avg[0]*data_new.reshape(-1, 1)**2 + fita_avg[1]*data_new.reshape(-1, 1) + fita_avg[2]

        if iter == 0:
            thres_auto = abs(data_com[0][0] - data_com[-1][0]) * hyper

        print('y channel iter:{}, auto_thres:{}'.format(iter, thres_auto))

        y_mat = np.repeat(data_com, flo_data.shape[1], axis=1)

        m_diff = np.abs(flo_data[:, :, 1] - y_mat)

        # mask (a matrix with the same size as the original optical flow field, and all of the values are true or false. Used to indicate which are outliers)
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
    print(thres)

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

    return flo_global, maxrad, outlier_merge, thres, thres_auto_x, thres_auto_y


if __name__ == '__main__':

    base_dir = '../data/origin_img'
    save_dir = '../data/flo_data'
    global_pic_dir = '../data/global_img'

    game_list = os.listdir(base_dir)
    for game in game_list:
        event_list = natsorted(os.listdir(os.path.join(base_dir, game)))
        for event in event_list:
            print(event)
            real_list = natsorted(glob.glob(os.path.join(base_dir, game, event) + '/*.jpg'))

            # used for frame supplementary (may be mistakes in the shot segmentation)
            first_filename = os.path.basename(real_list[0])
            flo_count = int(first_filename.replace(".jpg", "")) + 1
            last_filename = os.path.basename(real_list[-1])
            last_count = int(last_filename.replace(".jpg", ""))  # record the number of the last frame
            end = last_count - flo_count + 1  # because flo_count + 1

            for index in range(0, end, 1):

                img1_path = real_list[index]
                img2_path = real_list[index + 1]

                 # used for frame supplementary
                test_path = os.path.join(base_dir, game, event)
                test_count = int(os.path.basename(img2_path).replace(".jpg", ""))
                test_index = index
                test_pic = flo_count
                while test_count != flo_count:
                    shutil.copyfile(img2_path, test_path + '/{}.jpg'.format(test_pic))
                    real_list.insert(test_index + 1, test_path + '/{}.jpg'.format(test_pic))
                    img2_path = real_list[test_index + 1]
                    print('lost frame: %d' % test_pic)
                    test_count = int(os.path.basename(img2_path).replace(".jpg", ""))
                    test_index += 1
                    test_pic += 1

                flo_count += 1

            img_list = real_list[::1]
            img_num = len(img_list)
            # if img_num % 2 == 0:
            #     end = img_num - 2
            # else:
            #     end = img_num - 1
            end = img_num - 1

            save_count = 1

            for index in range(0, end, 1):

                img1_path = img_list[index]
                img2_path = img_list[index + 1]

                flo_data = run.generate_flow_data(img1_path, img2_path)

                maxrad_1, minu_1, maxu_1, minv_1, maxv_1 = fl.flow_to_image(flo_data)

                flo_count = int(os.path.basename(img2_path).replace(".jpg", ""))

                save_path = os.path.join(save_dir, game, event)
                isExists = os.path.exists(save_path)
                if not isExists:
                    os.makedirs(save_path)

                # save orginal 2-channel flow data
                np.save(save_path + '/{}'.format(flo_count), flo_data)

                # global_motion
                data_org = np.load(save_path + '/{}.npy'.format(flo_count)).astype(np.float32)

                resize_w = 32
                resize_h = 24

                data = cv2.resize(data_org, dsize=(resize_w, resize_h), interpolation=cv2.INTER_LINEAR)
                global_motion, maxrad, outlier_set , thres, thres_auto_x, thres_auto_y = global_motion_estimation(data, w=data_org.shape[1], h=data_org.shape[0])

                x_global_motion = global_motion[0:1, :, 0]
                y_global_motion = global_motion[:, 0:1, 1]

                x1 = (x_global_motion[0, -1] - x_global_motion[0, 0]) / 2
                x2 = x_global_motion[0, -1] - x1

                y1 = (y_global_motion[-1, 0] - y_global_motion[0, 0]) / 2
                y2 = y_global_motion[-1, 0] - y1

                x3 = math.sqrt(x1 * x1 + y1 * y1)
                y3 = math.sqrt(x2 * x2 + y2 * y2)

                print("zooming vector: (%f, %f)" % (x1, y1))
                print("translation vector: (%f, %f)" % (x2, y2))
                print("resultant zooming magnitude, resultant translation distance: (%f, %f)" % (x3, y3), '\n')

                # local_motion = data_org - global_motion
                #
                global_pic_save_path = os.path.join(global_pic_dir, game, event)
                isExists_1 = os.path.exists(global_pic_save_path)
                if not isExists_1:
                    os.makedirs(global_pic_save_path)

                flo_global_color = fl2.flow_to_image(global_motion, maxrad)
                img_out = Image.fromarray(flo_global_color)
                img_out.save(global_pic_save_path + '/global_{}.png'.format(flo_count))
                #
                # flo_local_color = fl2.flow_to_image(local_motion, maxrad)
                # img_out = Image.fromarray(flo_local_color)
                # img_out.save(save_path + '/local_{}.png'.format(flo_count))

                save_count += 1
