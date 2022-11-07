"""
基于生成的光流图进行全局运动估计
1. 计算4个角点
2. 基于角点值近似整个光流场
3. 每一行的y方向幅值相同
4. 每一列的x方向幅值相同
"""
from lib import flowlib as fl
from lib import flowlib_v2 as fl2
import numpy as np
# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot,savefig
from PIL import Image
import os
import cv2
import natsort
import pickle


def outlier_filter(data):
    data = data.reshape(1, -1)
    data = list(data[0])
    sort_value = np.argsort(data)
    remain_index = list(sort_value[int(len(sort_value)*0.2):int(len(sort_value)*0.8)])
    remain_value = [data[index] for index in remain_index]
    mean = np.mean(remain_value)
    return mean


def global_motion_estimate(flow_path, visualize_flow=False):
    flow = fl2.read_flow(flow_path)

    if visualize_flow:
        fl.visualize_flow(flow)

    max_rad = fl2.compute_maxrad(flow)
    # # 每一列的均值
    # horizen_mean = np.mean(flow[5:-5, 5:-5, 0], axis=0)
    # # 每一行的均值
    # vertical_mean = np.mean(flow[2:-2, 2:-2, 1], axis=1)

    # # 第一列
    # first_column = flow[5:-5, 5:6, 0]
    # m = outlier_filter(first_column)
    # # 最后一列
    # last_column = flow[5:-5, -6:-5, 0]

    # # 第一列的均值
    # x1 = horizen_mean[0]
    # # 最后一列的均值
    # x2 = horizen_mean[-1]
    # # 最后一列的均值
    # y1 = vertical_mean[0]
    # # 最后一列的均值
    # y2 = vertical_mean[-1]

    # 第一列的均值
    x1 = float(outlier_filter(flow[5:-5, 5:6, 0]))
    # 最后一列的均值
    x2 = float(outlier_filter(flow[5:-5, -6:-5, 0]))
    # 第一行的均值
    y1 = float(outlier_filter(flow[5:6, 5:-5, 1]))
    # 最后一行的均值
    y2 = float(outlier_filter(flow[-6:-5, 5:-5, 1]))

    # 初始化一个和光流图一样size的空矩阵
    flow_global = np.zeros((flow.shape[0], flow.shape[1], 2))

    # 基于x1~xw 线性插值生成一行
    x_direction = np.linspace(x1, x2, flow.shape[1]).reshape(1, -1)
    # x_direction = np.arange(x1, x2, (x2 - x1) / flow.shape[1]).reshape(1, -1)[:, 0:flow.shape[1]]
    # 基于y1~yh 线性插值生成一列
    # y_direction = np.arange(y1, y2, (y2 - y1) / flow.shape[0]).reshape(-1, 1)[0:flow.shape[0], :]
    y_direction = np.linspace(y1, y2, flow.shape[0]).reshape(-1, 1)

    # x方向幅值行向量纵向复制h次
    x_mat = np.repeat(x_direction, flow.shape[0], axis=0)
    # y方向幅值列向量横向复制w次
    y_mat = np.repeat(y_direction, flow.shape[1], axis=1)
    flow_global[:, :, 0] = x_mat
    flow_global[:, :, 1] = y_mat

    return flow_global, max_rad


def color_coding(flow, rad_org):
    max_rad_glo = fl2.compute_maxrad(flow)
    if abs(rad_org - max_rad_glo) < 1:
        img_ = fl.flow_to_image(flow)
    else:
        img_ = fl2.flow_to_image(flow, rad_org)
    return img_


if __name__ == '__main__':
    # path = r'D:\flow_multi_mode\flow\0aWfrZAM6Q8\other-2-pointer-failure\2803221.154\13.flo'
    # global_motion, max_rad = global_motion_estimate(path)
    # img = color_coding(global_motion, max_rad)

    # img_out = Image.fromarray(img)
    # img_out.save('filename.png')

    flow_dir = './flow_data'
    save_dir = 'D:/flow_multi_mode'

    #
    #
    # match_name_set = ['0aWfrZAM6Q8']
    # for match_name in match_name_set:
    #     print(match_name)
    #     event_name_set = os.listdir(os.path.join(flow_dir, match_name))
    #     for event_name in event_name_set:
    #         print(event_name)
    #         sub_time_set = os.listdir(os.path.join(flow_dir, match_name, event_name))
    #         for sub_time in sub_time_set:
    #             print(sub_time)
    #             flow_set = natsort.natsorted(os.listdir(os.path.join(flow_dir, match_name, event_name, sub_time)))
    #             # img_data = []
    #             for flow_name in flow_set:
    #                 flow = os.path.join(flow_dir, match_name, event_name, sub_time, flow_name)
    #                 # fl.visualize_flow(fl2.read_flow(flow))
    #                 # cv2.imshow(cv2.namedWindow("0"), cv2.imread(flow))
    #                 global_motion, max_rad = global_motion_estimate(flow)
    #                 # std = np.std(global_motion)
    #                 # print(std)
    #
    #                 x_global_motion = global_motion[0:1, :, 0]
    #                 y_global_motion = global_motion[:, 0:1, 1]
    #
    #                 x1 = (x_global_motion[0, -1] - x_global_motion[0, 0]) / 2
    #                 x2 = x_global_motion[0, -1] - x1
    #
    #                 y1 = (y_global_motion[-1, 0] - y_global_motion[0, 0]) / 2
    #                 y2 = y_global_motion[-1, 0] - y1
    #
    #                 print("缩放量为：(%f, %f)" % (x1, y1))
    #                 print("平移量为：(%f, %f)" % (x2, y2), '\n')
    #
    #
    #                 img = color_coding(global_motion, max_rad)
    #                 # plt.imshow(img)
    #                 # plt.show()
    #                 cv2.imwrite('D:/flow_multi_mode/1.png', cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    #                 # plt.savefig('D:/flow_multi_mode/1.jpg')
    #                 # plt.imshow(img)
    #                 # plt.show()
    #                 # savefig('D:/flow_multi_mode')
    #                 # img_data.append(img)

    flow = 'E:/project/capstone/main/flow/flo/9/2.flo'
    # fl.visualize_flow(fl2.read_flow(flow))
    # cv2.imshow(cv2.namedWindow("0"), cv2.imread(flow))
    global_motion, max_rad = global_motion_estimate(flow)
    # std = np.std(global_motion)
    # print(std)

    x_global_motion = global_motion[0:1, :, 0]
    y_global_motion = global_motion[:, 0:1, 1]

    x1 = (x_global_motion[0, -1] - x_global_motion[0, 0]) / 2
    x2 = x_global_motion[0, -1] - x1

    y1 = (y_global_motion[-1, 0] - y_global_motion[0, 0]) / 2
    y2 = y_global_motion[-1, 0] - y1

    print("缩放量为：(%f, %f)" % (x1, y1))
    print("平移量为：(%f, %f)" % (x2, y2), '\n')

