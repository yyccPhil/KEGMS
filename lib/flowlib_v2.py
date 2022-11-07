#!/usr/bin/python
"""
# ==============================
# flowlib.py
# library for optical flow processing
# Author: Ruoteng Li
# Date: 6th Aug 2016
# ==============================
"""
import sys

sys.path.append(r'E:\pyproject\gme\lib')
# import png
# import pfm
import numpy as np
import matplotlib.colors as cl
import matplotlib.pyplot as plt
import cv2
from PIL import Image

UNKNOWN_FLOW_THRESH = 1e7
SMALLFLOW = 0.0
LARGEFLOW = 1e8

"""
=============
Flow Section
=============
"""


def read_flow(filename):
    """
    read optical flow data from flow file
    :param filename: name of the flow file
    :return: optical flow data in numpy array
    """
    if filename.endswith('.flo'):
        flow = read_flo_file(filename)
    elif filename.endswith('.png'):
        flow = read_png_file(filename)
    elif filename.endswith('.pfm'):
        flow = read_pfm_file(filename)
    else:
        raise Exception('Invalid flow file format!')

    return flow


def write_flow(flow, filename):
    """
    write optical flow in Middlebury .flo format
    :param flow: optical flow map
    :param filename: optical flow file path to be saved
    :return: None
    """
    f = open(filename, 'wb')
    magic = np.array([202021.25], dtype=np.float32)
    (height, width) = flow.shape[0:2]
    w = np.array([width], dtype=np.int32)
    h = np.array([height], dtype=np.int32)
    magic.tofile(f)
    w.tofile(f)
    h.tofile(f)
    flow.tofile(f)
    f.close()


def compute_maxrad(flow):
    u = flow[:, :, 0]
    v = flow[:, :, 1]

    maxu = -999.
    maxv = -999.
    minu = 999.
    minv = 999.

    maxu = max(maxu, np.max(u))
    minu = min(minu, np.min(u))

    maxv = max(maxv, np.max(v))
    minv = min(minv, np.min(v))

    rad = np.sqrt(u ** 2 + v ** 2)  # 光流场方向
    maxrad = max(-1, np.max(rad))

    # print("max flow: %.4f\nflow range:\nu = %.3f .. %.3f\nv = %.3f .. %.3f" % (maxrad, minu, maxu, minv, maxv))
    return maxrad


def visualize_flow(flow, rad_max):
    img = flow_to_image(flow, rad_max)
    plt.imshow(img)
    plt.show()


def flow_to_image(flow, rad_max):
    """
    Convert flow into middlebury color code image
    :param rad_max: max amplitude of the motion vectors
    :param flow: optical flow map
    :return: optical flow image in middlebury color
    """
    u = flow[:, :, 0]
    v = flow[:, :, 1]

    maxu = -999.
    maxv = -999.
    minu = 999.
    minv = 999.
    # 把光流场中x y方向幅度均大于UNKNOWN_FLOW_THRESH的值置零
    idxUnknow = (abs(u) > UNKNOWN_FLOW_THRESH) | (abs(v) > UNKNOWN_FLOW_THRESH)
    u[idxUnknow] = 0
    v[idxUnknow] = 0

    # np.finfo(float).eps 用来提升数据精度
    # u / maxrad 对u v 进行归一化，得到的color coding结果与运动幅度无关，只与运动方向有关（）
    # 归一化后，x 方向 y方向的合成运动幅度必为1
    u = u / (rad_max + np.finfo(float).eps)
    v = v / (rad_max + np.finfo(float).eps)
    u = u + np.finfo(float).eps
    v = v + np.finfo(float).eps

    img = compute_color(u, v)

    # 在axis=3维度复制三遍将 m*n 的矩阵扩展成 m*n*t t个m*n矩阵值相等，idx是光流值异常的点坐标
    idx = np.repeat(idxUnknow[:, :, np.newaxis], 3, axis=2)
    img[idx] = 0

    return np.uint8(img)


"""
==============
Disparity Section
==============
"""


def read_disp_png(file_name):
    """
    Read optical flow from KITTI .png file
    :param file_name: name of the flow file
    :return: optical flow data in matrix
    """
    image_object = png.Reader(filename=file_name)
    image_direct = image_object.asDirect()
    image_data = list(image_direct[2])
    (w, h) = image_direct[3]['size']
    channel = len(image_data[0]) / w
    flow = np.zeros((h, w, channel), dtype=np.uint16)
    for i in range(len(image_data)):
        for j in range(channel):
            flow[i, :, j] = image_data[i][j::channel]
    return flow[:, :, 0] / 256


def disp_to_flowfile(disp, filename):
    """
    Read KITTI disparity file in png format
    :param disp: disparity matrix
    :param filename: the flow file name to save
    :return: None
    """
    f = open(filename, 'wb')
    magic = np.array([202021.25], dtype=np.float32)
    (height, width) = disp.shape[0:2]
    w = np.array([width], dtype=np.int32)
    h = np.array([height], dtype=np.int32)
    empty_map = np.zeros((height, width), dtype=np.float32)
    data = np.dstack((disp, empty_map))
    magic.tofile(f)
    w.tofile(f)
    h.tofile(f)
    data.tofile(f)
    f.close()


"""
==============
Image Section
==============
"""


def read_image(filename):
    """
    Read normal image of any format
    :param filename: name of the image file
    :return: image data in matrix uint8 type
    """
    img = Image.open(filename)
    im = np.array(img)
    return im


def warp_image(im, flow):
    """
    Use optical flow to warp image to the next
    :param im: image to warp
    :param flow: optical flow
    :return: warped image
    """
    from scipy import interpolate
    image_height = im.shape[0]
    image_width = im.shape[1]
    flow_height = flow.shape[0]
    flow_width = flow.shape[1]
    n = image_height * image_width
    (iy, ix) = np.mgrid[0:image_height, 0:image_width]
    (fy, fx) = np.mgrid[0:flow_height, 0:flow_width]
    fx = fx.astype(np.float64)
    fy = fy.astype(np.float64)
    fx += flow[:, :, 0]
    fy += flow[:, :, 1]
    mask = np.logical_or(fx < 0, fx > flow_width)
    mask = np.logical_or(mask, fy < 0)
    mask = np.logical_or(mask, fy > flow_height)
    fx = np.minimum(np.maximum(fx, 0), flow_width)
    fy = np.minimum(np.maximum(fy, 0), flow_height)
    points = np.concatenate((ix.reshape(n, 1), iy.reshape(n, 1)), axis=1)
    xi = np.concatenate((fx.reshape(n, 1), fy.reshape(n, 1)), axis=1)
    warp = np.zeros((image_height, image_width, im.shape[2]))
    for i in range(im.shape[2]):
        channel = im[:, :, i]
        plt.imshow(channel, cmap='gray')
        values = channel.reshape(n, 1)
        new_channel = interpolate.griddata(points, values, xi, method='cubic')
        new_channel = np.reshape(new_channel, [flow_height, flow_width])
        new_channel[mask] = 1
        warp[:, :, i] = new_channel.astype(np.uint8)

    return warp.astype(np.uint8)


"""
==============
Others
==============
"""


def pfm_to_flo(pfm_file):
    flow_filename = pfm_file[0:pfm_file.find('.pfm')] + '.flo'
    (data, scale) = pfm.readPFM(pfm_file)
    flow = data[:, :, 0:2]
    write_flow(flow, flow_filename)


def scale_image(image, new_range):
    """
    Linearly scale the image into desired range
    :param image: input image
    :param new_range: the new range to be aligned
    :return: image normalized in new range
    """
    min_val = np.min(image).astype(np.float32)
    max_val = np.max(image).astype(np.float32)
    min_val_new = np.array(min(new_range), dtype=np.float32)
    max_val_new = np.array(max(new_range), dtype=np.float32)
    scaled_image = (image - min_val) / (max_val - min_val) * (max_val_new - min_val_new) + min_val_new
    return scaled_image.astype(np.uint8)


def compute_color(u, v):
    """
    compute optical flow color map
    :param u: optical flow horizontal map
    :param v: optical flow vertical map
    :return: optical flow in color code
    """
    [h, w] = u.shape
    img = np.zeros([h, w, 3])
    nanIdx = np.isnan(u) | np.isnan(v)
    u[nanIdx] = 0
    v[nanIdx] = 0

    colorwheel = make_color_wheel()

    ncols = np.size(colorwheel, 0)

    # rad 勾股定理 x方向和y方向光流幅度融合
    rad = np.sqrt(u ** 2 + v ** 2)

    # 计算输入的两个分量的合成分量的角度[-1，1]，x1是y方向分量大小，x2是x方向分量大小
    # >>> x = np.array([-1, +1, +1, -1])
    # >>> y = np.array([-1, -1, +1, +1])
    # >>> np.arctan2(y, x) * 180 / np.pi
    # array([-135.,  -45.,   45.,  135.])
    a = np.arctan2(-v, -u) / np.pi

    fk = (a + 1) / 2 * (ncols - 1) + 1

    k0 = np.floor(fk).astype(int)

    k1 = k0 + 1
    k1[k1 == ncols + 1] = 1
    f = fk - k0

    for i in range(0, np.size(colorwheel, 1)):
        tmp = colorwheel[:, i]
        col0 = tmp[k0 - 1] / 255
        col1 = tmp[k1 - 1] / 255
        col = (1 - f) * col0 + f * col1

        idx = rad <= 1
        col[idx] = 1 - rad[idx] * (1 - col[idx])
        notidx = np.logical_not(idx)

        col[notidx] *= 0.75
        img[:, :, i] = np.uint8(np.floor(255 * col * (1 - nanIdx)))

    return img


def make_color_wheel():
    """
    Generate color wheel according Middlebury color code
    :return: Color wheel
    """
    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR

    colorwheel = np.zeros([ncols, 3])

    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.transpose(np.floor(255 * np.arange(0, RY) / RY))
    col += RY

    # YG
    colorwheel[col:col + YG, 0] = 255 - np.transpose(np.floor(255 * np.arange(0, YG) / YG))
    colorwheel[col:col + YG, 1] = 255
    col += YG

    # GC
    colorwheel[col:col + GC, 1] = 255
    colorwheel[col:col + GC, 2] = np.transpose(np.floor(255 * np.arange(0, GC) / GC))
    col += GC

    # CB
    colorwheel[col:col + CB, 1] = 255 - np.transpose(np.floor(255 * np.arange(0, CB) / CB))
    colorwheel[col:col + CB, 2] = 255
    col += CB

    # BM
    colorwheel[col:col + BM, 2] = 255
    colorwheel[col:col + BM, 0] = np.transpose(np.floor(255 * np.arange(0, BM) / BM))
    col += + BM

    # MR
    colorwheel[col:col + MR, 2] = 255 - np.transpose(np.floor(255 * np.arange(0, MR) / MR))
    colorwheel[col:col + MR, 0] = 255

    return colorwheel


def read_flo_file(filename):
    """
    Read from Middlebury .flo file
    :param flow_file: name of the flow file
    :return: optical flow data in matrix
    """
    f = open(filename, 'rb')
    magic = np.fromfile(f, np.float32, count=1)
    data2d = None

    if 202021.25 != magic:
        print('Magic number incorrect. Invalid .flo file')
    else:
        w = np.fromfile(f, np.int32, count=1)
        h = np.fromfile(f, np.int32, count=1)
        # print("Reading %d x %d flow file in .flo format" % (h, w))
        data2d = np.fromfile(f, np.float32, count=2 * w[0] * h[0])
        # reshape data into 3D array (columns, rows, channels)
        data2d = np.resize(data2d, (h[0], w[0], 2))
    f.close()
    return data2d


def read_png_file(flow_file):
    """
    Read from KITTI .png file
    :param flow_file: name of the flow file
    :return: optical flow data in matrix
    """
    flow_object = png.Reader(filename=flow_file)
    flow_direct = flow_object.asDirect()
    flow_data = list(flow_direct[2])
    (w, h) = flow_direct[3]['size']
    print("Reading %d x %d flow file in .png format" % (h, w))
    flow = np.zeros((h, w, 3), dtype=np.float64)
    for i in range(len(flow_data)):
        flow[i, :, 0] = flow_data[i][0::3]
        flow[i, :, 1] = flow_data[i][1::3]
        flow[i, :, 2] = flow_data[i][2::3]

    invalid_idx = (flow[:, :, 2] == 0)
    flow[:, :, 0:2] = (flow[:, :, 0:2] - 2 ** 15) / 64.0
    flow[invalid_idx, 0] = 0
    flow[invalid_idx, 1] = 0
    return flow


def read_pfm_file(flow_file):
    """
    Read from .pfm file
    :param flow_file: name of the flow file
    :return: optical flow data in matrix
    """

    (data, scale) = pfm.readPFM(flow_file)
    return data


# fast resample layer
def resample(img, sz):
    """
    img: flow map to be resampled
    sz: new flow map size. Must be [height,weight]
    """
    original_image_size = img.shape
    in_height = img.shape[0]
    in_width = img.shape[1]
    out_height = sz[0]
    out_width = sz[1]
    out_flow = np.zeros((out_height, out_width, 2))
    # find scale
    height_scale = float(in_height) / float(out_height)
    width_scale = float(in_width) / float(out_width)

    [x, y] = np.meshgrid(range(out_width), range(out_height))
    xx = x * width_scale
    yy = y * height_scale
    x0 = np.floor(xx).astype(np.int32)
    x1 = x0 + 1
    y0 = np.floor(yy).astype(np.int32)
    y1 = y0 + 1

    x0 = np.clip(x0, 0, in_width - 1)
    x1 = np.clip(x1, 0, in_width - 1)
    y0 = np.clip(y0, 0, in_height - 1)
    y1 = np.clip(y1, 0, in_height - 1)

    Ia = img[y0, x0, :]
    Ib = img[y1, x0, :]
    Ic = img[y0, x1, :]
    Id = img[y1, x1, :]

    wa = (y1 - yy) * (x1 - xx)
    wb = (yy - y0) * (x1 - xx)
    wc = (y1 - yy) * (xx - x0)
    wd = (yy - y0) * (xx - x0)
    out_flow[:, :, 0] = (Ia[:, :, 0] * wa + Ib[:, :, 0] * wb + Ic[:, :, 0] * wc + Id[:, :,
                                                                                  0] * wd) * out_width / in_width
    out_flow[:, :, 1] = (Ia[:, :, 1] * wa + Ib[:, :, 1] * wb + Ic[:, :, 1] * wc + Id[:, :,
                                                                                  1] * wd) * out_height / in_height

    return out_flow


def segment_flow(flow):
    h = flow.shape[0]
    w = flow.shape[1]
    u = flow[:, :, 0]
    v = flow[:, :, 1]

    idx = ((abs(u) > LARGEFLOW) | (abs(v) > LARGEFLOW))
    idx2 = (abs(u) == SMALLFLOW)
    class0 = (abs(u) < 0.2) & (abs(v) < 0.2)
    u[idx2] = 0.00001

    # 范围是 [-1 ,1] (0, 1)->1 (1, 0)->1
    angle = np.arctan2(-v, -u) / np.pi

    class1 = (angle >= 0) & (angle < 0.25)
    class2 = (angle >= 0.25) & (angle < 0.5)
    class3 = (angle >= 0.5) & (angle < 0.75)
    class4 = (angle >= 0.75) & (angle <= 1)
    class5 = (angle > -1) & (angle < -0.75)
    class6 = (angle >= -0.75) & (angle < -0.5)
    class7 = (angle >= -0.5) & (angle < -0.25)
    class8 = (angle >= -0.25) & (angle < 0)

    seg = np.zeros((h, w))

    seg[class1] = 1
    seg[class2] = 2
    seg[class3] = 3
    seg[class4] = 4
    seg[class5] = 5
    seg[class6] = 6
    seg[class7] = 7
    seg[class8] = 8
    seg[class0] = 0
    seg[idx] = 0

    return seg