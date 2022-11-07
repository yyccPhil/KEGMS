import time
from matplotlib import pyplot as plt
import os
import cv2
import numpy as np
import glob
from natsort import natsorted
import xlwt

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

    # base_dir = 'D:/py/keyframe_data/global_pic'
    base_dir = 'E:/project/data_cba/key_data/key_global_2'
    # save_path = 'D:/py/keyframe_data/'

    game_list = os.listdir(base_dir)
    for game in game_list:
        event_list = os.listdir(os.path.join(base_dir, game))
        for event in event_list:
            print(event)
            workbook = xlwt.Workbook(encoding='utf-8')
            pic_list = natsorted(os.listdir(os.path.join(base_dir, game, event)))
            img_count = 0
            for pic in pic_list:
                time_start = time.time()
                # print(pic)
                img_list = natsorted(glob.glob(os.path.join(base_dir, game, event, pic) + '/*.png'))
                img_num = len(img_list)
                end = img_num - 1

                firstfilename = os.path.basename(img_list[0])
                first_filename = firstfilename.replace("global_", "")

                save_count = 1

                worksheet = workbook.add_sheet('{}'.format(pic))
                worksheet.write(0, 0, label=u'pic')
                worksheet.write(0, 1, label=u'巴氏距离')
                worksheet.write(0, 2, label=u'相关性')
                worksheet.write(0, 3, label=u'卡方')
                worksheet.write(0, 4, label=u'HISTCMP_INTERSECT')

                for index in range(0, end, 1):
                    img1_path = img_list[index]
                    img_rgb1 = cv2.imread(img1_path)
                    img_hsi1 = rgb2hsi(img_rgb1)
                    # hsv1 = cv2.cvtColor(img_rgb1, cv2.COLOR_RGB2HSV)
                    # hist_h1 = cv2.calcHist([hsv1], [0], None, [360], [0, 359])
                    hist_hi1 = cv2.calcHist([img_hsi1], [0, 2], None, [25, 25], [0, 359, 0, 255])

                    img2_path = img_list[index + 1]
                    img_rgb2 = cv2.imread(img2_path)
                    img_hsi2 = rgb2hsi(img_rgb2)
                    # hsv2 = cv2.cvtColor(img_rgb2, cv2.COLOR_RGB2HSV)
                    # hist_h2 = cv2.calcHist([hsv2], [0], None, [360], [0, 359])
                    hist_hi2 = cv2.calcHist([img_hsi2], [0, 2], None, [25, 25], [0, 359, 0, 255])

                    myHist = Hisogram()
                    # match1, match2, match3, match4 = myHist.hist_compare(hist_h1, hist_h2)
                    match1, match2, match3, match4 = myHist.hist_compare(hist_hi1, hist_hi2)

                    img_name = os.path.basename(img2_path).replace(".png", "")
                    flo_count = img_name.replace("global_", "")

                    worksheet.write(save_count, 0, flo_count)
                    worksheet.write(save_count, 1, match1)
                    worksheet.write(save_count, 2, match2)
                    worksheet.write(save_count, 3, match3)
                    worksheet.write(save_count, 4, match4)

                    save_count += 1
                    img_count += 1
                time_end = time.time()
                print('Time cost = %fs' % (time_end - time_start))

            # print(img_count)
            xls_save_path = os.path.join('E:/project/data_cba/excel-2', game)
            isExists_2 = os.path.exists(xls_save_path)
            if not isExists_2:
                os.makedirs(xls_save_path)
            workbook.save(xls_save_path + '/{}_hist.xls'.format(event))  
