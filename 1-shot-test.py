import cv2
import numpy as np
import os
from scipy.spatial.distance import pdist
import glob
# from natsort import natsorted
from PIL import Image, ImageStat
import shutil

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


cap = cv2.VideoCapture('Untitled1.mp4')    # 将这个改为手动选择文件
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

dir = 'E:/1/'
os.makedirs(os.path.join(dir, 'hockey', str(o1)));   # 在这儿更改保存位置
save_dir = os.path.join(dir, 'hockey')     # 在这儿更改保存位置
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
        if  barMinus(lastframe3, frame3)>= 0.18:
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
#
# img_num_list = []
# for i in range(1, o1):                     # 剔除0号和末号
#     img_list = natsorted(glob.glob(save_dir + '//' + '{}'.format(i) + '/*.jpg'))
#     img_num = len(img_list)
#     if img_num <= 10:                   # 判断图片数量时，文件夹内图片的数量还得大于50，不然直接删
#         path = save_dir + '//' + '{}'.format(i)
#         shutil.rmtree(path)
#         print('小于80：{}'.format(i))
