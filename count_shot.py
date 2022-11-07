import os
import glob
from natsort import natsorted
import shutil

# base_dir = 'E:/project/capstone/data/origin_img/soccer'
base_dir = 'E:/project/capstone/data/test/nba'

sum = 0
event_list = os.listdir(base_dir)
for event in event_list:
    count = 0
    pic_list = natsorted(glob.glob(os.path.join(base_dir, event) + '/*.jpg'))
    # pic_list = natsorted(glob.glob(os.path.join(base_dir, event) + '/*.npy'))
    img_num = len(pic_list)
    # print(img_num)
    count += img_num
    print(count)
    sum += count
    print()
print(sum)
