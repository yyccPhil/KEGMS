import os
import glob
from natsort import natsorted
import shutil

base_dir = 'E:/project/capstone/data/data/key_data/key'

sum = 0
event_list = os.listdir(base_dir)
for event in event_list:
    pic_list = natsorted(os.listdir(os.path.join(base_dir, event)))
    for pic in pic_list:
        print(pic)
        count = 0
        flo_list = natsorted(os.listdir(os.path.join(base_dir, event, pic)))
        for time in flo_list:
            img_list = natsorted(glob.glob(os.path.join(base_dir, event, pic, time) + '/*.jpg'))
            img_num = len(img_list)
            # print(img_num)
            count += img_num
        print(count)
        sum += count
        print()
print(sum)
