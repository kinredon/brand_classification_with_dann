# -*- coding: utf-8 -*-
# created by kinredon
# /Users/kinredon/Documents/vehicle

import os
data_dir = "/Users/kinredon/Documents/vehicle"
# 找到所有品牌的文件夹
files = os.listdir(data_dir)  # 得到文件夹下的所有文件名称

with open('count.txt', 'w') as f:
    f.write("vehicle data statistical：" + "\n")
# 遍历品牌
count = 0
for file in files:
    if os.path.isdir(data_dir + '/' + file):
        print(file)
        try:
            names = os.listdir(data_dir + '/' + file)
            count += 1
            origin_img_num = 0
            spider_img_num = 0
            for name in names:
                if name.split('.')[-1] == "urls":
                    continue
                elif "pic_" in name:
                    spider_img_num = spider_img_num + 1
                else:
                    origin_img_num = origin_img_num + 1
            with open('count.txt', 'a') as f:
                if count % 2 == 0:
                    f.write(str(count) + "|" + str(file) + "\t| " + str(origin_img_num)
                            + "\t| " + str(spider_img_num)
                            + "\t|" + "\n")
                else:
                    f.write("|" + str(count) + "|" + str(file) + "\t| " + str(origin_img_num)
                            + "\t| " + str(spider_img_num)
                            + "\t|")

        except Exception as e:
            print(e)

