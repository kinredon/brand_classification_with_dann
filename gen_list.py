# coding: UTF-8
import os
import random
data_dir = '../vehicle/'                   # 图片目录
source_list_path = 'source_list.txt'       # 源list输出文件
target_list_path = 'target_list.txt'       # 目标list输出文件
label_path = 'labels.txt'                  # 标签输出文件
_RANDOM_SEED = 0                           # 随机种子

class_names_to_ids = {}
class_names = os.listdir(data_dir)
f = open(label_path, 'w')
count = 0
# 将类别输出到文件中
for i in range(0, len(class_names)):
    if os.path.isdir(data_dir + class_names[i]) and '_add' not in class_names[i]:
        class_names_to_ids[class_names[i]] = count
        count = count + 1
        f.write(class_names[i] + '\n')
f.close()


sfd = open(source_list_path, 'w')
tfd = open(target_list_path, 'w')

# 将图片地址与类别一一对应，并写入文件中
for class_name in class_names_to_ids.keys():
    if os.path.isdir(data_dir + class_name):
        source_nums = 0
        target_nums = 0
        images_list = os.listdir(data_dir + class_name + "_add")
        # random.seed(_RANDOM_SEED)                       # 设置随机种子
        # random.shuffle(images_list)                     # 随机调整顺序
        for image_name in images_list:
            if image_name.split('.')[-1] in ['jpeg', 'png', 'jpg']:
                sfd.write('{}/{} {}\n'.format( data_dir + class_name + "_add", image_name, class_names_to_ids[class_name]))

        images_list = os.listdir(data_dir + class_name)

        for image_name in images_list:
            if image_name.split('.')[-1] in ['jpeg', 'png', 'jpg']:
                tfd.write('{}/{} {}\n'.format(data_dir + class_name, image_name, class_names_to_ids[class_name]))

sfd.close()
tfd.close()
