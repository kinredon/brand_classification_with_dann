# coding: UTF-8
# 生成对应训练数据和验证数据文件列表
import random

def gen_test_and_train_set(list_path, train_list_path, test_list_path, _NUM_TEST = 240, _RANDOM_SEED = 0):

    fd = open(list_path)
    lines = fd.readlines()              # 读取所有的行
    fd.close()
    # random.seed(_RANDOM_SEED)           # 设置随机种子
    # random.shuffle(lines)               # 随机调整顺序
    ft = open(train_list_path, 'w')     # 写入验证集
    fd = open(test_list_path, 'w')  # 写入测试集
    for i, line in enumerate(lines):
        if i % 100 >= 0 and i % 100 < 20:
            fd.write(line)
        else:
            ft.write(line)
    ft.close()
    fd.close()


def gen_test_and_train_set2(list_path, train_list_path, test_list_path, f_list, percent = 0.2):

    fd = open(list_path)
    lines = fd.readlines()              # 读取所有的行
    fd.close()
    # random.seed(_RANDOM_SEED)           # 设置随机种子
    # random.shuffle(lines)               # 随机调整顺序

    ft = open(train_list_path, 'w')     # 写入验证集
    fd = open(test_list_path, 'w')  # 写入测试集
    sump = 0
    for i in f_list:
        for j in range(i):
            if j < i * percent:
                fd.write(lines[sump + j])
            else:
                ft.write(lines[sump + j])
        sump += i

    ft.close()
    fd.close()

if __name__ == '__main__':

    # _NUM_TEST = 240                             # 验证集的数量
    # _RANDOM_SEED = 0                            # 随机种子
    # list_path = 'source_list.txt'               # 文件列表
    # train_list_path = 'source_list_train.txt'   # 生成的训练数据文件
    # test_list_path = 'source_list_test.txt'     # 生成的验证数据文件
    # gen_test_and_train_set(list_path, train_list_path, test_list_path, _NUM_TEST=240, _RANDOM_SEED=0)
    #
    # list_path = 'target_list.txt'               # 文件列表
    # train_list_path = 'target_list_train.txt'   # 生成的训练数据文件
    # test_list_path = 'target_list_test.txt'     # 生成的验证数据文件
    # gen_test_and_train_set(list_path, train_list_path, test_list_path, percent=240, _RANDOM_SEED=0)
    ft_list = [200, 150, 300, 300, 300, 200, 300, 110]
    fs_list = [250, 400, 400, 300, 300, 160, 400, 400]

    list_path = 'source_list.txt'  # 文件列表
    train_list_path = 'source_list_train.txt'  # 生成的训练数据文件
    test_list_path = 'source_list_test.txt'  # 生成的验证数据文件
    gen_test_and_train_set2(list_path, train_list_path, test_list_path, fs_list, percent=0.2)

    list_path = 'target_list.txt'  # 文件列表
    train_list_path = 'target_list_train.txt'  # 生成的训练数据文件
    test_list_path = 'target_list_test.txt'  # 生成的验证数据文件
    gen_test_and_train_set2(list_path, train_list_path, test_list_path, ft_list, percent=0.2)
