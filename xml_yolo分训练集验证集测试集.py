# 导入的模块
import time

import torch
import os
import random
import xml.etree.ElementTree as ET
# 配置项
xml_config = {
    # Annotations path(Annotations 的文件夹路径)
    "Annotation":r"C:\Users\LazyShark\Desktop\bupi\fabrics\anno",
    # JPEGImages path(JPEGImages 的文件夹路径)
    "JPEGImages":r"C:\Users\LazyShark\Desktop\bupi\fabrics\images",
}

yolo_config = {
    # labels path(labels 的文件夹路径)
    "labels":r"C:\Users\LazyShark\Desktop\RZB_5\labels",
    # images path(images 的文件夹路径)
    "images":r"C:\Users\LazyShark\Desktop\RZB_5\images",
}

# 划分数据集


#获取指定目录下的所有图片
# print (glob.glob(r"/home/qiaoyunhao/*/*.png"),"\n")#加上r让字符串不转义
# 数据划分比例
#获取上级目录的所有.py文件
# print (glob.glob(r'../*.py')) #相对路径


# (训练集+验证集)与测试集的比例，默认情况下 (训练集+验证集):测试集 = 9:1

# 按照比例划分数据集
train_per = 0.8
valid_per = 0.1
test_per = 0.1

def xml(suffix = 'jpg'):
    data_list = [f for f in os.listdir(xml_config['JPEGImages']) if f.endswith(f'.{suffix}')]  # 保留对应尾缀的文件
    random.seed(666)  # 设置随机种子，之后的random操作中数值都一样
    random.shuffle(data_list)  # 将文本列表顺序打乱
    data_length = len(data_list)  # 获取列表长度

    train_point = int(data_length * train_per)  # 训练集数目
    train_valid_point = int(data_length * (train_per + valid_per))  # 训练集加验证集数目

    # 生成训练集，验证集, 测试集(8 : 1 : 1)
    train_list = data_list[:train_point]
    valid_list = data_list[train_point:train_valid_point]
    test_list = data_list[train_valid_point:]

    # 生成label标签:
    label = set()  # set()是一个集合，没有顺序不含相同元素
    for data_path in data_list:
        xml_path = os.path.join(xml_config['Annotation'], data_path.rsplit('.', 1)[0] + '.xml')
        # label = label | set([i.find('name').text for i in ET.parse(xml_path).findall('object')])  # 获得name的值并填充到一个列表加入set
        label = label | set([i.find('name').text for i in ET.parse(xml_path).iter('object')])  # 获得name的值并填充到一个列表加入set

    # 写入文件中
    ftrain = open(os.path.join(os.path.dirname(xml_config['JPEGImages']),'train.txt'), 'w')  # 返回上一级目录写文件
    fvalid = open(os.path.join(os.path.dirname(xml_config['JPEGImages']),'val.txt'), 'w')
    ftest = open(os.path.join(os.path.dirname(xml_config['JPEGImages']),'test.txt'), 'w')
    flabel = open(os.path.join(os.path.dirname(xml_config['JPEGImages']),'label.txt'), 'w')

    for i in train_list:
        save_test = os.path.join(xml_config['JPEGImages'], i[:-4] +"." + suffix + "\n")
        ftrain.write(save_test)
    for j in valid_list:
        save_test = os.path.join(xml_config['JPEGImages'], j[:-4] +"." + suffix + "\n")
        fvalid.write(save_test)
    for k in test_list:
        save_test = os.path.join(xml_config['JPEGImages'], k[:-4] +"." + suffix + "\n")
        ftest.write(save_test)
    for l in label:
        flabel.write(l + "\n")
    ftrain.close()
    fvalid.close()
    ftest.close()
    flabel.close()
    print("总数据量:{}, 训练集:{}, 验证集:{}, 测试集:{}, 标签:{}".format(len(data_list), len(train_list), len(valid_list),
                                                          len(test_list), len(label)))
    print("done!")


def yolo(suffix = 'jpg'):
    data_list = [f for f in os.listdir(yolo_config['images']) if f.endswith(f'.{suffix}')]  # 保留对应尾缀的文件
    random.seed(666)  # 设置随机种子，之后的random操作中数值都一样
    # random.seed(time.time())  # 设置随机种子，之后的random操作中数值都一样
    random.shuffle(data_list)  # 将文本列表顺序打乱
    data_length = len(data_list)  # 获取列表长度

    train_point = int(data_length * train_per)  # 训练集数目
    train_valid_point = int(data_length * (train_per + valid_per))  # 训练集加验证集数目

    # 生成训练集，验证集, 测试集(8 : 1 : 1)
    train_list = data_list[:train_point]
    valid_list = data_list[train_point:train_valid_point]
    test_list = data_list[train_valid_point:]

    # 写入文件中
    ftrain = open(os.path.join(os.path.dirname(yolo_config['images']),'train.txt'), 'w')  # 返回上一级目录写文件
    fvalid = open(os.path.join(os.path.dirname(yolo_config['images']),'valid.txt'), 'w')
    ftest = open(os.path.join(os.path.dirname(yolo_config['images']),'test.txt'), 'w')

    for i in train_list:
        save_test = os.path.join(yolo_config['images'], i[:-4] +"." + suffix + "\n")
        ftrain.write(save_test)
    for j in valid_list:
        save_test = os.path.join(yolo_config['images'], j[:-4] +"." + suffix + "\n")
        fvalid.write(save_test)
    for k in test_list:
        save_test = os.path.join(yolo_config['images'], k[:-4] +"." + suffix + "\n")
        ftest.write(save_test)
    ftrain.close()
    fvalid.close()
    ftest.close()
    print("总数据量:{}, 训练集:{}, 验证集:{}, 测试集:{}".format(len(data_list), len(train_list), len(valid_list),
                                                          len(test_list)))
    print("done!")

# 需要转换的尾缀
yolo('jpg')
