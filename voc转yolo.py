import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join


def convert(size, box):
    x_center = (box[0] + box[1]) / 2.0
    y_center = (box[2] + box[3]) / 2.0
    x = x_center / size[0]
    y = y_center / size[1]
    w = (box[1] - box[0]) / size[0]
    h = (box[3] - box[2]) / size[1]
    return (x, y, w, h)


def convert_annotation(xml_files_path, save_txt_files_path, classes):
    """xml_files_path为xml文件的路径, save_txt_files_path为保存txt文件的路径"""
    xml_files = os.listdir(xml_files_path)  #  读取xml文件路径下的所有文件，返回文件名列表
    xml_files = [f for f in xml_files if f.endswith('.xml')]  #  保留尾缀为.xml的文件
    print(xml_files)  #  展示文件夹下的所有文件
    for xml_name in xml_files:  #  遍历整个文件
        print(xml_name)  #  输出当前操作的文件
        xml_file = os.path.join(xml_files_path, xml_name)  #  xml_file为改文件的具体路径
        out_txt_path = os.path.join(save_txt_files_path, xml_name.rsplit('.', 1)[0] + '.txt')  #  写好输出文件的路径及名称
        out_txt_f = open(out_txt_path, 'w')  #  创建写入对象
        tree = ET.parse(xml_file)  #  将xml文档表示为树
        root = tree.getroot()  #  树的根目录
        size = root.find('size')  #  找到root目录下的size节点
        w = int(size.find('width').text)  #  找到size节点下的width节点并得到第一个子元素之前的文本
        h = int(size.find('height').text)  #  找到size节点下的height节点并得到第一个子元素之前的文本

        for obj in root.iter('object'):  #  迭代遍历整个root节点的object节点
            difficult = obj.find('difficult').text  # 读取difficult
            cls = obj.find('name').text  # 读取name
            if cls not in classes or int(difficult) == 1:  # 如果类不属于或者目标困难, 则不标注
                continue
            cls_id = classes.index(cls)  # 找到该类对应列表的编号
            xmlbox = obj.find('bndbox')  # 读取文件中的xywh
            b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
                 float(xmlbox.find('ymax').text))
            # b=(xmin, xmax, ymin, ymax)
            print(w, h, b)
            bb = convert((w, h), b)  # 转换
            out_txt_f.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')


if __name__ == "__main__":
    # 需要转换的类别，需要一一对应
    # classes1 = ['crazing','inclusion','patches','pitted_surface','rolled-in_scale','scratches']
    classes1 = ['youwu','jiaoban','huahen','fenbi','chacheyin','quebian','baohua','songruan']
    # 2、voc格式的xml标签文件路径
    xml_files1 = r'C:\Users\LazyShark\Desktop\RZB\anno'
    # 3、转化为yolo格式的txt标签文件存储路径
    save_txt_files1 = r'C:\Users\LazyShark\Desktop\RZB\2'

    convert_annotation(xml_files1, save_txt_files1, classes1)