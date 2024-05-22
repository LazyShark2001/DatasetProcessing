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


def convert_annotation(xml_files_path, save_txt_files_path, classes1, classes2, classes3, classes4):
    """xml_files_path为xml文件的路径, save_txt_files_path为保存txt文件的路径"""
    xml_files = os.listdir(xml_files_path)  #  读取xml文件路径下的所有文件
    print(xml_files)  #  展示文件夹下的所有文件
    for xml_name in xml_files:  #  遍历整个文件
        print(xml_name)  #  输出当前操作的文件
        xml_file = os.path.join(xml_files_path, xml_name)  #  xml_file为改文件的具体路径
        out_txt_path = os.path.join(save_txt_files_path, xml_name.split('.')[0] + '.txt')  #  写好输出文件的路径及名称
        out_txt_f = open(out_txt_path, 'w')  #  创建写入对象
        tree = ET.parse(xml_file)  #  将xml文档表示为树
        root = tree.getroot()  #  树的根目录
        w = int(root.find('Img_SizeWidth').text)  #  找到size节点下的width节点并得到第一个子元素之前的文本
        h = int(root.find('Img_SizeHeight').text)  #  找到size节点下的height节点并得到第一个子元素之前的文本
        # hrsc_objects = root.find('HRSC_Objects')

        for obj in root.iter('HRSC_Object'):  #  遍历整个root节点的HRSC_Object节点
            # difficult = obj.find('difficult').text  # 读取difficult
            cls = obj.find('Class_ID').text  # 读取name
            # if int(difficult) == 1:  # 如果类不属于或者目标困难, 则不标注
            #     continue
            if cls in classes1:
                cls_id = 0  # 找到该类对应列表的编号
            elif cls in classes2:
                cls_id = 1
            elif cls in classes3:
                cls_id = 2
            elif cls in classes4:
                cls_id = 3
            # 读取文件中的xywh
            b = (float(obj.find('box_xmin').text), float(obj.find('box_xmax').text), float(obj.find('box_ymin').text),
                 float(obj.find('box_ymax').text))
            # b=(xmin, xmax, ymin, ymax)
            print(w, h, b)
            bb = convert((w, h), b)  # 转换
            out_txt_f.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')


if __name__ == "__main__":
    # 需要转换的类别，需要一一对应
    classes1 = ['100000002', '100000005', '100000006', '100000012', '100000013', '100000031', '100000032',
                '100000033']  # 航母 Aircraft carrier
    classes2 = ['100000003', '100000007', '100000008', '100000009', '100000010', '100000011', '100000014',
                '100000015', '100000016', '100000017', '100000019', '100000028']  # 军舰 warcraft
    classes3 = ['100000001', '100000004', '100000018', '100000020', '100000022', '100000024', '100000025',
                '100000026', '100000029', '100000030']  # 商船 merchant
    classes4 = ['100000027']  # 潜艇 submarine
    # 2、voc格式的xml标签文件路径
    xml_files1 = r'D:\Remote Sensing Data\Object Detection\HRSC2016_dataset\HRSC2016\Test\Annotations'
    # 3、转化为yolo格式的txt标签文件存储路径
    save_txt_files1 = r'D:\Remote Sensing Data\Object Detection\HRSC2016_dataset\HRSC2016\Test\labels1'

    convert_annotation(xml_files1, save_txt_files1, classes1, classes2, classes3, classes4)