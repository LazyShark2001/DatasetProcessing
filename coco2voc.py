"""
    -*- coding: utf-8 -*-
    Time    : 2025/6/23 14:10
    Author  : LazyShark
    File    : coco2voc.py
"""
"""
不规则json转voc
"""
from lxml import etree, objectify
import os
import os
import json

def save_xml(file_name, save_folder, img_info, height, width, channel, bboxs_info):
    '''
    :param file_name:文件名
    :param save_folder:#保存的xml文件的结果
    :param height:图片的信息
    :param width:图片的宽度
    :param channel:通道
    :return:
    '''
    folder_name, img_name = img_info  # 得到图片的信息

    E = objectify.ElementMaker(annotate=False)

    anno_tree = E.annotation(
        E.folder(folder_name),
        E.filename(img_name),
        E.path(os.path.join(folder_name, img_name)),
        E.source(
            E.database('Unknown'),
        ),
        E.size(
            E.width(width),
            E.height(height),
            E.depth(channel)
        ),
        E.segmented(0),
    )

    labels, bboxs = bboxs_info  # 得到边框和标签信息
    for label, box in zip(labels, bboxs):
        anno_tree.append(
            E.object(
                E.name(label),
                E.pose('Unspecified'),
                E.truncated('0'),
                E.difficult('0'),
                E.bndbox(
                    E.xmin(box[0]),
                    E.ymin(box[1]),
                    E.xmax(box[2]),
                    E.ymax(box[3])
                )
            ))

    etree.ElementTree(anno_tree).write(os.path.join(save_folder, file_name), pretty_print=True, encoding='utf-8')

folder_path = r"C:\Users\LazyShark\Desktop\电力图纸"
save_folder = r"C:\Users\LazyShark\Desktop\1"
json_data_list = []
classes = set()
# 遍历文件夹中的所有文件
for filename in os.listdir(folder_path):
    # 检查文件是否是JSON文件
    if filename.endswith('.json'):
        file_path = os.path.join(folder_path, filename)

        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            json_data_list.append(data)

        file_name = filename[:-4]+'xml'
        img_info = (folder_path, data['imagePath'])

        labels = [i['label'] for i in data['shapes']]
        points = [[i['points'][0][0], i['points'][0][1], i['points'][1][0],i['points'][1][1]] for i in data['shapes']]
        classes = classes | set(labels)
        save_xml(file_name, save_folder, img_info, data['imageHeight'], data['imageWidth'], 1, (labels, points))

print(classes)







