'''https://blog.csdn.net/weixin_46170504/article/details/136571546'''

import os
import json
import cv2
import random
import time
from PIL import Image

# 部分同学都用的autodl, 用antodl举例
# 使用绝对路径

# 数据集 txt格式-labels标签文件夹
txt_labels_path = '改为自己数据集val的标签所在路径'

# 数据集图片images文件夹
datasets_img_path = '改为自己数据集val的图片所在路径'
# 这里 voc 为数据集文件名字，可以改成自己的路径

# xx.json生成之后存放的地址
save_path = 'dataset/annotations/'  # 指定生成的json文件的存放路径

classes_txt = '/TXToCOCO/class.txt改为class.txt文件所在路径'

with open(classes_txt, 'r') as fr:
    lines1 = fr.readlines()

categories = []
for j, label in enumerate(lines1):
    label = label.strip()
    categories.append({'id': j, 'name': label, 'supercategory': 'None'})
print(categories)

write_json_context = dict()
write_json_context['info'] = {'description': 'For object detection', 'url': '', 'version': '', 'year': 2021,
                              'contributor': '', 'date_created': '2021'}
write_json_context['licenses'] = [{'id': 1, 'name': None, 'url': None}]
write_json_context['categories'] = categories
write_json_context['images'] = []
write_json_context['annotations'] = []

imageFileList = os.listdir(datasets_img_path)

for i, imageFile in enumerate(imageFileList):
    imagePath = os.path.join(datasets_img_path, imageFile)
    image = Image.open(imagePath)
    W, H = image.size

    img_context = {}
    img_context['file_name'] = imageFile
    img_context['height'] = H
    img_context['width'] = W
    print(f"Before conversion: {imageFile[:-4]}")
    img_context['id'] = imageFile[:-4]
    int_id = int(img_context['id'])
    print(f"After conversion: {img_context['id']}")
    img_context['license'] = 1
    img_context['color_url'] = ''
    img_context['flickr_url'] = ''
    write_json_context['images'].append(img_context)

    txtFile = imageFile[:-4] + '.txt'

    with open(os.path.join(txt_labels_path, txtFile), 'r') as fr:
        lines = fr.readlines()
    for j, line in enumerate(lines):
        bbox_dict = {}

        class_id, x, y, w, h = line.strip().split(' ')
        class_id, x, y, w, h = int(class_id), float(x), float(y), float(w), float(h)
        xmin = (x - w / 2) * W
        ymin = (y - h / 2) * H
        xmax = (x + w / 2) * W
        ymax = (y + h / 2) * H
        w = w * W
        h = h * H
        bbox_dict['id'] = i * 10000 + j
        bbox_dict['image_id'] = int(imageFile[:-4])
        bbox_dict['category_id'] = class_id
        bbox_dict['iscrowd'] = 0
        height, width = abs(ymax - ymin), abs(xmax - xmin)
        bbox_dict['area'] = height * width
        bbox_dict['bbox'] = [xmin, ymin, w, h]
        bbox_dict['segmentation'] = [[xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax]]
        write_json_context['annotations'].append(bbox_dict)

name = os.path.join(save_path, "instances_val2017" + '.json')
with open(name, 'w') as fw:
    json.dump(write_json_context, fw, indent=2)
print("ok")
