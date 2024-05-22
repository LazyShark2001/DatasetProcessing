'''
为了方便实验对比结果，先将visdrone的验证集可视化出来。
代码是根据xml标签可视化的，需要将visdrone的txt标签转成xml
'''
import os
import os.path
import numpy as np
import xml.etree.ElementTree as xmlET
from PIL import Image, ImageDraw

#'1': 'people', '2': 'people','3': 'bicycle', '4': 'car', '5': 'car',
# 6':'others','7':'others','8':'others','9':'others','10': 'motor','11':'others'

# classes = ('crazing','inclusion','patches','pitted_surface','rolled-in_scale','scratches')
classes = ('youwu', 'jiaoban', 'huahen', 'danbaohua', 'fenbi', 'chacheyin', 'quebian', 'lousha', 'baohua', 'songruan')
# classes = ('airplane',)

#把下面的路径改为自己的路径即可
file_path_img = r'C:\Users\LazyShark\Desktop\data_RZB_split\data\data_crop\JPEGImages'  # 图片路径
file_path_xml = r'C:\Users\LazyShark\Desktop\data_RZB_split\data\data_crop\Anotations'  # xml文件路径
save_file_path = r'C:\Users\LazyShark\Desktop\data_RZB_split\data\data_crop\1'  # 保存图片文件夹

pathDir = os.listdir(file_path_xml)  # 将该文件夹下的所有文件名装进一个列表
for idx in range(len(pathDir)):
    filename = pathDir[idx] #xml文件名
    tree = xmlET.parse(os.path.join(file_path_xml, filename))#解析xml
    objs = tree.findall('object')  # 找到所有的object并返回列表
    num_objs = len(objs)
    boxes = np.zeros((num_objs, 5), dtype=np.uint16)

    for ix, obj in enumerate(objs):  #获得其索引和值
        bbox = obj.find('bndbox')
        # Make pixel indexes 0-based
        x1 = float(bbox.find('xmin').text)
        y1 = float(bbox.find('ymin').text)
        x2 = float(bbox.find('xmax').text)
        y2 = float(bbox.find('ymax').text)

        cla = obj.find('name').text
        label = classes.index(cla)

        boxes[ix, 0:4] = [x1, y1, x2, y2]  # 将boxex[ix, 0]到boxex[ix, 3]赋值
        boxes[ix, 4] = label

    image_name = os.path.splitext(filename)[0]  # 将文件名分割出来
    img = Image.open(os.path.join(file_path_img, image_name + '.jpg'))  #读取文件

    draw = ImageDraw.Draw(img)
    for ix in range(len(boxes)):
        xmin = int(boxes[ix, 0])
        ymin = int(boxes[ix, 1])
        xmax = int(boxes[ix, 2])
        ymax = int(boxes[ix, 3])
        # draw.text([xmin, ymin], classes[boxes[ix, 4]], (255, 0, 0))  # 文字的左上角

        # 改
        try:
            draw.rectangle([xmin, ymin, xmax, ymax], outline=(255, 0, 0))  # 绘制矩形
            from PIL import ImageFont
            font = ImageFont.truetype(r'C:\PycharmProjects\mygame\font\Arial.ttf', 20)
            draw.text([xmin, ymin], classes[boxes[ix, 4]], (255, 0, 0), font)  # 文字的左上角
            # 改
        except:
            draw.rectangle([xmin, ymin, xmax, ymax], outline=(255,))  # 绘制矩形
            from PIL import ImageFont
            font = ImageFont.truetype(r'C:\PycharmProjects\mygame\font\Arial.ttf', 20)
            draw.text([xmin, ymin], classes[boxes[ix, 4]], (255,), font)  # 文字的左上角
            # 改

    img.save(os.path.join(save_file_path, image_name + '.png'))
    print(image_name + '.png'+' 保存成功')
