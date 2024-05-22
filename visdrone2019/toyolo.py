'''
Author: 刘鸿燕 13752614153@163.com
Date: 2022-05-09 14:05:05
LastEditors: 刘鸿燕 13752614153@163.com
LastEditTime: 2022-05-09 15:38:09
FilePath: \VisDrone2019\data_process\visDrone2019_txt2txt_yolo.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'__background__', # always index 0
           'ignored regions','pedestrian', 'people','bicycle','car','van','truck','tricycle','awning-tricycle',
           'bus','motor','others'

'''
import os
from pathlib import Path
from PIL import Image
from tqdm import tqdm

def visdrone2yolo(dir):
    def convert_box(size, box):
        #Convert VisDrone box to YOLO CxCywh box,坐标进行了归一化
        dw = 1. / size[0]
        dh = 1. / size[1]
        return (box[0] + box[2] / 2) * dw, (box[1] + box[3] / 2) * dh, box[2] * dw, box[3] * dh

    # (dir / 'labels').mkdir(parents=True, exist_ok=True)  # make labels directory
    (dir / 'Annotations_YOLO').mkdir(parents=True, exist_ok=True)  # make labels directory
    pbar = tqdm((dir / 'annotations').glob('*.txt'), desc=f'Converting {dir}')
    for f in pbar:
        img_size = Image.open((dir / 'images' / f.name).with_suffix('.jpg')).size
        lines = []
        with open(f, 'r') as file:  # read annotation.txt
            for row in [x.split(',') for x in file.read().strip().splitlines()]:
                if row[4] == '0':  # VisDrone 'ignored regions' class 0
                    continue
                cls = int(row[5]) - 1
                box = convert_box(img_size, tuple(map(int, row[:4])))
                lines.append(f"{cls} {' '.join(f'{x:.6f}' for x in box)}\n")
                with open(str(f).replace(os.sep + 'annotations' + os.sep, os.sep + 'Annotations_YOLO' + os.sep), 'w') as fl:
                    fl.writelines(lines)  # write label.txt


dir = Path(r'C:\Users\Admin123\Desktop\visdrone')  # dataset文件夹下Visdrone2019文件夹路径
# Convert
for d in 'VisDrone2019-DET-train', 'VisDrone2019-DET-val', 'VisDrone2019-DET-test-dev':
    visdrone2yolo(dir / d)  # convert VisDrone annotations to YOLO labels

