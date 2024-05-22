# # 根据txt名称列表文件转移图像等数据文件
# # 其中，txt文件内容为图片集合，路径必须一致
# """
# https://blog.csdn.net/chao_xy/article/details/130179886
# """
# import shutil
#
# file = open('你的数据集路径/ImageSets/Main/train.txt', 'r')
# number_list = file.readlines()
# for i in range(len(number_list)):
#     number_list[i] = number_list[i].strip()
# print(number_list)
#
# src_path = '你的数据集路径/JPEGImages/'  # 图像路径
# target_path = '你的数据集路径/train2017/'
# while True:
#     try:
#         for number in number_list:
#             shutil.move(src_path + number + '.jpg', target_path + number + '.jpg')  # 文件名
#     except:
#         break
# 根据txt名称列表文件转移图像等数据文件

"""
https://blog.csdn.net/chao_xy/article/details/130179886
"""
import os
import shutil

file = open(r'C:\Users\LazyShark\Desktop\data\GC10-DET_1\test.txt', 'r')
number_list = file.readlines()
for i in range(len(number_list)):
    number_list[i] = number_list[i].strip()
    number_list[i] = os.path.basename(number_list[i]).rsplit('.',1)[0]
print(number_list)

images_path = r'C:\Users\LazyShark\Desktop\data\GC10-DET_1\images' + '\\'  # 原始图像路径
save_images_path = r'C:\Users\LazyShark\Desktop\data\GC10-DET_1\data\images\test' + '\\'  # 保存图像路径
label_path = r'C:\Users\LazyShark\Desktop\data\GC10-DET_1\labels' + '\\'
save_label_path = r'C:\Users\LazyShark\Desktop\data\GC10-DET_1\data\labels\test' + '\\'
while True:
    try:
        for number in number_list:
            try:  # 若只有图片无标签则跳过
                shutil.move(images_path + number + '.jpg', save_images_path + number + '.jpg')  # 文件名
                shutil.move(label_path + number + '.txt', save_label_path + number + '.txt')  # 文件名
            except:
                continue
        break
    except:
        break