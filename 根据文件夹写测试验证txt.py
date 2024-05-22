import os
import glob
import random


def transform(pic_dir, input_file, output_file):
    train_fileDir = input_file + "\\train\\"
    valid_fileDir = input_file + "\\val\\"
    test_fileDir = input_file + "\\test\\"
    train_pathDir = os.listdir(train_fileDir)
    valid_pathDir = os.listdir(valid_fileDir)
    test_pathDir = os.listdir(test_fileDir)


    ftrain = open(os.path.join(output_file, 'train.txt'), 'w')   # 返回上一级目录写文件
    fvalid = open(os.path.join(output_file, 'valid.txt'), 'w')
    ftest = open(os.path.join(output_file, 'test.txt'), 'w')

    for i in train_pathDir:
        ftrain.write(os.path.join(pic_dir, i[:-4]) + ".jpg\n")
    for j in valid_pathDir:
        fvalid.write(os.path.join(pic_dir, j[:-4]) + ".jpg\n")
    for k in test_pathDir:
        ftest.write(os.path.join(pic_dir, k[:-4]) + ".jpg\n")

    ftrain.close()
    fvalid.close()
    ftest.close()
pic_dir = r'C:\Users\Admin123\Desktop\flip_data\images'
input_file = r'C:\Users\Admin123\Desktop\flip_data\labels'
output_file = r'C:\Users\Admin123\Desktop\flip_data'
transform(pic_dir, input_file, output_file)



