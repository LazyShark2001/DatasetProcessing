
# from ultralytics.models import RTDETR
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# if __name__ == '__main__':
#     # Load a model
#     model = (RTDETR(r"/home/hy/disk/dick1/project/xgy/rtdetr/runs/detect/train47_ASF_AFF/weights/best.pt"))

#     metrics = model.val(split='test',batch=4, data='DIOR/DIOR.yaml')
import os
import glob
import random


def transform(pic_dir, input_file, output_file):
    train_fileDir = os.path.join(input_file,'train')
    valid_fileDir = os.path.join(input_file,'val')
    test_fileDir = os.path.join(input_file,'test')
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
pic_dir = r'/home/hy/disk/dick1/project/xgy/rtdetr/DIOR/images'
input_file = r'/home/hy/disk/dick1/project/xgy/rtdetr/DIOR/labels'
output_file = r'/home/hy/disk/dick1/project/xgy/rtdetr/DIOR'
transform(pic_dir, input_file, output_file)