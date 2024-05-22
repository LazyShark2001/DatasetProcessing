

import os, random, shutil


# def moveimg(fileDir, tarDir):
#     fileDir = fileDir + "\\"
#     pathDir = os.listdir(fileDir)  # 取图片的原始路径
#     filenumber = len(pathDir)
#     rate = 0.1  # 自定义抽取图片的比例，比方说100张抽10张，那就是0.1
#
#     picknumber = int(filenumber * rate)  # 按照rate比例从文件夹中取一定数量图片
#     sample = random.sample(pathDir, picknumber)  # 随机选取picknumber数量的样本图片
#     for name in sample:
#         shutil.move(fileDir + name, tarDir + "\\" + name)  #移动
#     return


# def moveimg(fileDir, tarDir):
#     # 自己模块  根据文件分
#     with open(r'C:\Users\Admin123\Desktop\SuperYOLO-main\dataset\VEDAI\fold01test.txt') as file_object:   #改
#         sample = file_object.readlines()
#     for name in sample:
#         name = name.rstrip() + '.JPG'           #改
#         print(name)
#         shutil.move(fileDir + name, tarDir + "\\" + name)        # 改
#     return
#     #自己模块


def moveimg(fileDir, tarDir):
    fileDir = fileDir + "\\"
    pathDir = os.listdir(fileDir)  # 取图片的原始路径
    filenumber = len(pathDir)
    rate = 0.2  # 自定义抽取图片的比例，比方说100张抽10张，那就是0.1

    picknumber = int(filenumber * rate)  # 按照rate比例从文件夹中取一定数量图片
    sample = random.sample(pathDir, picknumber)  # 随机选取picknumber数量的样本图片
    for name in sample:
        shutil.move(fileDir + name, tarDir + "\\" + name)  # 剪切
        #copy 复制
    return



def movelabel(file_list, file_label_train, file_label_val):
    for i in file_list:
        if i.endswith('.jpg'):               #改
            # filename = file_label_train + "\\" + i[:-4] + '.xml'  # 可以改成xml文件将’.txt‘改成'.xml'就可以了
            filename = file_label_train + "\\" + i[:-4] + '.txt'  # 可以改成xml文件将’.txt‘改成'.xml'就可以了
            file_val = file_label_val + "\\" + i[:-4] + '.txt'
            if os.path.exists(filename):
                shutil.move(filename, file_val)
                print(i + "处理成功！")


if __name__ == '__main__':
    # fileDir = r"D:\Remote Sensing Data\ships\All\All" + "\\"  # 源图片文件夹路径         改
    fileDir = r"C:\Users\LazyShark\Desktop\dataee\images\train"  # 源图片文件夹路径         改
    tarDir = r'C:\Users\LazyShark\Desktop\dataee\images\val'  # 图片移动到新的文件夹路径          改
    moveimg(fileDir, tarDir)
    file_list = os.listdir(tarDir)
    file_label_train = r"C:\Users\LazyShark\Desktop\dataee\labels\train"  # 源图片标签路径          改
    file_label_val = r"C:\Users\LazyShark\Desktop\dataee\labels\val"  # 标签     改
    # 移动到新的文件路径
    movelabel(file_list, file_label_train, file_label_val)
