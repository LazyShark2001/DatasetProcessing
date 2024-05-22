import os, shutil


def moveimg(fileDir, labelile):
    fileDir = fileDir + "\\"
    pathDir = os.listdir(fileDir)  # 取图片的原始路径
    for name in pathDir:
        y = False
        name = name[:-4] + '.txt'
        for i in range(1,6):
            s = f'_{i}.txt'
            if s in name:
                y = True
                break
            else:
                continue
        if not y:
            shutil.copy(fileDir + name, labelile + "\\" + name)  # 移动



if __name__ == '__main__':

    fileDir = r"C:\Users\LazyShark\Desktop\RZB_data\labels\val"  # 图片文件夹路径         改
    labelfile = r'C:\Users\LazyShark\Desktop\RZB\labels\train'  # 标签转移的目标文件夹
    moveimg(fileDir, labelfile)
