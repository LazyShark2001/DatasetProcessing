import os, random, shutil


def moveimg(fileDir, tarDir, labelile):
    fileDir = fileDir + "\\"
    tarDir = tarDir + "\\"
    pathDir = os.listdir(fileDir)  # 取图片的原始路径

    for name in pathDir:
        name = name[:-4] + '.xml'
        try:
            # shutil.copy(tarDir + name, labelile + "\\" + name)  #复制
            shutil.move(tarDir + name, labelile + "\\" + name)  #移动
        except:
            print(name)
            continue

if __name__ == '__main__':

    fileDir = r"C:\Users\LazyShark\Desktop\RZB_NEW\aug\quebian\images"  # 图片文件夹路径         改
    tarDir = r'C:\Users\LazyShark\Desktop\RZB_NEW\aug\quebian\xmls'  # 标签路径       改
    labelfile = r'C:\Users\LazyShark\Desktop\RZB_NEW\aug\quebian\asd'  # 标签转移的目标文件夹
    moveimg(fileDir, tarDir, labelfile)
