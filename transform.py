"""文件加路径"""


PATH = 'C:/Users/Admin123/Desktop/yolov5_master/NWPU-10/'  #chanhe the path firstly (PATH TO dataset)dddd

def changepath():
    for i in ['test','train','val']:
        path = PATH + 'ImageSets/Main/{}.txt'.format(i)  # 获取train.txt路径
        img_path = PATH + 'positive image set/'
        write_path=(PATH + 'ImageSets/{}_write.txt').format(i)
        with open(path, "r") as file:
            img_files = file.readlines()
            for j in range(len(img_files)):
                img_files[j] =  img_path + img_files[j].rstrip() + '.jpg'  #获得图片
        file.close()
        with open(write_path, "w") as file:  # 写入
            for j in range(len(img_files)):
                file.write(img_files[j]+'\n')
        file.close()

        # path = PATH + 'VEDAI/fold{}test.txt'.format(i)
        # img_path = PATH + 'VEDAI/images/'
        # write_path=PATH + 'VEDAI/fold{}test_write.txt'.format(i)
        # with open(path, "r") as file:
        #     img_files = file.readlines()
        #     for j in range(len(img_files)):
        #         img_files[j] =  img_path + img_files[j].rstrip()
        # file.close()
        # with open(write_path, "w") as file:
        #     for j in range(len(img_files)):
        #         file.write(img_files[j]+'\n')
        # file.close()

if __name__ == '__main__':
    changepath()