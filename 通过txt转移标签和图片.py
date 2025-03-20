import os, random, shutil


def moveimg(txtDir, savedir):
    with open(os.path.join(txtDir, 'train.txt')) as file_object:
        train_text = file_object.readlines()
    with open(os.path.join(txtDir, 'valid.txt')) as file_object:
        val_text = file_object.readlines()
    with open(os.path.join(txtDir, 'test.txt')) as file_object:
        test_text = file_object.readlines()

    for test in train_text:
        test = test.replace("\n", "")
        label_name = test.replace('images', 'labels').replace('.jpg', '.txt')
        save_image = os.path.join(savedir, 'images', 'train')
        save_label = os.path.join(savedir, 'labels', 'train')
        shutil.copy(test, save_image)  #复制
        shutil.copy(label_name, save_label)  # 复制

    for test in val_text:
        test = test.replace("\n", "")
        label_name = test.replace('images', 'labels').replace('.jpg', '.txt')
        save_image = os.path.join(savedir, 'images', 'val')
        save_label = os.path.join(savedir, 'labels', 'val')
        shutil.copy(test, save_image)  #复制
        shutil.copy(label_name, save_label)  # 复制

    for test in test_text:
        test = test.replace("\n", "")
        label_name = test.replace('images', 'labels').replace('.jpg', '.txt')
        save_image = os.path.join(savedir, 'images', 'test')
        save_label = os.path.join(savedir, 'labels', 'test')
        shutil.copy(test, save_image)  #复制
        shutil.copy(label_name, save_label)  # 复制

    # for test in train_text:
    #     test = test.replace("\n", "")
    #     test = os.path.basename(test).replace('.jpg', '.xml')
    #     shutil.copy(r'C:\Users\Admin123\Desktop\NEU-DET\annotations'+ '\\' + test, r'C:\Users\Admin123\Desktop\NEU_DET_AUG\data_NEU-DET\labels\xml')  #复制




if __name__ == '__main__':

    txtDir = r"C:\Users\LazyShark\Desktop\RZBYOLO\RZB"  # txt         改
    datasetfile = r'C:\Users\LazyShark\Desktop\dataset'  # 保存文件      路径
    moveimg(txtDir, datasetfile)

# with open(r'C:\Users\Admin123\Desktop\NEU-DET\train.txt') as file:
#     a = file.readlines()
#     for i in a:
#         i = i.replace("\n", "")
#         print(os.path.basename(i))
