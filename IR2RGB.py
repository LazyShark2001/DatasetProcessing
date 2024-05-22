import glob
from PIL import Image

filepath = glob.glob(r'C:\Users\LazyShark\Desktop\data_RZB_split\data\JPEGImages\*.jpg')
for filename in filepath:
    img=Image.open(filename).convert("RGB")
    img.save(filename)  # 原地保存
    print('{}/{}'.format(filepath.index(filename) + 1,filepath.__len__()) + ', ' + filename + '保存完成')

