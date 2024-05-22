# 将txt格式转换成xml格式数据集
from xml.dom.minidom import Document
import os
import cv2


def makexml(picPath, txtPath, xmlPath):  # txt所在文件夹路径，xml文件保存路径，图片所在文件夹路径
    """此函数用于将yolo格式txt标注文件转换为voc格式xml标注文件
    在自己的标注图片文件夹下建三个子文件夹，分别命名为picture、txt、xml
    """
    # 创建字典用来对类型进行转换，要与classes.txt文件中的类对应，且顺序要一致
    # dic = {  '0':'crazing','1':'inclusion','2':'patches','3':'pitted_surface','4':'rolled-in_scale','5':'scratches'}
    # dic = {  '0':'person','1':'bicycle','2':'car'}
    dic = {'0': 'youwu', '1': 'jiaoban', '2': 'huahen', '3': 'jiangban', '4': 'songjing',
           '5': 'cuwei', '6': 'other'}
    files = os.listdir(txtPath)
    a = 0
    for i, name in enumerate(files):
        a += 1
        xmlBuilder = Document()
        annotation = xmlBuilder.createElement("annotation")  # 创建annotation标签
        xmlBuilder.appendChild(annotation)
        txtFile = open(txtPath + name)
        txtList = txtFile.readlines()
        img = cv2.imread(picPath + name[0:-4] + ".jpg")  # 注意这里的图片后缀，.jpg/.png
        Pheight, Pwidth, Pdepth = img.shape

        folder = xmlBuilder.createElement("folder")  # folder标签
        foldercontent = xmlBuilder.createTextNode("datasetRGB")
        folder.appendChild(foldercontent)
        annotation.appendChild(folder)

        filename = xmlBuilder.createElement("filename")  # filename标签
        filenamecontent = xmlBuilder.createTextNode(name[0:-4] + ".jpg")
        filename.appendChild(filenamecontent)
        annotation.appendChild(filename)

        size = xmlBuilder.createElement("size")  # size标签
        width = xmlBuilder.createElement("width")  # size子标签width
        widthcontent = xmlBuilder.createTextNode(str(Pwidth))
        width.appendChild(widthcontent)
        size.appendChild(width)

        height = xmlBuilder.createElement("height")  # size子标签height
        heightcontent = xmlBuilder.createTextNode(str(Pheight))
        height.appendChild(heightcontent)
        size.appendChild(height)

        depth = xmlBuilder.createElement("depth")  # size子标签depth
        depthcontent = xmlBuilder.createTextNode(str(Pdepth))
        depth.appendChild(depthcontent)
        size.appendChild(depth)

        annotation.appendChild(size)
        b = 0
        for j in txtList:
            b += 1
            oneline = j.strip().split(" ")
            object = xmlBuilder.createElement("object")  # object 标签
            picname = xmlBuilder.createElement("name")  # name标签
            namecontent = xmlBuilder.createTextNode(dic[oneline[0]])
            picname.appendChild(namecontent)
            object.appendChild(picname)

            pose = xmlBuilder.createElement("pose")  # pose标签
            posecontent = xmlBuilder.createTextNode("Unspecified")
            pose.appendChild(posecontent)
            object.appendChild(pose)

            truncated = xmlBuilder.createElement("truncated")  # truncated标签
            truncatedContent = xmlBuilder.createTextNode("0")
            truncated.appendChild(truncatedContent)
            object.appendChild(truncated)

            difficult = xmlBuilder.createElement("difficult")  # difficult标签
            difficultcontent = xmlBuilder.createTextNode("0")
            difficult.appendChild(difficultcontent)
            object.appendChild(difficult)

            bndbox = xmlBuilder.createElement("bndbox")  # bndbox标签
            xmin = xmlBuilder.createElement("xmin")  # xmin标签
            mathData = int(((float(oneline[1])) * Pwidth + 1) - (float(oneline[3])) * 0.5 * Pwidth)
            xminContent = xmlBuilder.createTextNode(str(mathData))
            xmin.appendChild(xminContent)
            bndbox.appendChild(xmin)

            ymin = xmlBuilder.createElement("ymin")  # ymin标签
            mathData = int(((float(oneline[2])) * Pheight + 1) - (float(oneline[4])) * 0.5 * Pheight)
            yminContent = xmlBuilder.createTextNode(str(mathData))
            ymin.appendChild(yminContent)
            bndbox.appendChild(ymin)

            xmax = xmlBuilder.createElement("xmax")  # xmax标签
            mathData = int(((float(oneline[1])) * Pwidth + 1) + (float(oneline[3])) * 0.5 * Pwidth)
            xmaxContent = xmlBuilder.createTextNode(str(mathData))
            xmax.appendChild(xmaxContent)
            bndbox.appendChild(xmax)

            ymax = xmlBuilder.createElement("ymax")  # ymax标签
            mathData = int(((float(oneline[2])) * Pheight + 1) + (float(oneline[4])) * 0.5 * Pheight)
            ymaxContent = xmlBuilder.createTextNode(str(mathData))
            ymax.appendChild(ymaxContent)
            bndbox.appendChild(ymax)

            object.appendChild(bndbox)  # bndbox标签结束

            annotation.appendChild(object)

        f = open(xmlPath + name[0:-4] + ".xml", 'w')
        xmlBuilder.writexml(f, indent='\t', newl='\n', addindent='\t', encoding='utf-8')
        print('生成第%d个xml文件, 有%d个标签'%(a , b))
        f.close()


if __name__ == "__main__":
    picPath = r"C:\Users\LazyShark\Desktop\bupi\images" + "\\"  # 图片所在文件夹路径，后面的\\一定要带上
    txtPath = r"C:\Users\LazyShark\Desktop\bupi\labels" + "\\"# txt所在文件夹路径，后面的\\一定要带上
    xmlPath = r"C:\Users\LazyShark\Desktop\bupi\Annotation" + "\\"  # xml文件保存路径，后面的\\一定要带上
    makexml(picPath, txtPath, xmlPath)