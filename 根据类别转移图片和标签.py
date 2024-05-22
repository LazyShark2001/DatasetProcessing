import xml.etree.ElementTree as ET
import os
import shutil


def moveimg_img(fileDir, tarDir, name):
    fileDir = fileDir + f"{os.sep}"
    tarDir = tarDir + f"{os.sep}"
    name = name + '.jpg'
    try:
        shutil.copy(fileDir + name, tarDir + f"{os.sep}" + name)  #复制
    except:
        print(f"转移{name}失败")


def moveimg_xml(fileDir, tarDir, name):
    fileDir = fileDir + f"{os.sep}"
    tarDir = tarDir + f"{os.sep}"
    name = name + '.xml'
    try:
        shutil.copy(fileDir + name, tarDir + f"{os.sep}" + name)  #复制
    except:
        print(f"转移{name}失败")


def convert_annotation(img_files_path, xml_files_path, save_img_path, save_xml_path, classes):
    """xml_files_path为xml文件的路径, save_files_path为转移xml文件的路径"""
    xml_files = os.listdir(xml_files_path)  #  读取xml文件路径下的所有文件，返回文件名列表
    xml_files = [f for f in xml_files if f.endswith('.xml')]  # 保留尾缀为.xml的文件
    for xml_name in xml_files:  #  遍历整个文件
        xml_file = os.path.join(xml_files_path, xml_name)  #  xml_file为改文件的具体路径
        name = xml_name.rsplit('.', 1)[0]  #  写好输出文件的路径及名称
        tree = ET.parse(xml_file)  #  将xml文档表示为树
        root = tree.getroot()  #  树的根目录

        for obj in root.iter('object'):  #  迭代遍历整个root节点的object节点
            cls = obj.find('name').text  # 读取name
            if cls not in classes:  # 如果类不属于或者目标困难, 则不标注
                continue
            moveimg_img(img_files_path, save_img_path, name)
            moveimg_xml(xml_files_path, save_xml_path, name)
            break
        print(f"已转移文件: {name}")


if __name__ == "__main__":
    classes = ['danbaohua']
    img_files_path = r'C:\Users\LazyShark\Desktop\RZB_NEW\image'
    xml_files_path = r'C:\Users\LazyShark\Desktop\RZB_NEW\Anotations'
    save_img_path = r'C:\Users\LazyShark\Desktop\RZB_NEW\aug\danbaohua\images'
    save_xml_path = r'C:\Users\LazyShark\Desktop\RZB_NEW\aug\danbaohua\xmls'

    convert_annotation(img_files_path, xml_files_path, save_img_path, save_xml_path, classes)




