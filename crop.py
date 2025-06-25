import cv2
import os
import sys
import numpy as np
import glob
from multiprocessing import Pool
from functools import partial
import xml.etree.ElementTree as ET
from xml.dom.minidom import Document
from lxml import etree


def iou(BBGT, imgRect):
    """
    并不是真正的iou。计算每个BBGT和图像块所在矩形区域的交与BBGT本身的的面积之比，比值范围：0~1
    输入：BBGT：n个标注框，大小为n*4,每个标注框表示为[xmin,ymin,xmax,ymax]，类型为np.array
          imgRect：裁剪的图像块在原图上的位置，表示为[xmin,ymin,xmax,ymax]，类型为np.array
    返回：每个标注框与图像块的iou（并不是真正的iou），返回大小n,类型为np.array
    """
    left_top = np.maximum(BBGT[:, :2], imgRect[:2])  # 计算BBGT与图像块左上方坐标围成矩形的右下点坐标
    right_bottom = np.minimum(BBGT[:, 2:], imgRect[2:])  # 计算BBGT与图像块右下方坐标围成矩形的左上点坐标
    wh = np.maximum(right_bottom-left_top, 0)  # 计算差值，若BBGT与图像有交集，则两个值均为正
    inter_area = wh[:, 0]*wh[:, 1]  # 计算交集面积
    iou = inter_area/((BBGT[:, 2]-BBGT[:, 0])*(BBGT[:, 3]-BBGT[:, 1]))  # 交集面积除以BBGT总面积
    BB = np.concatenate((left_top, right_bottom), axis=1)  # 得到框在图像中的位置
    return iou, BB


def get_bbox(xml_path):
    '''
    得到xml下的类别与坐标
    :param xml_path:
    :return:
    '''
    BBGT = []
    tree = ET.parse(xml_path)  # 将xml文档表示为树
    root = tree.getroot()  # 树的根目录
    for obj in root.iter('object'):  # 迭代遍历整个root节点的object节点
        difficult = obj.find('difficult').text  # 读取difficult
        cls = obj.find('name').text  # 读取name
        xmlbox = obj.find('bndbox')  # 读取文件中的xywh
        xmin = int(float(xmlbox.find('xmin').text))
        ymin = int(float(xmlbox.find('ymin').text))
        xmax = int(float(xmlbox.find('xmax').text))
        ymax = int(float(xmlbox.find('ymax').text))
        label = cls
        BBGT.append([xmin, ymin, xmax, ymax, label])
    return np.array(BBGT)

def split(imgname, dirsrc, dirdst, subsize=800, gap=200, iou_thresh=0.3, ext='.png'):
    """
    imgname:   待裁切图像名（带扩展名）
    dirsrc:    待裁切的图像保存目录的上一个目录，默认图像与标注文件在一个文件夹下，图像在images下，标注在labelTxt下，标注文件格式为每行一个gt,
               格式为xmin,ymin,xmax,ymax,class,想读其他格式自己动手改
    dirdst:    裁切的图像保存目录的上一个目录，目录下有images,labelTxt两个目录分别保存裁切好的图像或者txt文件，
               保存的图像和txt文件名格式为 oriname_min_ymin.png(.txt),(xmin,ymin)为裁切图像在原图上的左上点坐标,txt格式和原文件格式相同
    subsize:   裁切图像的尺寸，默认为正方形，想裁切矩形自己动手改
    gap:       相邻行或列的图像重叠的宽度
    iou_thresh:小于该阈值的BBGT不会保存在对应图像的txt中（在图像过于边缘或与图像无交集）
    ext:       保存图像的格式
    """
    img_path = os.path.join(dirsrc, 'JPEGImages', imgname)
    # img = cv2.imread(os.path.join(os.path.join(dirsrc,'JPEGImages'), imgname), -1)  # 读取图片信息
    try:
        with open(img_path, "rb") as f:
            img = cv2.imdecode(np.frombuffer(f.read(), np.uint8), 0)
    except:
        img = None
    xml_path = os.path.join(os.path.join(dirsrc, 'Anotations'), imgname.split('.')[0]+'.xml')  # 得到xml文件路径
    BBGT = get_bbox(xml_path)

    img_h,img_w = img.shape[:2]  # 得到空间宽高
    top = 0  # 图片上方标量
    reachbottom = False  # 到达底部标记
    while not reachbottom:  # 若没到达底部
        reachright = False  # 到达最右标记
        left = 0  # 图片左边标量
        if top + subsize >= img_h:  # 若下方超界，则从下往上裁剪
            reachbottom = True
            top = max(img_h-subsize,0)
        while not reachright:  # 若没到达右边
            if left + subsize >= img_w:  # 若右方超界，则从右往左裁剪
                reachright = True
                left = max(img_w-subsize,0)
            imgsplit = img[top:min(top+subsize,img_h),left:min(left+subsize,img_w)]  # 裁剪图片
            if imgsplit.shape[:2] != (subsize,subsize):  # 若长度跟预设不等， 补充黑框
                try:  # 若为多通道
                    template = np.zeros((subsize, subsize, imgsplit.shape[2]), dtype=np.uint8)
                    template[0:imgsplit.shape[0], 0:imgsplit.shape[1]] = imgsplit
                    imgsplit = template
                except:  # 若为单通道
                    template = np.zeros((subsize,subsize),dtype=np.uint8)
                    template[0:imgsplit.shape[0],0:imgsplit.shape[1]] = imgsplit
                    imgsplit = template
            imgrect = np.array([left,top,min(left+subsize,img_w),min(top+subsize,img_h)]).astype('float32')  # 得到图片切割位置
            ious, X = iou(BBGT[:,:4].astype('float32'), imgrect)  # 输入坐标信息
            BB = np.concatenate((X, BBGT[:, 4:]), axis = 1)  # 得到框在图中的坐标
            BBpatch = BB[ious > iou_thresh]  # 比较iou阈值,选出大于阈值的目标
            ## abandaon images with 0 bboxes
            if len(BBpatch) > 0:
                # print(len(BBpatch))
                # cv2.imwrite(os.path.join(os.path.join(dirdst, 'JPEGImages'),
                #                          imgname.split('.')[0] + '_' + str(left) + '_' + str(top) + ext), imgsplit)  # 保存图片
                save_path = os.path.join(dirdst, 'JPEGImages', f"{imgname.split('.')[0]}_{left}_{top}{ext}")
                os.makedirs(os.path.dirname(save_path), exist_ok=True)  # 确保目录存在

                # 使用imencode
                success, buf = cv2.imencode(ext, imgsplit)
                if success:
                    with open(save_path, "wb") as f:
                        f.write(buf)
                xml = os.path.join(os.path.join(dirdst, 'Anotations'),
                                        imgname.split('.')[0] + '_' + str(left) + '_' + str(top) + '.xml')  # 得到保存标签体制
                ann = GEN_Annotations(dirsrc)
                try:
                    ann.set_size(imgsplit.shape[0], imgsplit.shape[1], imgsplit.shape[2])  # 写入size
                except:
                    ann.set_size(imgsplit.shape[0], imgsplit.shape[1], 1)
                for bb in BBpatch:
                    x1, y1, x2, y2, target_id = int(float(bb[0])) - left, int(float(bb[1])) - top, int(float(bb[2])) - left, int(float(bb[3])) - top, bb[4]  # 得到坐标
                    # target_id, x1, y1, x2, y2 = anno_info
                    label_name = target_id  # 得到类别
                    ann.add_pic_attr(label_name, x1, y1, x2, y2)  # 写入object
                ann.savefile(xml)  # 保存

            left += subsize-gap  # 右滑
        top += subsize-gap  # 下滑


class GEN_Annotations:
    '''
    创建一个对象，这个对象为xml的对象，能实现写位置，写大小，保存文件操作
    '''
    def __init__(self, filename):
        self.root = etree.Element("annotation")

        child1 = etree.SubElement(self.root, "folder")
        child1.text = "VOC2007"

        child2 = etree.SubElement(self.root, "filename")
        child2.text = filename

        child3 = etree.SubElement(self.root, "source")

        child4 = etree.SubElement(child3, "annotation")
        child4.text = "PASCAL VOC2007"
        child5 = etree.SubElement(child3, "database")
        child5.text = "Unknown"


    def set_size(self, witdh, height, channel):
        size = etree.SubElement(self.root, "size")
        widthn = etree.SubElement(size, "width")
        widthn.text = str(witdh)
        heightn = etree.SubElement(size, "height")
        heightn.text = str(height)
        channeln = etree.SubElement(size, "depth")
        channeln.text = str(channel)

    def savefile(self, filename):
        tree = etree.ElementTree(self.root)
        tree.write(filename, pretty_print=True, xml_declaration=False, encoding='utf-8')

    def add_pic_attr(self, label, xmin, ymin, xmax, ymax):
        object = etree.SubElement(self.root, "object")
        namen = etree.SubElement(object, "name")
        namen.text = label
        bndbox = etree.SubElement(object, "bndbox")
        xminn = etree.SubElement(bndbox, "xmin")
        xminn.text = str(xmin)
        yminn = etree.SubElement(bndbox, "ymin")
        yminn.text = str(ymin)
        xmaxn = etree.SubElement(bndbox, "xmax")
        xmaxn.text = str(xmax)
        ymaxn = etree.SubElement(bndbox, "ymax")
        ymaxn.text = str(ymax)

if __name__ == '__main__':
    import tqdm
    dirsrc= r'C:\Users\LazyShark\Desktop\data'      # 待裁剪图像所在目录的上级目录，图像在JPEGImages文件夹下，标注文件在Anotations下
    dirdst= dirsrc + '//' + 'split'   # 裁剪结果存放目录，格式和原图像目录一样
    if not os.path.exists(dirdst):  # 创建裁剪保存文件夹
        os.mkdir(dirdst)
    if not os.path.exists(os.path.join(dirdst, 'JPEGImages')):
        os.mkdir(os.path.join(dirdst, 'JPEGImages'))
    if not os.path.exists(os.path.join(dirdst, 'Anotations')):
        os.mkdir(os.path.join(dirdst, 'Anotations'))


    subsize = 640  # 裁剪后的图像大小
    gap = 192  # 允许重叠的大小
    iou_thresh = 0.4  # iou阈值
    ext = '.jpg'  # 保存图像尾缀

    imglist = glob.glob(f'{dirsrc}/JPEGImages/*.jpg')  # 得到文件夹下所有图片的路径
    imgnameList = [os.path.split(imgpath)[-1] for imgpath in imglist] 
    for imgname in tqdm.tqdm(imgnameList): 
        split(imgname, dirsrc, dirdst, subsize, gap, iou_thresh, ext)