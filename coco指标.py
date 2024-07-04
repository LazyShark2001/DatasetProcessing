# import argparse
# from pycocotools.coco import COCO
# from pycocotools.cocoeval import COCOeval
#
#
# def parse_opt():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--anno_json', type=str, default=r'C:\Users\LazyShark\Desktop\RZB\coco\annotations\instances_test2017.json', help='training model path')
#     parser.add_argument('--pred_json', type=str, default='2.json', help='data yaml path')
#
#     return parser.parse_known_args()[0]
#
# if __name__ == '__main__':
#     opt = parse_opt()
#     anno_json = opt.anno_json
#     pred_json = opt.pred_json
#
#     anno = COCO(anno_json)  # init annotations api
#     print(pred_json)
#     pred = anno.loadRes(pred_json)  # init predictions api
#     eval = COCOeval(anno, pred, 'bbox')
#     eval.evaluate()
#     eval.accumulate()
#     eval.summarize()
#
#
# import json
# import os
#
# # 读取源文件
#
# with open(r'C:\Users\LazyShark\Desktop\RZB\coco\annotations\instances_test2017.json', 'r') as file:
#     data_anno = json.load(file)
#
# loc = r'runs/detect/train42/predictions.json'
# with open(loc, 'r') as file:
#     data_pred = json.load(file)
#
# dic = {}
# for item in data_anno['images']:
#     dic[item['file_name']] = item
#
# # 对每个字典进行处理
# for item in data_pred:
#     # 如果字典中包含 'image_id' 键
#     if 'image_id' in item:
#         # 将 'image_id' 值转换为5位数形式，并在前面用0填充
#         item['image_id'] = dic[item['image_id'] + '.jpg']['id']  # 注意尾缀
#
# # 获取当前脚本所在目录
# current_dir = os.path.dirname(os.path.abspath(__file__))
# # 构建新文件的路径
# # new_file_path = os.path.join(current_dir, '2.json')
#
# # 将修改后的内容写入到新文件中
# with open(loc, 'w') as file:
#     json.dump(data_pred, file, indent=4)

# '''先运行下面的，将anno中的标签image_id统一，再运行上面'''
#
# import argparse
# from pycocotools.coco import COCO
# from pycocotools.cocoeval import COCOeval
#
#
# def parse_opt():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--anno_json', type=str, default=r'C:\Users\LazyShark\Desktop\RZB\coco\annotations\instances_test2017.json', help='training model path')
#     parser.add_argument('--pred_json', type=str, default='2.json', help='data yaml path')
#
#     return parser.parse_known_args()[0]
#
# if __name__ == '__main__':
#     opt = parse_opt()
#     anno_json = opt.anno_json
#     pred_json = opt.pred_json
#
#     anno = COCO(anno_json)  # init annotations api
#     print(pred_json)
#     pred = anno.loadRes(pred_json)  # init predictions api
#     eval = COCOeval(anno, pred, 'bbox')
#     eval.evaluate()
#     eval.accumulate()
#     eval.summarize()

#
# import json
# import os
#
# # 读取源文件
#
# with open(r'C:\Users\LazyShark\Desktop\RZB\coco\annotations\instances_test2017.json', 'r') as file:
#     data_anno = json.load(file)
#
#
# with open(r'runs/detect/train42/predictions.json', 'r') as file:
#     data_pred = json.load(file)
#
# dic = {}
# for item in data_anno['images']:
#     dic[item['file_name']] = item
#
# # 对每个字典进行处理
# for item in data_pred:
#     # 如果字典中包含 'image_id' 键
#     if 'image_id' in item:
#         # 将 'image_id' 值转换为5位数形式，并在前面用0填充
#         item['image_id'] = dic[item['image_id'] + '.jpg']['id']  # 注意尾缀
#
# # 获取当前脚本所在目录
# current_dir = os.path.dirname(os.path.abspath(__file__))
# # 构建新文件的路径
# new_file_path = os.path.join(current_dir, '2.json')
#
# # 将修改后的内容写入到新文件中
# with open(new_file_path, 'w') as file:
#     json.dump(data_pred, file, indent=4)







import argparse
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import json
import os

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--anno_json', type=str, default=r'C:\Users\LazyShark\Desktop\RZB\coco\annotations\instances_test2017.json', help='training model path')
    parser.add_argument('--pred_json', type=str, default='runs/detect/train4/predictions.json', help='data yaml path')
    parser.add_argument('--endswith', type=str, default='.jpg', help='swith')

    return parser.parse_known_args()[0]



if __name__ == '__main__':
    opt = parse_opt()
    anno_json = opt.anno_json
    pred_json = opt.pred_json
    endswith = opt.endswith
    # 读取源文件
    with open(anno_json, 'r') as file:
        data_anno = json.load(file)

    with open(pred_json, 'r') as file:
        data_pred = json.load(file)

    dic = {}
    for item in data_anno['images']:
        dic[item['file_name']] = item

    # 对每个字典进行处理
    for item in data_pred:
        # 如果字典中包含 'image_id' 键
        if 'image_id' in item:
            # 将 'image_id' 值转换为5位数形式，并在前面用0填充
            try:
                item['image_id'] = dic[item['image_id'] + endswith]['id']  # 注意尾缀
            except:
                print(item['image_id'])

    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 构建新文件的路径
    new_file_path = os.path.join(current_dir, 'cache_coco.json')

    with open(new_file_path, 'w') as file:
        json.dump(data_pred, file, indent=4)

    anno = COCO(anno_json)  # init annotations api
    pred = anno.loadRes(new_file_path)  # init predictions api
    eval = COCOeval(anno, pred, 'bbox')
    eval.evaluate()
    eval.accumulate()
    eval.summarize()

    os.remove(new_file_path)
