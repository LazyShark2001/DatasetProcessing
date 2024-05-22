'''https://jingjing.blog.csdn.net/article/details/137162505'''
import json
import glob
import os
import shutil
from pathlib import Path
import numpy as np
from tqdm import tqdm


def make_folders(path='../out/'):
    # Create folders

    if os.path.exists(path):
        shutil.rmtree(path)  # delete output folder
    os.makedirs(path)  # make new output folder
    os.makedirs(path + os.sep + 'labels')  # make new labels folder
    os.makedirs(path + os.sep + 'images')  # make new labels folder
    return path


def convert_coco_json(json_dir='./annotations/'):
    jsons = glob.glob(json_dir + '*.json')

    # Import json
    for json_file in sorted(jsons):
        fn = 'out/labels/%s/' % Path(json_file).stem.replace('voc_', '')  # folder name
        fn_images = 'out/images/%s/' % Path(json_file).stem.replace('voc_', '')  # folder name
        os.makedirs(fn,exist_ok=True)
        os.makedirs(fn_images,exist_ok=True)
        with open(json_file) as f:
            data = json.load(f)
        print(fn)
        # Create image dict
        images = {'%g' % x['id']: x for x in data['images']}

        # Write labels file
        for x in tqdm(data['annotations'], desc='Annotations %s' % json_file):
            if x['iscrowd']:
                continue

            img = images['%g' % x['image_id']]
            h, w, f = img['height'], img['width'], img['file_name']
            file_path='./images/'+fn.split('/')[-2]+"/"+f
            # The Labelbox bounding box format is [top left x, top left y, width, height]
            box = np.array(x['bbox'], dtype=np.float64)
            box[:2] += box[2:] / 2  # xy top-left corner to center
            box[[0, 2]] /= w  # normalize x
            box[[1, 3]] /= h  # normalize y

            if (box[2] > 0.) and (box[3] > 0.):  # if w > 0 and h > 0
                with open(fn + Path(f).stem + '.txt', 'a') as file:
                    file.write('%g %.6f %.6f %.6f %.6f\n' % (x['category_id'] - 1, *box))
            file_path_t=fn_images+f
            print(file_path,file_path_t)
            shutil.copy(file_path,file_path_t)

convert_coco_json()
