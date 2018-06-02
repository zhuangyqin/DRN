#!/usr/bin/env python
# encoding: utf-8

"""
@author: ‘zyq‘
@license: Apache Licence 
@file: CountCityScapes.py
@time: 2017/10/12 10:10
"""
import itertools
import os
import numpy as np
from collections import namedtuple
from PIL import Image

def load_image_set_index(data_path, image_set_main_folder, image_set_sub_folder):
    """
    find out which indexes correspond to given image set
    :return: the indexes of given image set
    """

    # Collection all subfolders
    image_set_main_folder_path = os.path.join(data_path, image_set_main_folder, image_set_sub_folder)
    image_name_set = [filename for parent, dirname, filename in os.walk(image_set_main_folder_path)]
    image_name_set = list(itertools.chain.from_iterable(image_name_set))
    index_set = ['' for x in range(len(image_name_set))]
    valid_index_count = 0
    for i, image_name in enumerate(image_name_set):
        splited_name_set = image_name.split('_')
        ext_split = splited_name_set[len(splited_name_set) - 1].split('.')
        ext = ext_split[len(ext_split) - 1]
        if splited_name_set[len(splited_name_set) - 1] != 'flip.png' and ext == 'png':
            index_set[valid_index_count] = "_".join(splited_name_set[:len(splited_name_set) - 1])
            valid_index_count += 1

    return index_set[:valid_index_count]

if __name__ == "__main__":
    data_path = "../data/cityscapes"
    image_set_main_folder = "gtFine"
    image_set_sub_folder = "train"

    Label = namedtuple('Label', [
        'name',
        'id',
        'trainId',
        'category',
        'categoryId',
        'hasInstances',
        'ignoreInEval',
        'color',
    ])

    labels = [
        #       name       id    trainId   category            catId     hasInstances   ignoreInEval   color
        Label('unlabeled', 0, 255, 'void', 0, False, True, (0, 0, 0)),
        Label('ego vehicle', 1, 255, 'void', 0, False, True, (0, 0, 0)),
        Label('rectification border', 2, 255, 'void', 0, False, True, (0, 0, 0)),
        Label('out of roi', 3, 255, 'void', 0, False, True, (0, 0, 0)),
        Label('static', 4, 255, 'void', 0, False, True, (0, 0, 0)),
        Label('dynamic', 5, 255, 'void', 0, False, True, (111, 74, 0)),
        Label('ground', 6, 255, 'void', 0, False, True, (81, 0, 81)),
        Label('road', 7, 0, 'flat', 1, False, False, (128, 64, 128)),
        Label('sidewalk', 8, 1, 'flat', 1, False, False, (244, 35, 232)),
        Label('parking', 9, 255, 'flat', 1, False, True, (250, 170, 160)),
        Label('rail track', 10, 255, 'flat', 1, False, True, (230, 150, 140)),
        Label('building', 11, 2, 'construction', 2, False, False, (70, 70, 70)),
        Label('wall', 12, 3, 'construction', 2, False, False, (102, 102, 156)),
        Label('fence', 13, 4, 'construction', 2, False, False, (190, 153, 153)),
        Label('guard rail', 14, 255, 'construction', 2, False, True, (180, 165, 180)),
        Label('bridge', 15, 255, 'construction', 2, False, True, (150, 100, 100)),
        Label('tunnel', 16, 255, 'construction', 2, False, True, (150, 120, 90)),
        Label('pole', 17, 5, 'object', 3, False, False, (153, 153, 153)),
        Label('polegroup', 18, 255, 'object', 3, False, True, (153, 153, 153)),
        Label('traffic light', 19, 6, 'object', 3, False, False, (250, 170, 30)),
        Label('traffic sign', 20, 7, 'object', 3, False, False, (220, 220, 0)),
        Label('vegetation', 21, 8, 'nature', 4, False, False, (107, 142, 35)),
        Label('terrain', 22, 9, 'nature', 4, False, False, (152, 251, 152)),
        Label('sky', 23, 10, 'sky', 5, False, False, (70, 130, 180)),
        Label('person', 24, 11, 'human', 6, True, False, (220, 20, 60)),
        Label('rider', 25, 12, 'human', 6, True, False, (255, 0, 0)),
        Label('car', 26, 13, 'vehicle', 7, True, False, (0, 0, 142)),
        Label('truck', 27, 14, 'vehicle', 7, True, False, (0, 0, 70)),
        Label('bus', 28, 15, 'vehicle', 7, True, False, (0, 60, 100)),
        Label('caravan', 29, 255, 'vehicle', 7, True, True, (0, 0, 90)),
        Label('trailer', 30, 255, 'vehicle', 7, True, True, (0, 0, 110)),
        Label('train', 31, 16, 'vehicle', 7, True, False, (0, 80, 100)),
        Label('motorcycle', 32, 17, 'vehicle', 7, True, False, (0, 0, 230)),
        Label('bicycle', 33, 18, 'vehicle', 7, True, False, (119, 11, 32)),
        Label('license plate', -1, -1, 'vehicle', 7, False, True, (0, 0, 142)),
    ]

    ID2trainID_dic = {label.id: label for label in labels}

    ID2trainID = 255 * np.ones((256,),np.int32)
    for l in ID2trainID_dic:
        if l in (-1, 255):
            continue
        ID2trainID[l] = ID2trainID_dic[l].trainId

    indexs = load_image_set_index(data_path,image_set_main_folder,image_set_sub_folder)
    result = np.zeros((256))
    for index in indexs:
        index_folder = index.split('_')[0]
        image_file = os.path.join(data_path, image_set_main_folder, image_set_sub_folder, index_folder,
                                  index + '_labelIds.png')
        assert os.path.exists(image_file), 'Path does not exist: {}'.format(image_file)
        img = np.array(Image.open(image_file)).astype(np.uint8)
        img = ID2trainID[img]
        img = img.reshape((-1))
        result += np.bincount(img)

    print result
