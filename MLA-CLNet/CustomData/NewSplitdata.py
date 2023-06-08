import os
import json
import pickle
import random

import matplotlib.pyplot as plt


def read_split_data(root: str, mode: str):

    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)

    # 遍历文件夹，一个文件夹对应一个类别
    class_names = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]
    # 排序，保证顺序一致
    class_names.sort()
    # 生成类别名称以及对应的数字索引
    class_indices = dict((k, v) for v, k in enumerate(class_names))
    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
    with open('../class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    train_images_path = []  # 存储训练集的所有图片路径
    train_images_label = []  # 存储训练集图片对应索引信息

    every_class_num = []  # 存储每个类别的样本总数
    # 遍历每个文件夹下的文件
    for cla in class_names:
        cla_path = os.path.join(root, cla)
        # 遍历获取supported支持的所有文件路径
        images = [os.path.join(root, cla, i) for i in os.listdir(cla_path)]
        # 获取该类别对应的索引
        image_class = class_indices[cla]
        # 记录该类别的样本数量
        every_class_num.append(len(images))


        for img_path in images:
                train_images_path.append(img_path)
                train_images_label.append(image_class)


    print("{} images were found in the {} dataset.".format(sum(every_class_num), mode))
    print("{} images for {}.".format(len(train_images_path), mode))



    return train_images_path, train_images_label




def write_pickle(list_info: list, file_name: str):
    with open(file_name, 'wb') as f:
        pickle.dump(list_info, f)


def read_pickle(file_name: str) -> list:
    with open(file_name, 'rb') as f:
        info_list = pickle.load(f)
        return info_list
