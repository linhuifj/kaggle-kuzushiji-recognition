# -*- coding: utf-8 -*-

import random
import shutil
import json
import cv2
import os

###########################################
######### Change this to your path ########
###########################################
kaggle_data_root = '/path/to/data/root/'


ann_id_global = 1

def extract_info(line, img_id):
    global ann_id_global
    path = os.path.join(kaggle_data_root, 'train' + '_images/')
    line = line.strip()
    img_name, labels = line.split(',')
    labels = labels.split()

    file_name = img_name + '.jpg'
    file_path = path + '/' + file_name
    print(file_path)
    img = cv2.imread(file_path)
    height, width = img.shape[0], img.shape[1]
    annotations = []
    image_info = {'file_name':file_name, 'id':img_id, 'height':height, 'width':width}
    for i in range(len(labels)//5):
        j = 5*i
        ann_id = ann_id_global
        ann_id_global += 1
        bbox = list(map(int, labels[j+1:j+5]))
        category_id = 1
        iscrowd = 0
        w, h = bbox[2], bbox[3]
        area = w * h
        annotations.append({'id': ann_id, 'image_id':img_id, 'bbox':bbox, 'category_id':category_id, 'iscrowd': iscrowd, 'area': area})
    #print(annotations)
    return annotations, img_id, image_info



if __name__ == '__main__':

    # book names used as val set, like 'umgy-010'
    val_books = []

    #STAGE = 'train'
    train_path = 'train_images'
    val_path = 'val_images'

    lines = []
    with open(os.path.join(kaggle_data_root, "train.csv"), 'r') as f:
        lines = f.readlines()[1:]

    train_annotations = []
    val_annotations = []
    train_images = []
    val_images = []

    img_id_global = 1
    for l in lines:
        anno, image_id, image_info = extract_info(l, img_id_global)
        img_id_global += 1
        name = image_info["file_name"]
        if name.find('-') > -1:
            book = name.split('-')[0]
        else:
            book = name.split('_')[0]

        if book in val_books:
            val_annotations.extend(anno)
            val_images.append(image_info)
        else:
            train_annotations.extend(anno)
            train_images.append(image_info)
    categories = [{'id':1, 'name':'text'}]

    train_dict_to_save = {'annotations':train_annotations, 'images':train_images, 'categories':categories}
    val_dict_to_save = {'annotations':val_annotations, 'images':val_images, 'categories':categories}
    with open(os.path.join(kaggle_data_root, "Kuzushiji_split_by_book_annotation_train.json"), 'w') as f:
        json.dump(train_dict_to_save, f)
    # with open("Kuzushiji_split_by_book_annotation_val.json", 'w') as f:
        # json.dump(val_dict_to_save, f)
