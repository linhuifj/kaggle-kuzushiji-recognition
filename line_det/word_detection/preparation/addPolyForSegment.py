# -*- coding: utf-8 -*-

import cv2
import numpy as np
import glob
import os
import json

from multiprocessing import Pool

###########################################
######### Change this to your path ########
###########################################
kaggle_data_root = '/path/to/data/root/'
train_json_path = os.path.join(kaggle_data_root, 'Kuzushiji_split_by_book_annotation_train.json')
train_img_root = os.path.join(kaggle_data_root, 'train_images/')

with open(train_json_path, 'r') as f:
    train_json = json.load(f)
print('train_json loaded.')


train_images = {}
for im in train_json['images']:
    train_images[im['id']] = im


for img_id, im in train_images.items():
    img = cv2.imread(os.path.join(train_img_root, im['file_name']))
    im['img'] = img
    print(f'img loaded: {img_id}')

def addPoly(anno):
    print(f'anno id: {anno["id"]}')
    img_id = anno['image_id']
    img = train_images[img_id]['img'] # cv2 imread
    x, y, w, h = anno['bbox']
    mask = get_mask(img, x, y, w, h)
    contours = get_unique_mask(mask)
    c = contours[0][0]
    segment = c.flatten().tolist()
    area = cv2.contourArea(c)
    anno['iscrowd'] = 0
    anno['segmentation'] = [segment]
    anno['area'] = area
    return anno


def get_mask(img, x, y, width, height):

    #load the cropped area and apply an Otsu threshold
    cropped_img = np.array(img[y:y+height,x:x+width,:])
    blurred_img = cv2.GaussianBlur(cropped_img,(5,5),0)
    img_gray = cv2.cvtColor(blurred_img, cv2.COLOR_BGR2GRAY)
    ret, otsu = cv2.threshold(img_gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    #Place back the cropped area into a mask with the original image size
    img_height, img_width = img.shape[:2]
    img_mask = np.full((img_height,img_width),0)
    img_mask[y:y+height,x:x+width] = otsu

    return img_mask

#Apply larger dilation until the mask is in one block
def get_unique_mask(cropped_mask):
    is_dilation_complete = False
    cropped_mask = cropped_mask.astype("uint8")

    #Check if the current mask embeds all features in one "polygon"
    contours= cv2.findContours(cropped_mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    if len(contours[0])==1:
        is_dilation_complete = True
        #just a bit of dilation to make the mask smoother
        kernel = np.ones((5,5),np.uint8)
        dilation = cv2.dilate(cropped_mask,kernel,iterations = 1)

    #Otherwise, let's dilate the mask until it embeds all features
    kernel_factor = 1
    while not is_dilation_complete:
        kernel_size = kernel_factor*2
        kernel = np.ones((kernel_size,kernel_size),np.uint8)
        dilation = cv2.dilate(cropped_mask,kernel,iterations = 1)

        contours= cv2.findContours(dilation,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        if len(contours[0])==1:
            is_dilation_complete = True

        kernel_factor+=1

    #Draw the contours so it fills potential holes in the masks
    #return cv2.drawContours(dilation, contours[0], 0, (255 , 255 , 255),thickness=cv2.FILLED)
    return contours

pool = Pool(48)
new_annos = pool.map(addPoly, train_json['annotations'])
pool.close()
pool.join()


with open(train_json_path, 'r') as f:
    origin_train_json = json.load(f)

# Construct new json
new_train_json = {}
new_train_json["annotations"] = new_annos
new_train_json["images"] = origin_train_json["images"]
new_train_json["categories"] = origin_train_json["categories"]
# save to disk
new_train_json_path = os.path.join(kaggle_data_root, 'Polys_Segment_Kuzushiji_split_by_book_annotation_train.json')

with open(new_train_json_path, 'w') as f:
    json.dump(new_train_json, f)





















