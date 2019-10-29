# -*- coding: utf-8 -*-

import cv2
import numpy as np
import glob
import os
import json

from multiprocessing import Pool

from collections import defaultdict

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

def addRLE(anno):
    print(f'anno id: {anno["id"]}')
    img_id = anno['image_id']
    img = train_images[img_id]['img'] # cv2 imread
    x, y, w, h = anno['bbox']
    mask = get_mask(img, x, y, w, h)
    dilated_mask = get_unique_mask(mask)
    rle_list = rle_encoding(dilated_mask)
    anno['iscrowd'] = 1
    anno['segmentation'] = {'counts':rle_list, 'size':[dilated_mask.shape[0], dilated_mask.shape[1]]}
    return anno

def rle_encoding(x):
    dots = np.where(x.T.flatten() == 255)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if (b>prev+1): run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return list(map(int, run_lengths))

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
    return cv2.drawContours(dilation, contours[0], 0, (255 , 255 , 255),thickness=cv2.FILLED)

imgid2imgname = {}
for im in train_json["images"]:
    imgid2imgname[im["id"]] = im['file_name']

imgid2annos = defaultdict(list)
for i, an in enumerate(train_json['annotations']):
    imgid2annos[an['image_id']].append(an)

def processImage(x):
    imgname, img, annos_in_img = x
    stuffmask = np.zeros((img.shape[0], img.shape[1]), np.uint8)
    for anno in annos_in_img:
        x, y, w, h = anno['bbox']
        mask = get_mask(img, x, y, w, h)
        dilated_mask = get_unique_mask(mask)
        stuffmask[dilated_mask > 0] = 1
    # save png
    pngname = imgname[:-4] + '.png'
    cv2.imwrite(os.path.join(kaggle_data_root, 'annotations/train_thingstuffmap/' + pngname), stuffmask)
    print(f'saved img: {pngname}')
    return imgname

tmp = []
for im in train_json['images']:
    imgid = im['id']
    img = im['img']
    imgname = im['file_name']
    annos = imgid2annos[imgid]
    tmp.append((imgname, img, annos))

pool = Pool(48)
results = pool.map(processImage, tmp)
pool.close()
pool.join()


print(f'Processed imgs: {len(results)}')






