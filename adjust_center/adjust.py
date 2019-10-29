#coding: utf-8
import cv2
import numpy as np
from multiprocessing import Pool, TimeoutError
import sys


def adhere_cloest_min(img, pred_x, pred_y, pixels_range = 10):
    col = img[pred_y - pixels_range: pred_y + pixels_range, pred_x]
    row = img[pred_y, pred_x - pixels_range:pred_x + pixels_range]
    col_i = np.argmin(col)
    row_i = np.argmin(row)
#    if col[col_i] < 50 and row[row_i] < 50:
    return pred_x - pixels_range + row_i, pred_y - pixels_range + col_i
#    else:
#        return pred_x, pred_y

def bin(img):
#    blurimg = cv2.GaussianBlur(img,(11,11),cv2.BORDER_DEFAULT) 
    ret, img = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return img

def adhere_cloest_black(img, pred_x, pred_y, pixels_range_i = 40, pixels_range_j = 20):
    i = 0
    while i < pixels_range_i and pred_x + i < img.shape[1]-1 and pred_x - i > 0 and img[pred_y, pred_x + i] > 0 and img[pred_y, pred_x - i] > 0:
        i += 1
    j = 0
    while j < pixels_range_j and pred_y + j < img.shape[0]-1 and pred_y - j > 0 and img[pred_y + j, pred_x] > 0 and img[pred_y - j, pred_x] > 0:
        j += 1
    if i < pixels_range_i:
        if  img[pred_y, pred_x + i]  == 0:
            pred_x = pred_x + i
        elif img[pred_y, pred_x - i]  == 0:
            pred_x = pred_x - i

    if j < pixels_range_j:
        if img[pred_y + j, pred_x] == 0:
            pred_y = pred_y + j
        elif img[pred_y - j, pred_x] == 0:
            pred_y = pred_y - j
    return pred_x, pred_y
        


def adjust_pred(imgfn_preds):
    imgfn_preds = imgfn_preds.split(',')
    
    if 'test' in imgfn_preds[0]:
        img = cv2.imread('../../data/test_imgs/' + imgfn_preds[0] + '.jpg', 0)
    else:
        img = cv2.imread('../../data/train_imgs/' + imgfn_preds[0] + '.jpg', 0)

    img = bin(img)
    if(len(imgfn_preds[1]) == 0):
        return imgfn_preds[0] + ',' + imgfn_preds[1]
    
    preds = imgfn_preds[1].split(' ')
    for i in range(int(len(preds)/3)):
        i_u = i * 3
        i_x = i_u + 1
        i_y = i_u + 2
        x = int(preds[i_x])
        y = int(preds[i_y])
        x, y = adhere_cloest_black(img, x, y)
        preds[i_x] = str(x)
        preds[i_y] = str(y)
    preds = ' '.join(preds)
    return  imgfn_preds[0] + ',' + preds
    

imgfn_preds = []
first = True
firstline = ''

for l in open(sys.argv[1], 'r').readlines():
    l = l.strip()
    if first:
        first = False
        firstline = l
        continue
    imgfn_preds.append(l)

pool = Pool(processes=16)# start 8 worker processes    
imgfn_preds = pool.map(adjust_pred, imgfn_preds)

print (firstline)
for l in imgfn_preds:
    print (l)
