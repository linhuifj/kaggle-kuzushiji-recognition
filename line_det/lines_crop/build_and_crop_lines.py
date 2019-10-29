# -*- coding: utf-8 -*-

import os
import cv2
import pandas as pd
import numpy as np
import glob
from tqdm import tqdm
from multiprocessing import Pool
from TextProposalConnector import TextProposalConnector

###########################################
######### Change this to your path ########
###########################################
kaggle_data_root = '/path/to/data/root/'
kaggle_traincsv_path = os.path.join(kaggle_data_root, 'train.csv')
kaggle_unicode_translation_path = os.path.join(kaggle_data_root, 'unicode_translation.csv')

# Train set
anno_file_path = os.path.join(kaggle_data_root, 'train.csv')
img_root = os.path.join(kaggle_data_root, 'train_images/')
TRAIN_SET = True
line_imgs_save_root = os.path.join(kaggle_data_root, 'lines_img_train/')
line_txt_file_path = os.path.join(kaggle_data_root, 'lines_label_train.txt')

# # Test set
# anno_file_path = os.path.join(kaggle_data_root, 'htc_det_boxes.csv')
# img_root = os.path.join(kaggle_data_root, 'test_images/')
# TRAIN_SET = False
# line_imgs_save_root = os.path.join(kaggle_data_root, 'lines_img_test/')
# line_txt_file_path = os.path.join(kaggle_data_root, 'lines_pos_test.txt')


if not os.path.exists(line_imgs_save_root):
    os.makedirs(line_imgs_save_root)

class CharLabel:
    def __init__(self,x,y,w,h,char,uni, charint):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.cx = self.x + self.w//2
        self.cy = self.y + self.h//2
        self.char = char
        self.uni = uni
        self.charint = charint

def textlinesInOneImage(img, labels):
    tmp_strs = labels.strip().split(' ')
    text_proposals = []
    scores = []
    for i in range(len(tmp_strs) // 5):
        j = 5 * i
        x = int(round(float(tmp_strs[j + 1])))
        y = int(round(float(tmp_strs[j + 2])))
        w = int(round(float(tmp_strs[j + 3])))
        h = int(round(float(tmp_strs[j + 4])))
        score = 1
        text_proposals.append([y, x, y+h, x+w])
        scores.append(score)
    img_h, img_w = img.shape[:2]
    img_c = 3
    text_proposals_np = np.array(text_proposals)
    scores_np = np.array(scores)

    connector = TextProposalConnector()
    return connector.get_text_lines_vertical(text_proposals_np, scores_np, (img_h, img_w, img_c))

def crop_line_img(img, linetops, line_height, line_len):
    line_img = np.zeros((line_len, line_height), np.uint8)
    for i in range(len(linetops) // 2):
        j = 2*i
        ori_x, ori_y = linetops[j], linetops[j+1]
        # 当前的 y 对应了一排 x，分别填充相应像素
        # print(f'len(linetops)={len(linetops)}, j={j}, i={i}, line_len={line_len}, line_height={line_height}')
        # print(f'linetops[j]-line_height = {linetops[j]-line_height}, linetops[j] = {linetops[j]}')
        crop = img[ori_y, max(0, linetops[j]-line_height) : linetops[j]]
        line_img[i, line_height-len(crop):] = crop
    return line_img

def worker_build_and_crop_lines_for_1_img(target_img):
    labels_str = df[df['image_id']==target_img]['labels'].iloc[0]

    try:
        labels = np.array(labels_str.split(' ')).reshape(-1, 5)
    except:
        labels = []
    if len(labels) == 0:
        return []

    img = cv2.imread(os.path.join(img_root, df[df['image_id']==target_img]['image_id'].iloc[0] + '.jpg'))
    text_lines = textlinesInOneImage(img, labels_str)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    line_strs = []

    if TRAIN_SET:
        y2chars = [ [] for x in range(img.shape[0])]
        for codepoint, x, y, w, h in labels:
            x, y, w, h = int(x), int(y), int(w), int(h)
            char = unicode_map[codepoint]
            charint = unicode_map_ct[codepoint]
            a = CharLabel(x,y,w,h,char,codepoint, charint)
            y2chars[a.cy].append(a)

    for line_num, line in enumerate(text_lines):
        line_height = line[0] - line[6]
        line_len = line[3] - line[1]
        line_top_xs = [int(round(line[0] + i*(line[2] - line[0])/line_len)) for i in range(line_len)]
        line_top_ys = [line[1] + i for i in range(line_len)]
        linetops = list(map(int, np.rollaxis(np.array([line_top_xs, line_top_ys]), 1).reshape(-1).tolist()))
        line_name = target_img + '_l_' + str(line_num)

        if TRAIN_SET:
            chars = []
            # for lt in l['linetops']:
            for yi, y in enumerate(line_top_ys):
                for c in y2chars[int(y)]:
                # for c in y2chars[int(lt['y'])]:
                    # if c.cx > lt['x']-l['text_height'] and c.cx < lt['x']:
                    if c.cx > line_top_xs[yi] - line_height and c.cx < line_top_xs[yi]:
                        chars.append(c)
            chars = sorted(chars, key = lambda x: x.cy, reverse = False)
            codes= ' '.join([str(c.charint) for c in chars])
            texts = ''.join([c.char for c in chars])

        line_img = crop_line_img(img, linetops, line_height, line_len)
        # rotate 90 degree, counter clock
        line_img = cv2.transpose(line_img)
        line_img = cv2.flip(line_img, 0)

        # save line img
        cv2.imwrite(os.path.join(line_imgs_save_root, line_name+'.jpg'), line_img)

        # save line label
        if TRAIN_SET:
            # train set with text label
            line_str = f'{line_name} {texts}'
        else:
            # test and val set without text label
            line_str = f'{line_name} EMPTY#{line_height}#{",".join(list(map(str, linetops)))}'
        line_strs.append(line_str)

    return line_strs



if __name__ == '__main__':
    df = pd.read_csv(anno_file_path)

    if TRAIN_SET:
        df_train = pd.read_csv(kaggle_traincsv_path)
        unicode_map = {codepoint: char for codepoint, char in pd.read_csv(kaggle_unicode_translation_path).values}
        unicode_map_ct = {}
        with open(kaggle_unicode_translation_path, 'r') as f:
            for l in f.readlines():
                l = l.strip().split(',')
                unicode_map_ct[l[0]] = len(unicode_map_ct)

    img_names = list(map(lambda x: os.path.basename(x)[:-4], glob.glob(img_root + '*')))
    with Pool(40) as pool:
        all_line_strs = list(tqdm(pool.imap(worker_build_and_crop_lines_for_1_img, img_names), total=len(img_names)))

    # save line labels to txt file
    res_strs = []
    for tmp in all_line_strs:
        res_strs += tmp
    with open(line_txt_file_path, 'w') as f:
        f.write('\n'.join(res_strs))



# # target_img = '200014685-00010_1'
# # target_img = 'test_003aa33a'

# # img = cv2.imread(os.path.join(img_root, df['image_id'][100] + '.jpg'))
# img = cv2.imread(os.path.join(img_root, df[df['image_id']==target_img]['image_id'].iloc[0] + '.jpg'))
# textlinesInOneImage(img, df[df['image_id']==target_img]['labels'].iloc[0])

# # 可视化组行结果
# from matplotlib import pyplot as plt
# def imgshow(img):
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # plt.imshow(img)

# import matplotlib as mpl
# mpl.rcParams['figure.figsize'] = (20,20)

# def show_connected_lines(img, text_lines):
    # for line in text_lines:
# #         print(line)
        # points = np.array([[line[0], line[1]], [line[2], line[3]], [line[4], line[5]], [line[6], line[7]]])
# #         print(points)
        # cv2.polylines(img, [points], 1, (255, 0, 0), 4)
    # imgshow(img)

# show_connected_lines(img, textlinesInOneImage(img, df[df['image_id']==target_img]['labels'].iloc[0]))
