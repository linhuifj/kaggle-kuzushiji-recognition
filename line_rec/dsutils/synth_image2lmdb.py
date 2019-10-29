import os, sys
import os.path as osp
from PIL import Image
import six
import string

import lmdb
import pickle

import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from torchvision.datasets import ImageFolder
from torchvision import transforms, datasets
import cv2
import matplotlib.pyplot as plt
import numpy as np
import time

def get_bounding_color(img):
    h, w = img.shape
    v = np.array(sorted(img.flatten()))[-2*(h+w):]
    return max(100, v.sum()/(2*h+2*w) + 1)

class SynthImageLMDB(data.Dataset):
    def __init__(self, db_path, txt_lines_path, transform=None, transform2 = None, size=(32,800), gray_bin_ratio = 0.5, max_label_length = 50):
        self.db_path = db_path
        self.env = lmdb.open(db_path, subdir=osp.isdir(db_path),
                             readonly=True, lock=False,
                             readahead=False, meminit=False)
        with self.env.begin(write=False) as txn:
            self.char2unicodes = pickle.loads(txn.get(b'char2unicodes'))
            self.unicode2imgs = pickle.loads(txn.get(b'unicode2imgs'))
        
        self.txt_lines = []
        for l in open(txt_lines_path, 'r').readlines():
            if len(l) > max_label_length:
                l = l[:max_label_length-1]
            self.txt_lines.append(l.strip())

        self.length = len(self.txt_lines)
        self.random_state = np.random.RandomState(1234)
        self.sz = size
        self.gray_ratio = gray_bin_ratio * 10
        self.img2bounding_color = dict()
        self.transform = transform
        self.transform2 = transform2        
        
    def __getitem__(self, index):
        txt = list(self.txt_lines[index])
        env = self.env
        imgs = []
        txt1 = []
        imgs_bg = [] #background of image
        gray_bin_idx = 0
        if self.random_state.randint(0, 10) < self.gray_ratio:
            gray_bin_idx = 1
        for c in txt:
            if c not in self.char2unicodes:
                continue
            uni = self.char2unicodes[c]
            if uni not in self.unicode2imgs:
                continue
            txt1.append(c)
            l = len(self.unicode2imgs[uni][gray_bin_idx])
            idx = self.random_state.randint(0,l)
            imgs.append(self.unicode2imgs[uni][gray_bin_idx][idx])
            imgs_bg.append(0)
            
        txt = ''.join(txt1)
        if len(imgs) == 0 or len(txt) == 0:
            return self.__getitem__(self.random_state.randint(0, self.__len__()))
        #sum of height
        hsum = 0
        #distance
        dists = [] 
        with env.begin(write=False) as txn:
            for i in range(len(imgs)):
                img_name = imgs[i].encode('utf-8')
                byteflow = txn.get(img_name)
                unpacked = pickle.loads(byteflow)
                    
                if unpacked.shape[1] > self.sz[0]:
                    new_w = self.sz[0] - self.random_state.randint(0,self.sz[0]/5)
                    scale = unpacked.shape[1]/new_w
                    new_h = unpacked.shape[0] / scale
                    unpacked = cv2.resize(unpacked, (int(new_w), int(new_h)), cv2.INTER_AREA)
                imgs[i] = unpacked
                
                if img_name not in self.img2bounding_color:
                    imgs_bg[i] = get_bounding_color(unpacked)
                    self.img2bounding_color[img_name] = imgs_bg[i]
                else:
                    imgs_bg[i] = self.img2bounding_color[img_name]
                    
                hsum += unpacked.shape[0]
                dist = self.random_state.randint(0,5)
                if dist > 1:
                    dist = self.random_state.randint(0,10)
                dists.append(dist)
                hsum += dist

        #get the middle color
        middle_col = 255
        if len(imgs_bg) > 0:
            middle_col = int(sorted(imgs_bg)[int(len(imgs_bg)/2)])
        for i in range(len(imgs)):
            imgs[i] = imgs[i].astype(np.float64)
            imgs[i] *= (middle_col / imgs_bg[i])
        
        img = np.random.normal(0, 15, (hsum,self.sz[0]))
        img += np.full((hsum,self.sz[0]),middle_col)
        
        st = 0
        for i in range(len(dists)):
            st += dists[i]
            et = st + imgs[i].shape[0]
            lst_range = self.sz[0] - 2 - imgs[i].shape[1]
            if lst_range > 2:
                lst = self.random_state.randint(2,lst_range)
            else:
                lst = max(0,int(lst_range/2))
            img[st:et,lst:lst + imgs[i].shape[1]] = imgs[i]
            st = et
            
        #because we add noise to img, so we need to avoid overflow of uint8
        img[img >254] = 255
        img[img < 1] = 0
                    
        img = img.swapaxes(-2,-1)[...,::-1,:]
        img = img.astype(np.uint8)

        scale = self.random_state.randint(-30,30)
        scale /= 100.0
        new_w = int(img.shape[1] * (1 + scale))
        if new_w > self.sz[1]:
            new_w = self.sz[1]

        scale = new_w / img.shape[1]

        cw = np.full((len(imgs),2), 0.0) #center and width
        last_end = 0
        
        for i in range(0, len(imgs)):
            cw[i][0] = last_end + dists[i] + imgs[i].shape[0]/2
            cw[i][1] = imgs[i].shape[0]
            last_end += dists[i] + imgs[i].shape[0]
        
        cw*=scale
        img = cv2.resize(img, (new_w, self.sz[0]))        
        
        img = img.astype(np.uint8)

        if self.transform2 is not None:
            augmented = self.transform2(image=img)
            img = augmented['image']

        if self.transform is not None:
            img = self.transform(img)
            
        """
        debug use
        for i in range(0, len(imgs)):
            cx = int(cw[i][0])
            w = cw[i][1]
            
            img[14:16,cx-1:cx+1] = 255
            img[:, int(cx + w/2)] = 255
            img[:, int(cx - w/2)] = 255           
        """
        return img, txt, cw
    
    def __len__(self):
        return self.length

    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.db_path + ')'

def folder2lmdb(img_label_list, root = './', write_frequency=50):
    unicode2imgs = dict()
    char2unicodes = dict()

    for l in open('../../data/unicode_translation.csv', 'r').readlines():
        l = l.strip().split(',')
        char2unicodes[l[1]] = l[0]
    imgpaths = []
    for l in open(img_label_list, 'r').readlines():
        imgpath = l.strip()
        l = l.strip().split('/')
        if l[1] not in unicode2imgs:
            unicode2imgs[l[1]] = [[],[]]            
        if l[0] == 'bin':
            unicode2imgs[l[1]][0].append(imgpath)
        elif l[0] == 'gray':
            unicode2imgs[l[1]][1].append(imgpath)
        imgpaths.append(imgpath)
        
    lmdb_path = "train_lines_data.lmdb"
    print("Generate LMDB to %s" % lmdb_path)
    isdir = os.path.isdir(lmdb_path)

    db = lmdb.open(lmdb_path, subdir=isdir,
                   map_size=1099511627776 * 2, readonly=False,
                   meminit=False, map_async=True)

    txn = db.begin(write=True)
    for idx in range(len(imgpaths)):
        imgpath = imgpaths[idx]
        img = cv2.imread(imgpath, 0)
        scale = max(img.shape[0], img.shape[1])/64
        if scale > 1:
            img = cv2.resize(img, (int(img.shape[1]/scale), int(img.shape[0]/scale)), cv2.INTER_AREA)
        txn.put(imgpath.encode('utf-8'), pickle.dumps(img), pickle.HIGHEST_PROTOCOL)
        if idx % write_frequency == 0 and idx > 0:
            print('%d/%d' % (idx,len(imgpaths)))
            txn.commit()
            txn = db.begin(write=True)
    # finish iterating through dataset
    txn.commit()
    with db.begin(write=True) as txn:
        txn.put(b'unicode2imgs', pickle.dumps(unicode2imgs))
        txn.put(b'char2unicodes', pickle.dumps(char2unicodes))

    print("Flushing database ...")
    db.sync()
    db.close()


if __name__ == "__main__":
    make_data_flag = False
    if make_data_flag:
        folder2lmdb("filelist")
    else:
        from distort_aug import DistortAug
        from brightness_aug import TenBrightAug, IncBrightAug, ColorAug, GrayImg, BinImg
        from resize_aug import ResizeAug
        from jpeg_aug import JPEGAug
        
        from albumentations.pytorch import ToTensor
        from albumentations import (
            CLAHE, Blur, OpticalDistortion, GridDistortion, ElasticTransform, Solarize, RandomBrightnessContrast, RandomBrightness, Cutout, InvertImg, RandomContrast, RandomGamma, OneOf, Compose, JpegCompression, RandomShadow, PadIfNeeded, ToGray
        )

        tbaug = TenBrightAug()
        incbaug = IncBrightAug()
        colaug = ColorAug()
        distortaug = DistortAug()
#        grayimg = GrayImg()
#        binimg = BinImg()
        resizeimg = ResizeAug(800,32)
        jpgaug = JPEGAug(1)
        
        randtf = transforms.RandomApply([
            distortaug,
            jpgaug
            ])

        tf01 = Compose([
            JpegCompression(quality_lower = 20, quality_upper = 100),
#            RandomShadow(num_shadows_lower=1, num_shadows_upper=2),
            Cutout(num_holes=8, max_h_size=8, max_w_size=8),
            InvertImg(p=0.3),
        ])

        tf = transforms.Compose([
#                                 distortaug,
#                                 colaug,
#                                 tbaug,
#                                 incbaug,
#                                 jpgaug,
#                                 grayimg,
#                                 binimg,
#                                 tbaug,
#                                 incbaug,
            randtf,
            resizeimg
        ]
        )
        
        dataset = SynthImageLMDB('../data/char_data.lmdb', '../data/alllines', transform = tf, transform2 = tf01)
        
        data_loader = DataLoader(dataset, num_workers=2, shuffle=True, batch_size=1, collate_fn = lambda x:x)
        for idx, data in enumerate(data_loader):
            plt.subplot(11,1,idx+1)
            plt.imshow(data[0][0],cmap='gray')
            cv2.imwrite('/home/linhui/test/' + data[0][1] + '.jpg', data[0][0])
            if idx > 9:
                break
        plt.show()
        plt.savefig('/home/linhui/aug.jpg')
        
#            plt.imshow(data[0])
