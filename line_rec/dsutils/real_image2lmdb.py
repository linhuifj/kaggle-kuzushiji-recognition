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

class RealImageLMDB(data.Dataset):
    def __init__(self, db_path, transform=None, transform2 = None, testBooks=[], isTest = False, character = None, max_batch_length = 40):
        self.db_path = db_path
        self.env = lmdb.open(db_path, subdir=osp.isdir(db_path),
                             readonly=True, lock=False,
                             readahead=False, meminit=False)
        self.character = list(character)
        self.testBooks=testBooks
        
        with self.env.begin(write=False) as txn:
            self.length = pickle.loads(txn.get(b'__len__'))
            self.keys = pickle.loads(txn.get(b'__keys__'))
            keys2 = []
            for l in self.keys:
                l = l.strip()
                if isTest:
                    if not self.checkIsTest(l):
                        continue
                else:
                    if self.checkIsTest(l):
                        continue
                keys2.append(l.strip())
            self.keys = keys2
            self.length = len(keys2)
            #min(len(keys2), self.length)

        self.max_batch_length = max_batch_length
        self.transform = transform
        self.transform2 = transform2
        
    def checkIsTest(self, fname):
        for l in self.testBooks:
            if l in fname:
                return True
        return False
    
    def __getitem__(self, index):
        img, label = None, None
        env = self.env
        while True:
            with env.begin(write=False) as txn:
                byteflow = txn.get(self.keys[index].encode('utf-8'))
            unpacked = pickle.loads(byteflow)
            # load image
            img = unpacked[0]
            label = unpacked[1]
            if len(label) <= self.max_batch_length:
                break
            index = (index + 1) % self.length
            
        if self.character is not None:
            label = ''.join([self.character[int(i)-1] for i in label])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform2 is not None:
            augmented = self.transform2(image=img)
            img = augmented['image']
            
        if self.transform is not None:
            img = self.transform(img)
            
        return img,label, None

    def __len__(self):
        return self.length

    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.db_path + ')'

def collate_fn(batch):
    batch.sort(key=lambda x: len(x[1]), reverse=True)
    img, label = zip(*batch)
    pad_label = []
    lens = []
    max_len = len(label[0])
    for i in range(len(label)):
        temp_label = [4800] * max_len
        temp_label[:len(label[i])] = label[i]
        pad_label.append(temp_label)
        lens.append(len(label[i]))
    return torch.stack(img), torch.tensor(pad_label), torch.tensor(lens)

def folder2lmdb(img_label_list, root = 'data/train/', write_frequency=50):
    imgs = []
    labels = []
    keys = []
    for l in open(img_label_list, 'r').readlines():
        l = l.strip().split(' ')
        labels.append(l[1:])
        keys.append(l[0])
        imgpath = root + '/' + l[0] + '.jpg'
        imgs.append(imgpath)

    lmdb_path = "train_lines_data.lmdb"
    print("Generate LMDB to %s" % lmdb_path)
    isdir = os.path.isdir(lmdb_path)

    db = lmdb.open(lmdb_path, subdir=isdir,
                   map_size=1099511627776 * 2, readonly=False,
                   meminit=False, map_async=True)

    txn = db.begin(write=True)
    for idx in range(len(imgs)):
        imgpath = imgs[idx]
        label = labels[idx]
        label = np.asarray([int(x) for x in label], np.int)
        #read data
        img = cv2.imread(imgpath, cv2.IMREAD_UNCHANGED)
        height = img.shape[0] # keep original height
        width = img.shape[1]
        width = width * 64 / height
        dim = (int(width), 64)
        img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        txn.put(keys[idx].encode('utf-8'), pickle.dumps((img,label), pickle.HIGHEST_PROTOCOL))
        if idx % write_frequency == 0 and idx > 0:
            print("[%d/%d]" % (idx, len(labels)))
            txn.commit()
            txn = db.begin(write=True)
        
    # finish iterating through dataset
    txn.commit()
    with db.begin(write=True) as txn:
        txn.put(b'__keys__', pickle.dumps(keys))
        txn.put(b'__len__', pickle.dumps(len(imgs)))

    print("Flushing database ...")
    db.sync()
    db.close()


if __name__ == "__main__":
    make_data_flag = False
    if make_data_flag:
        folder2lmdb("64/train_lines.all.jpg2label")
    else:
        from distort_aug import DistortAug
        from brightness_aug import TenBrightAug, IncBrightAug, ColorAug, GrayImg, BinImg
        from resize_aug import ResizeAug
        from albumentations import Compose, RandomCrop, Normalize, HorizontalFlip, Resize
        from albumentations.pytorch import ToTensor
        from albumentations import (
            HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
            Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, ElasticTransform, HueSaturationValue,
            IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine,
            ToGray, PadIfNeeded, RandomShadow, ImageCompression, JpegCompression, GridDistortion, Solarize, RandomBrightnessContrast, RandomBrightness, Cutout, InvertImg, RandomContrast, RandomGamma, IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose
        )
        
        #test here
        tbaug = TenBrightAug()
        incbaug = IncBrightAug()
        colaug = ColorAug()
        distortaug = DistortAug()
        grayimg = GrayImg()
        binimg = BinImg()
        resizeimg = ResizeAug(800,32)
        
        tf = transforms.Compose([#distortaug,
#                                 colaug,
#                                 tbaug,
#                                 incbaug,
#                                 grayimg,
#                                 binimg,
#                                 tbaug,
#                                 incbaug,
                                 resizeimg
        ]
        )

        """
        OpticalDistortion(
                distort_limit=0.05,
                shift_limit=10,
                border_mode = cv2.BORDER_WRAP,
                p=0.5)
        """

        """
            ElasticTransform(p=0.5, alpha=1, sigma=50, alpha_affine = 0.2, border_mode=cv2.BORDER_CONSTANT)
        """

        """
            Cutout(num_holes=16, max_h_size=16, max_w_size=16)
        """

        """
            InvertImg()
        """

        """
            RandomContrast(limit = 0.5)

            RandomBrightness()
        """
        tf2 = Compose([
            OneOf([
                OpticalDistortion(distort_limit=0.05,
                                  shift_limit=10,
                                  border_mode = cv2.BORDER_WRAP,
                                  p=0.5),
                ElasticTransform(p=0.5, alpha=1, sigma=50, alpha_affine = 0.2, border_mode=cv2.BORDER_CONSTANT)
            ]),
                              
            OneOf([
                CLAHE(),
                Solarize(),
                RandomBrightness(),
                RandomContrast(limit = 0.2),
                RandomBrightnessContrast(),
            ]),

            JpegCompression(quality_lower = 20, quality_upper = 100),
            RandomShadow(num_shadows_lower=1, num_shadows_upper=2),            
            PadIfNeeded(min_height = 64, min_width = 100, border_mode=cv2.BORDER_CONSTANT, p = 0.5),            
            Cutout(num_holes=8, max_h_size=16, max_w_size=16),
            InvertImg(p=0.3),            
            ToGray()
#            GridDistortion(num_steps=10, distort_limit = (0, 0.1),  border_mode = cv2.BORDER_CONSTANT)
#               OneOf([
#                   GridDistortion(p=0.1),
#                   IAAPiecewiseAffine(p=0.3),
#               ], p=1),
        ]
        )
            
        charset = open('../data/characters','r').read()
        dataset = RealImageLMDB('../data/train_lh_xyl.lmdb', transform = tf, transform2 = tf2, character=charset)
        
        data_loader = DataLoader(dataset, num_workers=2, shuffle=True, batch_size=1, collate_fn = lambda x:x)
        for idx, data in enumerate(data_loader):
            plt.subplot(11,1,idx+1)
            plt.imshow(data[0][0])
            if idx >=10:
                break
        plt.show()
        plt.savefig('/home/linhui/aug.jpg')
        
#            plt.imshow(data[0])
