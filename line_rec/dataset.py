#coding: utf-8
from __future__ import print_function
import argparse
import random
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import numpy as np
from torch.nn import CTCLoss
import os
import sys 
import collections
import os, sys
import os.path as osp
from PIL import Image
import six
import string

import lmdb
import pickle
import editdistance

import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from torchvision.datasets import ImageFolder
from torchvision import transforms, datasets
import cv2
import traceback
import os
import sys
import re
import six
import math
import lmdb
import torch

import numpy as np
from torch.utils.data import Dataset, ConcatDataset, Subset
from torch._utils import _accumulate
import torchvision.transforms as transforms
import torch.utils.data.distributed
import torch.distributed as dist

from dsutils.brightness_aug import TenBrightAug, IncBrightAug, ColorAug, GrayImg, BinImg
from dsutils.distort_aug import DistortAug
from dsutils.resize_aug import ResizeAug
from dsutils.jpeg_aug import JPEGAug
from dsutils.real_image2lmdb import RealImageLMDB
from dsutils.real_image2lmdb_txt import RealImageTxtLMDB
from dsutils.synth_image2lmdb import SynthImageLMDB


from albumentations.pytorch import ToTensor
from albumentations import (
    CLAHE, Blur, OpticalDistortion, GridDistortion, ElasticTransform, Solarize, RandomBrightnessContrast, RandomBrightness, Cutout, InvertImg, RandomContrast, RandomGamma, OneOf, Compose, JpegCompression, RandomShadow, PadIfNeeded, ToGray
)

        
def collate_fn(batch):
    imgs, labels, pos = zip(*batch)
    return torch.stack(imgs), labels, pos




""" in fact it's unnecessary to write data prefetch myself, pytorch dataloader already done this """
""" I know this after I have writen this """
#class Batch_Balanced_Dataset(Thread):
class Batch_Balanced_Dataset():
    def __init__(self, opt):
#        super(Batch_Balanced_Dataset, self).__init__()
        self.tbaug = TenBrightAug()
        self.incbaug = IncBrightAug()
        self.colaug = ColorAug()
        self.distortaug = DistortAug()
        self.grayimg = GrayImg()
        self.binimg = BinImg()
        self.jpgaug = JPEGAug(0.8)
        self.resizeimg = ResizeAug(opt.imgW,opt.imgH)        
        self.resizeimg_test = ResizeAug(opt.imgW,opt.imgH, rand_scale=False)
        l1 = len(opt.train_data.strip().split(','))
        if l1 == 1:
            self.batch_sizes = [int(opt.batch_size)]
        elif l1 == 3:
            self.batch_sizes = [int(opt.batch_size * opt.batch_ratio), opt.batch_size - int(opt.batch_size * opt.batch_ratio)]
        elif l1 == 4:
            b1 = int(opt.batch_size * opt.batch_ratio2)
            self.batch_sizes = [int(b1 * opt.batch_ratio), 0, 0]
            self.batch_sizes[1] = b1 - self.batch_sizes[0]
            self.batch_sizes[2] = opt.batch_size - self.batch_sizes[0] - self.batch_sizes[1]

        if not opt.alldata:
            self.test_books = [
                '200021660',
                '200005598',
                'hnsd006',
                'umgy003',
                '100249416',
                'umgy011',
                'umgy010'
            ]
        else:
            print ('use all data')
            self.test_books = [
#                '200021660',
#               '200005598',
                'hnsd006',
                'umgy003',
                '100249416',
                'umgy011',
                'umgy010'
            ]            


        self.train_tf1 = transforms.Compose([self.distortaug,
                                             self.colaug,
                                             self.tbaug,
                                             self.incbaug,
                                             self.grayimg,
                                             self.binimg,
                                             self.tbaug,
                                             self.incbaug,
                                             self.jpgaug,
                                             self.resizeimg,
                                             transforms.ToTensor()
        ])


        self.train_tf0 = transforms.Compose([
            self.grayimg,
            self.resizeimg,
            transforms.ToTensor()
        ])

        self.train_tf00 = Compose([
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
        ])


        self.train_tf01 = Compose([
            JpegCompression(quality_lower = 20, quality_upper = 100),
            Cutout(num_holes=4, max_h_size=8, max_w_size=8),
            InvertImg(p=0.3)
        ])
        
        self.randtf = transforms.RandomApply([
            self.distortaug
        ])
        
        self.train_tf2 = transforms.Compose([
            self.resizeimg_test,
            transforms.ToTensor()
        ])
        
        self.data_loader_list = []
        self.datasets = []
        self.data_samplers = []
        self.opt = opt
        
        self.train_data = opt.train_data.strip().split(',')

        if not 'txt' in self.train_data[0]:
            self.datasets.append(RealImageLMDB(self.train_data[0], transform = self.train_tf0, transform2 = self.train_tf00, testBooks=self.test_books, character = opt.character, max_batch_length = opt.batch_max_length))
        else:
            self.datasets.append(RealImageTxtLMDB(self.train_data[0], transform = self.train_tf0, transform2 = self.train_tf00, testBooks=self.test_books, max_batch_length = opt.batch_max_length))            
        
        if len(self.train_data) == 3:
            self.datasets.append(SynthImageLMDB(self.train_data[1], self.train_data[2], transform = self.train_tf2, transform2 = self.train_tf01, size = (opt.imgH, opt.imgW), gray_bin_ratio = 0.5, max_label_length = opt.batch_max_length))

        if len(self.train_data) == 4:
            self.datasets.append(RealImageTxtLMDB(self.train_data[3], transform = self.train_tf0, transform2 = self.train_tf00, testBooks=self.test_books, max_batch_length = opt.batch_max_length))
            
        for i in range(len(self.datasets)):
            train_sampler = torch.utils.data.distributed.DistributedSampler(self.datasets[i])
            self.data_samplers.append(train_sampler)
            self.data_loader_list.append(DataLoader(self.datasets[i], num_workers=int(opt.workers), shuffle=False, sampler=train_sampler, batch_size=self.batch_sizes[i], collate_fn = collate_fn, pin_memory=True, drop_last = True))

        self.dataloader_iter_list = [iter(i) for i in self.data_loader_list]

        self.test_tf = transforms.Compose([self.grayimg,
                                           self.resizeimg_test,
                                           transforms.ToTensor()                         
                                          ])

        self.test_dataset = RealImageLMDB(self.train_data[0], self.test_tf, testBooks=self.test_books, isTest = True, character = opt.character)
        self.test_sampler = torch.utils.data.distributed.DistributedSampler(self.test_dataset)
        self.test_loader = DataLoader(self.test_dataset, num_workers=int(opt.workers), shuffle=False, sampler = self.test_sampler, batch_size=max(2, int(opt.batch_size/8)), collate_fn = collate_fn, drop_last = True)

    def sync_get_batch(self):
        balanced_batch_images = []
        balanced_batch_texts = []
        balanced_batch_poses = []
        for i, data_loader_iter in enumerate(self.dataloader_iter_list):
            try:
                image, text, pos = data_loader_iter.next()
                balanced_batch_images.append(image)
                balanced_batch_texts += text
                balanced_batch_poses += pos
            except StopIteration:
                self.dataloader_iter_list[i] = iter(self.data_loader_list[i])
                image, text, pos = self.dataloader_iter_list[i].next()
                balanced_batch_images.append(image)
                balanced_batch_texts += text
                balanced_batch_poses += pos
            except ValueError:
                traceback.print_exc()
                pass

        balanced_batch_images = torch.cat(balanced_batch_images, 0)
        
        return balanced_batch_images.to(self.opt.device), balanced_batch_texts, balanced_batch_poses

        
    def getValDataloader(self):
        return self.test_loader

