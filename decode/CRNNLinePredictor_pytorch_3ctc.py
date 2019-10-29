#coding: utf-8
import sys
import numpy as np
import json
from collections import defaultdict, Counter
from resize_aug import ResizeAug
#from __future__ import print_function
import argparse
import random
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import numpy as np
#from torch_baidu_ctc import CTCLoss
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

from model import Model
import ctcdecode

model_file = 'crnn/best_norm_ED.pth'
#'crnn/32in_2xchannel_2lstm1024_skip_leaky142_0.9200856711317998'
label_file = 'crnn/label.4800'

class Predictor:
    alphabet = ['_' for i in range(4788)]
    alphabet_size = len(alphabet)   
    def __init__(self, opt):
        for l in open(label_file,'r').readlines():
            l = l.strip().split(' ')
            if l[0] == '4800':
                continue
            self.alphabet[int(l[0])] = l[1]


        opt.imgH = 32
        opt.imgW = 800
        opt.Transformation = 'None'
        opt.FeatureExtraction = 'ResNet'
        opt.input_channel=1
        opt.num_class=4787
        opt.output_channel=512
        opt.hidden_size = 512
        opt.dropout = 0.5
        opt.rnnlayers= 1
        opt.rnndropout=0
        opt.batch_max_length=40
        
        self.opt = opt
        if opt.lm is not None:
            self.lm_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), opt.lm)
            self.bm_decoder = ctcdecode.CTCBeamDecoder(self.alphabet, beam_width=opt.beam_width, num_processes = 16,
                                                       blank_id = 0, model_path=self.lm_path, alpha = opt.alpha, beta = opt.beta)
        else:
            self.bm_decoder = ctcdecode.CTCBeamDecoder(self.alphabet, beam_width=opt.beam_width, num_processes = 16,
                                                       blank_id = 0, alpha = opt.alpha, beta = opt.beta)
            
        
        self.net = Model(opt)
        # weight initialization
        for name, param in self.net.named_parameters():
            if 'localization_fc2' in name:
                print(f'Skip {name} as it is already initialized')
                continue
            try:
                if 'bias' in name:
                    init.constant_(param, 0.0)
                elif 'weight' in name:
                    init.kaiming_normal_(param)
            except Exception as e:  # for batchnorm.
                if 'weight' in name:
                    param.data.fill_(1)
                    continue

        # data parallel for multi-GPU
        self.net = torch.nn.DataParallel(self.net).cuda()
    
        self.net.load_state_dict(torch.load(opt.m))
        self.net.eval()
        self.trans = ResizeAug(800,32,rand_scale=False)
        self.toT =   transforms.ToTensor()

        
    def decodeimg(self, img):
        file_bytes = np.asarray(bytearray(img), dtype=np.uint8)
        img_data_ndarray1 = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
        img_data_ndarray = self.trans(img_data_ndarray1)
        if img_data_ndarray.shape != (32,800):
            print ('ERROR', img_data_ndarray.shape)
        w =img_data_ndarray1.shape[1] * 32 / img_data_ndarray1.shape[0]
        if w > 800:
            w = 800
        scale = img_data_ndarray1.shape[1] / w
            
        return True,img_data_ndarray, img_data_ndarray1, scale

    def greedy_ctc(self, ctc):
        #  collapse repeating characters
        arg_max = np.argmax(ctc, axis=1)
        repeat_filter = arg_max[1:] != arg_max[:-1]
        repeat_filter = np.concatenate([[True], repeat_filter])
        collapsed = arg_max[repeat_filter]

        # discard blank tokens (the blank is always last in the alphabet)
        blank_filter = np.where(collapsed < (self.alphabet_size - 1))[0]
        final_sequence = collapsed[blank_filter]
        full_decode = ''.join([self.alphabet[letter_idx] for letter_idx in final_sequence])
        return full_decode

    
    def softmax(self,x):
        """Compute softmax values for each sets of scores in x."""
        result=np.zeros_like(x)
        M,N = x.shape
        for m in range(M):
            S=np.sum(np.exp(x[m,:]))
            result[m,:]=np.exp(x[m,:])/S
        return result


    def predict_ctc(self, m, dbgflags):
        m1=[]
        for x in m:
            m1.append(self.toT(self.trans(x)))
        m = torch.stack(m1)
        m = m.cuda()
        text_for_pred = torch.LongTensor(len(m), 41).fill_(0).cuda()
        if 'Attn' in self.opt.Prediction:
            results = self.net(m, text_for_pred)[0].log_softmax(2)
        else:
            results = self.net(m, text_for_pred).log_softmax(2)
        # Calculate evaluation loss for CTC deocder.                                                                                     
        results = results.cpu().detach().numpy() # B X T X C
        return results


    def predict_ctcbs(self, m, dbgflags):
        m1=[]
        for x in m:
            m1.append(self.toT(self.trans(x)))
        m = torch.stack(m1)
        m = m.cuda()
        text_for_pred = torch.LongTensor(len(m), 41).fill_(0).cuda()
        if 'Attn' in self.opt.Prediction:
            results = self.net(m, text_for_pred)[0].softmax(2)
        else:
            results = self.net(m, text_for_pred).softmax(2)

        return results

    def dump_ctc(self, ctc):
        for l in range(len(ctc)):
            c = (-ctc[l]).argsort()
            c = c[:50]
            c = [self.alphabet[x] for x in c]
            c = ' '.join(c)
            print (c)
            
        
    def beam_ctc_pos(self, ctcs):
        beam_results, beam_scores, timesteps, out_seq_len = self.bm_decoder.decode(ctcs)
        beam_scores = beam_scores.cpu().numpy()
        ret = []
        for i in range(len(ctcs)):
            self.dump_ctc(ctcs[i])
            
            smallest_score= 1000
            idx = -1
            for j in range(self.opt.beam_width):
                if beam_scores[i][j] < smallest_score:
                    smallest_score = beam_scores[i][j]
                    idx = j
            ts = timesteps[i][idx].cpu().numpy()
            s = ''.join([self.alphabet[beam_results[i][idx][x]] for x in range(out_seq_len[i][idx])])
            pos = [[ts[x]*4, ts[x]*4] for x in range(out_seq_len[i][idx])]
#            print (pos)
            ret.append([s,pos, smallest_score])
        return ret


    def greedy_ctc_pos(self, ctc):
        arg_max = np.argmax(ctc, axis=1)
        ctc_argsort = np.argsort(ctc, axis=1)
        text = ''
        beg = -1
        pos = []
        cur = ''
        for i in range(ctc.shape[0]):
            if arg_max[i] == 0:
                if beg >= 0:
                    text += cur
                    pos.append([beg, i-1])
                    beg = -1
            elif i > 0 and arg_max[i] != arg_max[i-1]:
                if beg > -1:
                    text += cur
                    pos.append([beg, i-1])
                beg = i
                cur = self.alphabet[arg_max[i]]
            elif i > 0 and arg_max[i] == arg_max[i-1]:
                pass
            elif i == ctc.shape[0]- 1:
                if beg >= 0:
                    text += cur
                    pos.append([beg, i])
                break
            elif arg_max[i] != 0:
                cur = self.alphabet[arg_max[i]]
                beg = i

        scores = []
        post = 0
        for t in range(ctc.shape[0]):
            if len(pos) == 0:
                break
            break
            #adjust end
            argsort_idxes = ctc_argsort[t][::-1][:3]
            st = pos[post][0]
            idx = arg_max[st]
            if arg_max[t] == 0 and idx in argsort_idxes and pos[post][1] < t:
                pos[post][1] = t
            if post + 1 < len(pos) and t >= pos[post+1][0]:
                post += 1
                
             
        for i in range(len(pos)):
            pos[i][0] = (pos[i][0]) * 4
            pos[i][1] = (pos[i][1]) * 4
        return text, pos
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--f', type=str, required=True, help='img line')
    parser.add_argument('--m', type=str, required=True, help='model file')
    parser.add_argument('--lm', type=str, required=True, help='language model file')    
    parser.add_argument('--beam_width', type=int, default=100, help='beam width') 
    parser.add_argument('--alpha', type=float, default=1.5, help='alpha for language model')
    parser.add_argument('--beta', type=float, default=1, help='beta for sentence length')
    parser.add_argument('--SequenceModeling', type=str, default=None, help='BiLSTM|None, default is None')
    parser.add_argument('--Prediction', type=str, default='CTC_Attn', help='CTC|Attn|CTC_Attn')
    parser.add_argument('--output_channel', type=int, default=512, help='output_channel default 512')
    parser.add_argument('--hidden_size', type=int, default=512, help='hidden_size default 512')        

    opt = parser.parse_args()    
    torch.nn.Module.dump_patches = True
    f = open(opt.f, 'rb')
    data = f.read()
    p = Predictor(opt)
    f,data,data1,scale = p.decodeimg(data)
#    r = p.predict_ctc([data],False)
    rb = p.predict_ctcbs([data],False)
#    txt, pos = p.greedy_ctc_pos(r[0])
#    print (txt, pos)    
#    print (pos)
    txt_pos = p.beam_ctc_pos(rb)
    txt, pos = txt_pos[0][0], txt_pos[0][1]
    print (txt,pos)
    data = cv2.cvtColor(data, cv2.COLOR_GRAY2BGR)
    data1 = cv2.cvtColor(data1, cv2.COLOR_GRAY2BGR)
    for i in range(len(pos)):
#        print(pos[i][0], pos[i][1])
        if pos[i][0] == pos[i][1]:
            pos[i][1] +=2

        e = 10
        if i%2 == 0:
            e = 20
        cv2.line(data1,(int(scale * float(pos[i][0])),1),(int(scale * pos[i][0]), e),(255,0,0),2)
        cv2.line(data1,(int(scale * pos[i][0]),e),(int(scale * pos[i][1]),e),(0,0,255),2)
        cv2.line(data1,(int(scale * pos[i][1]),e+1),(int(scale * pos[i][1]),data1.shape[0] - 1),(0,0,255),2)
            
        cv2.line(data,(int(1 * pos[i][0]),1),(int(1 * pos[i][0]),data.shape[0] - 1 ),(255,0,0),2)
        cv2.line(data,(int(1 * pos[i][1]),1),(int(1 * pos[i][1]),data.shape[0] - 1),(0,0,255),2)        

    cv2.imwrite('/home/linhui/crnn_line_test.jpg', data)
    cv2.imwrite('/home/linhui/crnn_line_test1.jpg', data1)
    
