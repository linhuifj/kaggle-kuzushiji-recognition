#coding: utf-8
import sys
from CRNNLinePredictor_pytorch_3ctc import Predictor
from multiprocessing import Pool, TimeoutError
import os.path
import argparse

abnormal_path = 'abnormal_list'
testlist_path = '../croplines/64/test.list'
vallist_path = '../croplines/64/val.list'

imglist = [ l.strip().split(' ')[0] for l in open(vallist_path,'r').readlines()]

parser = argparse.ArgumentParser()
parser.add_argument('--m', type=str, required=True, help='modelflie')
opt = parser.parse_args()
opt.lm = None
opt.beam_width=0
opt.alpha=0
opt.beta=0
p = Predictor(opt)

imgs = []
fnames = []
scales= []

def save_ctc(ctc, fname):
    f = open(fname, 'w')
    for i in range(200):
        f.write(' '.join([str(c) for c in list(ctc[i])]))
        f.write('\n')
    f.close()
    print >> sys.stderr, 'writing',fname
            

def decode_ctc(ctc_and_fname_scale):
    ctc = ctc_and_fname_scale[0]
    fname=ctc_and_fname_scale[1]
    scale = ctc_and_fname_scale[2]
    r,pos = p.greedy_ctc_pos(ctc)
    if pos is None:
        return None
    return (fname, len(pos), len(list(r)), r, 0, ','.join([str(int(x[0]*scale)) + ',' + str(int(x[1]*scale)) for x in pos]))


pool = Pool(processes=16)# start 8 worker processes

for l in imglist:
    print ('predicting', l.split('/')[-1], file = sys.stderr)
    f = open(l, 'rb')
    data = f.read()
    f,data,data1,scale = p.decodeimg(data)
    imgs.append(data)
    fnames.append(l)
    scales.append(scale)
    if len(imgs) >= 10:    
        ctcs = p.predict_ctc(imgs,False)
        ctcs_and_fnames_scales = []
        for i in range(len(imgs)):
            ctcs_and_fnames_scales.append([ctcs[i],fnames[i], scales[i]])
        rs = pool.map(decode_ctc, ctcs_and_fnames_scales)
        for r in rs:
            if r is None:
                continue
            print (r[0],r[1],r[2],r[3],r[4],r[5])
        imgs = []
        fnames = []
        scales = []
        
if len(imgs) > 0:
    ctcs = p.predict_ctc(imgs,False)
    ctcs_and_fnames_scales = []
    for i in range(len(imgs)):
        ctcs_and_fnames_scales.append([ctcs[i], fnames[i], scales[i] ])
    rs = pool.map(decode_ctc, ctcs_and_fnames_scales)
    for r in rs:
        if r is None:
            continue
        print (r[0],r[1],r[2],r[3],r[4],r[5])
    imgs = []
    fnames = []
    scales = []
