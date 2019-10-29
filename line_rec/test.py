import os
import time
import string
import argparse

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import numpy as np
from nltk.metrics.distance import edit_distance
import torch.distributed as dist

from utils import CTCLabelConverter, AttnLabelConverter, Averager
from model import Model

def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
#    rt /= dist.get_world_size()
    return rt

def reduce_var(var, opt):
    rt = torch.tensor(var).to(opt.device)
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
#    rt /= dist.get_world_size()
    return rt


def validation(model, criterion, evaluation_loader, converter, opt, converter2 = None):
    """ validation or evaluation """
    n_correct = 0
    n_correct_ctc = 0
    norm_ED = 0
    norm_ED_ctc = 0
    length_of_data = 0
    infer_time = 0
    valid_loss_avg = Averager()

    for i, (image_tensors, labels, pos) in enumerate(evaluation_loader):
        batch_size = image_tensors.size(0)
        length_of_data = length_of_data + batch_size
        image = image_tensors.to(opt.device)

        # For max length prediction
        length_for_pred = torch.IntTensor([opt.batch_max_length] * batch_size).to(opt.device)
        text_for_pred = torch.LongTensor(batch_size, opt.batch_max_length + 1).fill_(0).to(opt.device)

        text_for_loss, length_for_loss = converter.encode(labels, batch_max_length=opt.batch_max_length)

        start_time = time.time()
        if 'CTC' == opt.Prediction:
            preds = model(image, text_for_pred).log_softmax(2)
            
            forward_time = time.time() - start_time

            # Calculate evaluation loss for CTC deocder.
            preds_size = torch.IntTensor([preds.size(1)] * batch_size)
            preds = preds.permute(1, 0, 2)  # to use CTCloss format

            # To avoid ctc_loss issue, disabled cudnn for the computation of the ctc_loss
            # https://github.com/jpuigcerver/PyLaia/issues/16
            torch.backends.cudnn.enabled = False
            cost = criterion(preds, text_for_loss, preds_size, length_for_loss)
            torch.backends.cudnn.enabled = True

            # Select max probabilty (greedy decoding) then decode index to character
            _, preds_index = preds.max(2)
            preds_index = preds_index.transpose(1, 0).contiguous().view(-1)
            preds_str = converter.decode(preds_index.data, preds_size.data)
        else:
            if opt.Prediction == 'CTC_Attn':
                pp = model(image, text_for_pred, is_train=False)
                preds,alphas = pp[1]
                """ ctc preds here """
                preds_ctc = pp[0].log_softmax(2)
                # Calculate evaluation loss for CTC deocder.                                                                                                              
                preds_ctc_size = torch.IntTensor([preds_ctc.size(1)] * batch_size)
                preds_ctc = preds_ctc.permute(1, 0, 2)  # to use CTCloss format
                # Select max probabilty (greedy decoding) then decode index to character      
                _, preds_ctc_index = preds_ctc.max(2)
                preds_ctc_index = preds_ctc_index.transpose(1, 0).contiguous().view(-1)
                preds_ctc_str = converter2.decode(preds_ctc_index.data, preds_ctc_size.data)
            
            else:
                preds,alphas = model(image, text_for_pred, is_train=False)                
            forward_time = time.time() - start_time

            preds = preds[:, :text_for_loss.shape[1] - 1, :]
            target = text_for_loss[:, 1:]  # without [GO] Symbol
            cost = criterion(preds.contiguous().view(-1, preds.shape[-1]), target.contiguous().view(-1))
            # select max probabilty (greedy decoding) then decode index to character
            _, preds_index = preds.max(2)
            preds_str = converter.decode(preds_index, length_for_pred)
            oldlabels = labels
            labels = converter.decode(text_for_loss[:, 1:], length_for_loss)

        cost_r =reduce_tensor(cost)
        if i % 100 == 0 and opt.local_rank == 0:
            print ('test iter', i, ',forward_time', forward_time, ',loss=',cost_r)
        cost = cost_r
        infer_time += forward_time
        valid_loss_avg.add(cost)

        # calculate accuracy.
        for pred, gt in zip(preds_str, labels):
            if 'Attn' in opt.Prediction:
                pred = pred[:pred.find('[s]')]  # prune after "end of sentence" token ([s])
                gt = gt[:gt.find('[s]')]
            if pred == gt:
                n_correct += 1
            if len(gt) == 0:
                norm_ED += 1
            else:
                norm_ED += edit_distance(pred, gt) / len(gt)
            
        if 'CTC_Attn' in opt.Prediction:
            for pred_ctc, gt in zip(preds_ctc_str, oldlabels):
                if pred_ctc == gt:
                    n_correct_ctc += 1
                if len(gt) == 0:
                    norm_ED_ctc +=1
                else:
                    norm_ED_ctc += edit_distance(pred_ctc, gt) / len(gt)

    n_correct = reduce_var(n_correct, opt)
    n_correct_ctc = reduce_var(n_correct_ctc, opt)
    length_of_data = reduce_var(length_of_data, opt)
    norm_ED_ctc = reduce_var(norm_ED_ctc, opt)
    norm_ED = reduce_var(norm_ED, opt)

    accuracy = n_correct.float()*100 / float(length_of_data)
    accuracy_ctc = n_correct_ctc.float()*100 / float(length_of_data)
    
    print ('n_correct = ', n_correct, n_correct_ctc, 'length_of_data = ', length_of_data, 'accuracy = ', accuracy)

    if 'CTC_Attn' in opt.Prediction:
        return valid_loss_avg.val(), accuracy_ctc, accuracy, norm_ED_ctc, norm_ED, preds_str, labels, infer_time, length_of_data
    
    return valid_loss_avg.val(), accuracy, norm_ED, preds_str, labels, infer_time, length_of_data



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_data', required=True, help='path to evaluation dataset')
    parser.add_argument('--benchmark_all_eval', action='store_true', help='evaluate 10 benchmark evaluation datasets')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument('--batch_size', type=int, default=192, help='input batch size')
    parser.add_argument('--saved_model', required=True, help="path to saved_model to evaluation")
    """ Data processing """
    parser.add_argument('--batch_max_length', type=int, default=25, help='maximum-label-length')
    parser.add_argument('--imgH', type=int, default=32, help='the height of the input image')
    parser.add_argument('--imgW', type=int, default=100, help='the width of the input image')
    parser.add_argument('--rgb', action='store_true', help='use rgb input')
    parser.add_argument('--character', type=str, default='0123456789abcdefghijklmnopqrstuvwxyz', help='character label')
    parser.add_argument('--sensitive', action='store_true', help='for sensitive character mode')
    parser.add_argument('--PAD', action='store_true', help='whether to keep ratio then pad for image resize')
    parser.add_argument('--data_filtering_off', action='store_true', help='for data_filtering_off mode')
    """ Model Architecture """
    parser.add_argument('--Transformation', type=str, required=True, help='Transformation stage. None|TPS')
    parser.add_argument('--FeatureExtraction', type=str, required=True, help='FeatureExtraction stage. VGG|RCNN|ResNet')
    parser.add_argument('--SequenceModeling', type=str, required=True, help='SequenceModeling stage. None|BiLSTM')
    parser.add_argument('--Prediction', type=str, required=True, help='Prediction stage. CTC|Attn')
    parser.add_argument('--num_fiducial', type=int, default=20, help='number of fiducial points of TPS-STN')
    parser.add_argument('--input_channel', type=int, default=1, help='the number of input channel of Feature extractor')
    parser.add_argument('--output_channel', type=int, default=512,
                        help='the number of output channel of Feature extractor')
    parser.add_argument('--hidden_size', type=int, default=256, help='the size of the LSTM hidden state')

    opt = parser.parse_args()

    """ vocab / character number configuration """
    if opt.sensitive:
        opt.character = string.printable[:-6]  # same with ASTER setting (use 94 char).

    cudnn.benchmark = True
    cudnn.deterministic = True
    opt.num_gpu = torch.cuda.device_count()

    test(opt)
