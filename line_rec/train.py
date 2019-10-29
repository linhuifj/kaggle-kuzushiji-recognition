import os
import sys
import time
import random
import string
import argparse

import torch
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.optim as optim
import torch.utils.data
import numpy as np

from utils import CTCLabelConverter, AttnLabelConverter, Averager
from dataset import Batch_Balanced_Dataset
from model import Model
from test import validation
from test import reduce_tensor
from alphaloss import alpha_loss
import traceback
#from torch.nn.parallel import DistributedDataParallel as DDP
from apex.parallel import DistributedDataParallel as DDP
from apex import amp, optimizers
from apex.fp16_utils import *
from apex.multi_tensor_apply import multi_tensor_applier



import torch.distributed as dist


def train(opt):
    print (opt.local_rank)
    opt.device = torch.device('cuda:{}'.format(opt.local_rank))
    device = opt.device
    """ dataset preparation """
    train_dataset = Batch_Balanced_Dataset(opt)
    
    valid_loader = train_dataset.getValDataloader()
    print('-' * 80)

    """ model configuration """
    if 'CTC' == opt.Prediction:
        converter = CTCLabelConverter(opt.character, opt)
    elif 'Attn' == opt.Prediction:
        converter = AttnLabelConverter(opt.character, opt)
    elif 'CTC_Attn' == opt.Prediction:
        converter = CTCLabelConverter(opt.character, opt),AttnLabelConverter(opt.character, opt)
        
    opt.num_class = len(opt.character)

    if opt.rgb:
        opt.input_channel = 3
    model = Model(opt)

    model.to(opt.device)
    
    print (model)
    print('model input parameters', opt.rgb, opt.imgH, opt.imgW, opt.num_fiducial, opt.input_channel, opt.output_channel,
          opt.hidden_size, opt.num_class, opt.batch_max_length, opt.Transformation, opt.FeatureExtraction,
          opt.SequenceModeling, opt.Prediction)

    # weight initialization
    for name, param in model.named_parameters():
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

    """ setup loss """
    if 'CTC' == opt.Prediction:
        criterion = torch.nn.CTCLoss(zero_infinity=True).to(device)
    elif 'Attn' == opt.Prediction:
        criterion = torch.nn.CrossEntropyLoss(ignore_index=0).to(device),torch.nn.MSELoss(reduction="sum").to(device)
        # ignore [GO] token = ignore index 0
    elif 'CTC_Attn' == opt.Prediction:
        criterion = torch.nn.CTCLoss(zero_infinity=True).to(device),torch.nn.CrossEntropyLoss(ignore_index=0).to(device),torch.nn.MSELoss(reduction='sum').to(device)
    
    # loss averager
    loss_avg = Averager()

    # filter that only require gradient decent
    filtered_parameters = []
    params_num = []
    for p in filter(lambda p: p.requires_grad, model.parameters()):
        filtered_parameters.append(p)
        params_num.append(np.prod(p.size()))
        
    if opt.local_rank == 0:
        print('Trainable params num : ', sum(params_num))
    # [print(name, p.numel()) for name, p in filter(lambda p: p[1].requires_grad, model.named_parameters())]

    # setup optimizer
    if opt.sgd:
        optimizer = optim.SGD(filtered_parameters, lr=opt.lr, momentum=0.9, weight_decay=opt.weight_decay)
    elif opt.adam:
        optimizer = optim.Adam(filtered_parameters, lr=opt.lr, betas=(opt.beta1, 0.999))
    else:
        optimizer = optim.Adadelta(filtered_parameters, lr=opt.lr, rho=opt.rho, eps=opt.eps)
    if opt.local_rank == 0:
        print("Optimizer:")
        print(optimizer)
        
    if opt.sync_bn:
        model = apex.parallel.convert_syncbn_model(model)

    if opt.amp > 1:
        model, optimizer = amp.initialize(model, optimizer, opt_level="O" + str(opt.amp),  keep_batchnorm_fp32=True, loss_scale="dynamic")
    else:
        model, optimizer = amp.initialize(model, optimizer, opt_level="O" + str(opt.amp))


    # data parallel for multi-GPU
    model = DDP(model)
    
    if opt.continue_model != '':
        print(f'loading pretrained model from {opt.continue_model}')
        try:
            model.load_state_dict(torch.load(opt.continue_model, map_location=torch.device('cuda', torch.cuda.current_device())))
        except:
            traceback.print_exc()
            print (f'COPYING pretrained model from {opt.continue_model}')
            pretrained_dict = torch.load(opt.continue_model, map_location=torch.device('cuda', torch.cuda.current_device()))
            
            model_dict = model.state_dict()            
            pretrained_dict2 = dict()
            for k, v in pretrained_dict.items():
                if opt.Prediction == 'Attn':
                    if 'module.Prediction_attn.' in k:
                        k = k.replace('module.Prediction_attn.', 'module.Prediction.')
                if k in model_dict and model_dict[k].shape == v.shape:
                    pretrained_dict2[k] = v
            
            model_dict.update(pretrained_dict2)
            model.load_state_dict(model_dict)

    model.train()    
            
    """ final options """
    with open(f'./saved_models/{opt.experiment_name}/opt.txt', 'a') as opt_file:
        opt_log = '------------ Options -------------\n'
        args = vars(opt)
        for k, v in args.items():
            opt_log += f'{str(k)}: {str(v)}\n'
        opt_log += '---------------------------------------\n'
        opt_log += str(model)
        print(opt_log)
        opt_file.write(opt_log)
    """ start training """
    start_iter = 0

    start_time = time.time()
    best_accuracy = -1
    best_norm_ED = 1e+6
    i = start_iter

    ct = opt.batch_mul
    model.zero_grad()

    dist.barrier()
    while(True):
        # train part
        start = time.time()
        image, labels, pos = train_dataset.sync_get_batch()
        end = time.time()
        data_t = end - start
 
        start = time.time()
        batch_size = image.size(0)

        if 'CTC' == opt.Prediction:
            text, length = converter.encode(labels, batch_max_length=opt.batch_max_length)
            preds = model(image, text).log_softmax(2)
            preds_size = torch.IntTensor([preds.size(1)] * batch_size).to(device)
            preds = preds.permute(1, 0, 2)  # to use CTCLoss format

            # To avoid ctc_loss issue, disabled cudnn for the computation of the ctc_loss
            # https://github.com/jpuigcerver/PyLaia/issues/16
            torch.backends.cudnn.enabled = False
            cost = criterion(preds, text, preds_size, length)
            torch.backends.cudnn.enabled = True
        elif 'Attn' == opt.Prediction:
            text, length = converter.encode(labels, batch_max_length=opt.batch_max_length)                       
            preds = model(image, text[:, :-1]) # align with Attention.forward
            preds_attn = preds[0]
            preds_alpha = preds[1]                        
            target = text[:, 1:]  # without [GO] Symbol
            cost = criterion[0](preds_attn.view(-1, preds_attn.shape[-1]), target.contiguous().view(-1))
            
            if opt.posreg_w > 0.001:
                cost_pos = alpha_loss(preds_alpha, pos, opt, criterion[1])
                print ('attn_cost = ',cost, 'pos_cost = ', cost_pos * opt.posreg_w)
                cost += opt.posreg_w * cost_pos
            else:
                print ('attn_cost = ',cost_attn)
                
        elif 'CTC_Attn' == opt.Prediction:
            text_ctc, length_ctc = converter[0].encode(labels, batch_max_length=opt.batch_max_length)
            text_attn, length_attn = converter[1].encode(labels, batch_max_length=opt.batch_max_length)
            """ ctc prediction and loss """

            #should input text_attn here
            preds = model(image, text_attn[:, :-1])
            preds_ctc = preds[0].log_softmax(2)
            
            preds_ctc_size = torch.IntTensor([preds_ctc.size(1)] * batch_size).to(device)
            preds_ctc = preds_ctc.permute(1, 0, 2)  # to use CTCLoss format

            # To avoid ctc_loss issue, disabled cudnn for the computation of the ctc_loss
            # https://github.com/jpuigcerver/PyLaia/issues/16
            
            torch.backends.cudnn.enabled = False
            cost_ctc = criterion[0](preds_ctc, text_ctc, preds_ctc_size, length_ctc)
            torch.backends.cudnn.enabled = True

            """ attention prediction and loss """
            preds_attn = preds[1][0] # align with Attention.forward
            preds_alpha = preds[1][1]
            
            target = text_attn[:, 1:]  # without [GO] Symbol

            cost_attn = criterion[1](preds_attn.view(-1, preds_attn.shape[-1]), target.contiguous().view(-1))

            cost = opt.ctc_attn_loss_ratio * cost_ctc + (1-opt.ctc_attn_loss_ratio) * cost_attn
            
            if opt.posreg_w > 0.001:
                cost_pos = alpha_loss(preds_alpha, pos, opt, criterion[2])
                cost += opt.posreg_w * cost_pos
                cost_ctc = reduce_tensor(cost_ctc)
                cost_attn = reduce_tensor(cost_attn)
                cost_pos = reduce_tensor(cost_pos)
                if opt.local_rank == 0:                    
                    print ('ctc_cost = ',cost_ctc, 'attn_cost = ',cost_attn, 'pos_cost = ', cost_pos * opt.posreg_w)
            else:
                cost_ctc = reduce_tensor(cost_ctc)
                cost_attn = reduce_tensor(cost_attn)                
                if opt.local_rank == 0:
                    print ('ctc_cost = ',cost_ctc, 'attn_cost = ',cost_attn)

        cost /= opt.batch_mul
        if opt.amp > 0:
            with amp.scale_loss(cost, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            cost.backward()

        """ https://github.com/davidlmorton/learning-rate-schedules/blob/master/increasing_batch_size_without_increasing_memory.ipynb """
        ct -= 1
        if ct == 0:
            if opt.amp > 0:
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), opt.grad_clip)
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip)  # gradient clipping with 5 (Default)            
            optimizer.step() 
            model.zero_grad()
            ct = opt.batch_mul
        else:
            continue

        train_t = time.time() - start
        cost = reduce_tensor(cost)
        loss_avg.add(cost)
        if opt.local_rank == 0:
            print('iter', i, 'loss =', cost, ', data_t=', data_t, ',train_t=', train_t, ', batchsz=', opt.batch_mul * opt.batch_size)
        sys.stdout.flush()
        # validation part
        if (i > 0 and i % opt.valInterval == 0) or (i == 0 and opt.continue_model != ''):
            elapsed_time = time.time() - start_time
            print(f'[{i}/{opt.num_iter}] Loss: {loss_avg.val():0.5f} elapsed_time: {elapsed_time:0.5f}')
            # for log
            with open(f'./saved_models/{opt.experiment_name}/log_train.txt', 'a') as log:
                log.write(f'[{i}/{opt.num_iter}] Loss: {loss_avg.val():0.5f} elapsed_time: {elapsed_time:0.5f}\n')
                loss_avg.reset()

                model.eval()
                with torch.no_grad():
                    if 'CTC_Attn' in opt.Prediction:
                        # we only count for attention accuracy, because ctc is used to help attention
                        valid_loss, current_accuracy_ctc, current_accuracy, current_norm_ED_ctc, current_norm_ED, preds, labels, infer_time, length_of_data = validation(
                            model, criterion[1], valid_loader, converter[1], opt, converter[0])
                    elif 'Attn' in opt.Prediction:
                        valid_loss, current_accuracy, current_norm_ED, preds, labels, infer_time, length_of_data = validation(
                            model, criterion[0], valid_loader, converter, opt)
                    else:
                        valid_loss, current_accuracy, current_norm_ED, preds, labels, infer_time, length_of_data = validation(
                            model, criterion, valid_loader, converter, opt)                                                
                model.train()

                for pred, gt in zip(preds[:10], labels[:10]):
                    if 'Attn' in opt.Prediction:
                        pred = pred[:pred.find('[s]')]
                        gt = gt[:gt.find('[s]')]
                    print(f'{pred:20s}, gt: {gt:20s},   {str(pred == gt)}')
                    log.write(f'{pred:20s}, gt: {gt:20s},   {str(pred == gt)}\n')

                valid_log = f'[{i}/{opt.num_iter}] valid loss: {valid_loss:0.5f}'
                valid_log += f' accuracy: {current_accuracy:0.3f}, norm_ED: {current_norm_ED:0.2f}'
                
                if 'CTC_Attn' in opt.Prediction:
                    valid_log += f' ctc_accuracy: {current_accuracy_ctc:0.3f}, ctc_norm_ED: {current_norm_ED_ctc:0.2f}'
                    current_accuracy = max(current_accuracy, current_accuracy_ctc)
                    current_norm_ED = min(current_norm_ED, current_norm_ED_ctc)

                if opt.local_rank == 0:
                    print(valid_log)
                    log.write(valid_log + '\n')

                    # keep best accuracy model
                    if current_accuracy > best_accuracy:
                        best_accuracy = current_accuracy
                        torch.save(model.state_dict(), f'./saved_models/{opt.experiment_name}/best_accuracy.pth')
                        torch.save(model, f'./saved_models/{opt.experiment_name}/best_accuracy.model')
                    if current_norm_ED < best_norm_ED:
                        best_norm_ED = current_norm_ED
                        torch.save(model.state_dict(), f'./saved_models/{opt.experiment_name}/best_norm_ED.pth') 
                        torch.save(model, f'./saved_models/{opt.experiment_name}/best_norm_ED.model')                   
                    best_model_log = f'best_accuracy: {best_accuracy:0.3f}, best_norm_ED: {best_norm_ED:0.2f}'
                    print(best_model_log)
                    log.write(best_model_log + '\n')

        # save model per iter.
        if (i + 1) % opt.save_interval == 0 and opt.local_rank == 0:
            torch.save(
                model.state_dict(), f'./saved_models/{opt.experiment_name}/iter_{i+1}.pth')

        if i == opt.num_iter:
            print('end the training')
            sys.exit()
        if opt.prof_iter > 0 and i > opt.prof_iter:
            sys.exit()
            
        i += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_name', help='Where to store logs and models')
    parser.add_argument('--train_data', required=True, help='path to training dataset')
    parser.add_argument('--manualSeed', type=int, default=1111, help='for random seed setting')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
    parser.add_argument('--batch_size', type=int, default=192, help='input batch size')
    parser.add_argument('--num_iter', type=int, default=300000, help='number of iterations to train for')
    parser.add_argument('--valInterval', type=int, default=2000, help='Interval between each validation')
    parser.add_argument('--continue_model', default='', help="path to model to continue training")
    parser.add_argument('--adam', action='store_true', help='Whether to use adam (default is Adadelta)')
    parser.add_argument('--sgd', action='store_true', help='Whether to use sgd (default is Adadelta)')
    parser.add_argument('--cuda_benchmark', action='store_true', default=True, help='if input size not vary, use this, else do not use it')    
    parser.add_argument('--amp', type=int, default = 0, help='amp optimization level, default 0')        
    parser.add_argument('--prof_iter', type=int, default = 0, help='run in profile mode, exit for prof_iter iterations')    
    parser.add_argument('--lr', type=float, default=1, help='learning rate, default=1.0 for Adadelta')
    parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam. default=0.9')
    parser.add_argument('--rho', type=float, default=0.95, help='decay rate rho for Adadelta. default=0.95')
    parser.add_argument('--eps', type=float, default=1e-8, help='eps for Adadelta. default=1e-8')
    parser.add_argument('--weight_decay', type=float, default=1e-8, help='weight_decay for sgd. default=5e-4')    
    parser.add_argument('--grad_clip', type=float, default=50, help='gradient clipping value. default=50')
    parser.add_argument('--batch_mul', type=int, default=1, help='batch multiplyer. default=1') 
    parser.add_argument('--posreg_w', type=float, default=0, help='weight for position regression')
    parser.add_argument('--save_interval', type=int, default=5e4, help='save model every interval')
    parser.add_argument('--alldata', action='store_true', help='whether use all data')
    parser.add_argument('--sync_bn', action='store_true', help='synchronize batchnorm')               

    """ Data processing """
    parser.add_argument('--batch_ratio', type=float, default=0.5,
                        help='assign ratio for each selected data in the batch')
    parser.add_argument('--batch_ratio2', type=float, default=0.8,
                        help='assign ratio for each selected data in the batch')    
    parser.add_argument('--total_data_usage_ratio', type=str, default='1.0',
                        help='total data usage ratio, this ratio is multiplied to total number of data.')
    parser.add_argument('--batch_max_length', type=int, default=40, help='maximum-label-length')
    parser.add_argument('--imgH', type=int, default=32, help='the height of the input image')
    parser.add_argument('--imgW', type=int, default=800, help='the width of the input image')
    parser.add_argument('--feat_step_size', type=int, default=201, help='feature_step_size, determined by model')    
    parser.add_argument('--rgb', action='store_true', help='use rgb input')
    parser.add_argument('--character', type=str, default='', help='character label')
    parser.add_argument('--sensitive', action='store_true', help='for sensitive character mode')
    parser.add_argument("--local_rank", default=0, type=int)
    
    parser.add_argument("--master_addr", default='tcp://127.0.0.1:12345', type=str)
    parser.add_argument("--world_size", default=1, type=int)        

    """ Model Architecture """
    parser.add_argument('--Transformation', type=str, required=True, help='Transformation stage. None|TPS')
    parser.add_argument('--FeatureExtraction', type=str, required=True, help='FeatureExtraction stage. VGG|RCNN|ResNet')
    parser.add_argument('--SequenceModeling', type=str, required=True, help='SequenceModeling stage. None|BiLSTM')
    parser.add_argument('--Prediction', type=str, required=True, help='Prediction stage. CTC|Attn|CTC_Attn')
    parser.add_argument('--num_fiducial', type=int, default=20, help='number of fiducial points of TPS-STN')
    parser.add_argument('--input_channel', type=int, default=1, help='the number of input channel of Feature extractor')
    parser.add_argument('--output_channel', type=int, default=512,
                        help='the number of output channel of Feature extractor')
    parser.add_argument('--hidden_size', type=int, default=256, help='the size of the LSTM hidden state')
    parser.add_argument('--ctc_attn_loss_ratio', type=float, default=0.2, help='loss = ctc_attn_loss_ratio * ctc_loss + (1-ctc_attn_loss_ratio)* attn_loss')
    parser.add_argument('--dropout', type=float, default=0, help='dropout after feature')
    parser.add_argument('--rnndropout', type=float, default=0, help='dropout for rnn, only useful when rnnlayers > 1')
    parser.add_argument('--rnnlayers', type=int, default=1, help='layers of blstm')    
    
    opt = parser.parse_args()

    if not opt.experiment_name:
        opt.experiment_name = f'{opt.Transformation}-{opt.FeatureExtraction}-{opt.SequenceModeling}-{opt.Prediction}'
        opt.experiment_name += f'-Seed{opt.manualSeed}'
        # print(opt.experiment_name)

    os.makedirs(f'./saved_models/{opt.experiment_name}', exist_ok=True)

    """ vocab / character number configuration """
    
    opt.character = open(opt.character).read().strip()

    """ Seed and GPU setting """
    # print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    np.random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    torch.cuda.manual_seed(opt.manualSeed)

    cudnn.benchmark = True
    cudnn.deterministic = True
    
    torch.cuda.set_device(opt.local_rank)
    # FOR DISTRIBUTED:  Initialize the backend.  torch.distributed.launch will provide
    # environment variables, and requires that you use init_method=`env://`.
    dist.init_process_group(backend='nccl',
                            init_method=opt.master_addr, rank=opt.local_rank, world_size = opt.world_size)
    
    train(opt)
