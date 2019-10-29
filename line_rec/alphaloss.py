import torch
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.optim as optim
import torch.utils.data
import numpy as np
import math

""" this method needs less computation than scipy """
def normpdf(x, mean, sd):
    var = float(sd)**2
    denom = (2*math.pi*var)**.5
    num = math.exp(-(float(x)-float(mean))**2/(2*var))
    return num/denom

def alpha_loss(alpha, pos_gt, opt, loss):
#    l2 = torch.nn.MSELoss(reduction = "sum")   
    alpha_dim = alpha.size(2)
    pos_gt_a = []
    pos_pred_a_idx = []
    wordct = 0
    for i in range(len(pos_gt)):
        if pos_gt[i] is not None:
            wordct += len(pos_gt[i])
            pos_gt_a.append(pos_gt[i])
            pos_pred_a_idx.append(i)
            
    alpha = alpha.index_select(0, torch.LongTensor(pos_pred_a_idx).to(opt.device))            
    batch_gt_alpha = np.full((len(pos_gt_a), opt.batch_max_length + 1, alpha_dim),0, dtype=float)
    for i in range(len(pos_gt_a)):
        for j in range(len(pos_gt_a[i])):
            center = pos_gt_a[i][j][0]
            width = pos_gt_a[i][j][1]
            center = center / opt.imgW * alpha_dim
            width = width / opt.imgW * alpha_dim
            sum_of_pdf = 0
            for k in range(alpha_dim):
                pdf = normpdf(k, center, width/2)
                batch_gt_alpha[i][j][k] = pdf
                sum_of_pdf += pdf
            batch_gt_alpha[i][j]/=sum_of_pdf
    gt = torch.from_numpy(batch_gt_alpha).float().to(opt.device)
#    print (batch_gt_alpha[0][0])
#    print (alpha[0][0])
#    cost1 = l2(gt[0][0], alpha[0][0])
#    print (cost1)
    cost = loss(alpha, gt)
    return cost
        
    
