# -*- coding: utf-8 -*-

import torch
from multiprocessing import Pool
from tqdm import tqdm
import os
import glob
import mmcv
from mmcv.runner import load_checkpoint
from mmdet.models import build_detector
from mmdet.apis import inference_detector, show_result, init_detector

###########################################
######### Change this to your path ########
###########################################
kaggle_data_root = '/path/to/data/root/'

def det_box_to_str(box):
    b = list(map(lambda x: str(round(x, 4)), box))
    return b[4] + ' ' + ' '.join(b[:4])

def convert_np_to_kagglelabels(boxes_np):
    boxes_np[:,2] = boxes_np[:,2] - boxes_np[:,0]
    boxes_np[:,3] = boxes_np[:,3] - boxes_np[:,1]
    boxes = boxes_np.tolist()
    return ' '.join(list(map(det_box_to_str, boxes)))


# 导入模型参数
cfg = mmcv.Config.fromfile('./kuzu_htc_x101_64x4d_fpn_2gpu.py')
cfg.model.pretrained = None

# 构建化模型和加载检查点卡
model = init_detector(cfg, './work_dir/latest.pth', device='cuda:0')

lines_to_write = []
test_images_path = os.path.join(kaggle_data_root, 'test_images/')
imgs = glob.glob(test_images_path + '*')

#################
from multiprocessing import set_start_method
try:
    set_start_method('spawn')
except RuntimeError:
    pass

def worker(img):
    result = inference_detector(model, img)
    basename = os.path.basename(img)[:-4]
    return basename + ',' + convert_np_to_kagglelabels(result[0][0])

if __name__ == '__main__':
    torch.multiprocessing.freeze_support()

    with Pool(3) as pool:
        lines_to_write = list(tqdm(pool.imap(worker, imgs), total=len(imgs)))

    csv_save_path = os.path.join(kaggle_data_root, 'htc_det_boxes.csv')
    with open(csv_save_path, 'w') as f:
        f.write('image_id,labels\n')
        f.write('\n'.join(lines_to_write))
