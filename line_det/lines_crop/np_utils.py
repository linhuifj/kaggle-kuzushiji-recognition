# -*- coding: utf-8 -*-

import numpy as np

def threshold(coords, min_, max_):
    return np.maximum(np.minimum(coords, max_), min_)

def clip_boxes(boxes, im_shape):
    """
    裁剪边框到图像内
    :param boxes: 边框 [n,(y1,x1,y2,x2)]
    :param im_shape: tuple(H,W,C)
    :return:
    """
    boxes[:, 0::2] = threshold(boxes[:, 0::2], 0, im_shape[1])
    boxes[:, 1::2] = threshold(boxes[:, 1::2], 0, im_shape[0])
    return boxes

