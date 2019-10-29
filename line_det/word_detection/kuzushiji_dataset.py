# -*- coding: utf-8 -*-

from .coco import CocoDataset
from .registry import DATASETS

@DATASETS.register_module
class KuzushijiDataset(CocoDataset):

    CLASSES = ('text')
