# -*- coding: utf-8 -*-


import numpy as np
from TextProposalGraphBuilder import TextProposalGraphBuilder
from np_utils import clip_boxes
# from .text_proposal_graph_builder import TextProposalGraphBuilder
# from .np_utils import clip_boxes


class TextProposalConnector:
    """
        连接文本框构建文本行
    """

    def __init__(self):
        self.graph_builder = TextProposalGraphBuilder()

    def group_text_proposals(self, text_proposals, scores, im_size):
        """
        将文本框连接起来，按照文本行分组
        :param text_proposals: 文本框，[n,(y1,x1,y2,x2)]
        :param scores: 文本框得分，[n]
        :param im_size: 图像尺寸,tuple(H,W,C)
        :return: list of list; 文本行列表，每个文本行是文本框索引号列表
        """
        graph = self.graph_builder.build_graph(text_proposals, scores, im_size)
        return graph.sub_graphs_connected()

    def fit_y(self, X, Y, x1, x2):
        """
        一元线性函数拟合X,Y,并返回x1,x2的的函数值
        """
        len(X) != 0
        # 只有一个点返回 y=Y[0]
        if np.sum(X == X[0]) == len(X):
            return Y[0], Y[0]
        p = np.poly1d(np.polyfit(X, Y, 1))
        return p(x1), p(x2)

    def get_text_lines_horizontal(self, text_proposals, scores, im_size):
        """
        获取文本行
        :param text_proposals: 文本框，[n,(y1,x1,y2,x2)]
        :param scores: 文本框得分，[n]
        :param im_size: 图像尺寸,tuple(H,W,C)
        :return: 文本行，边框和得分,numpy数组   [m,(y1,x1,y2,x2,score)]
        """
        tp_groups = self.group_text_proposals(text_proposals, scores, im_size)
        text_lines = np.zeros((len(tp_groups), 9), np.float32)
        # print("len(tp_groups):{}".format(len(tp_groups)))
        # 逐个文本行处理
        for index, tp_indices in enumerate(tp_groups):
            text_line_boxes = text_proposals[list(tp_indices)]
            # 宽度方向最小值和最大值
            x_min = np.min(text_line_boxes[:, 1])
            x_max = np.max(text_line_boxes[:, 3])

            # 文本框宽度的一半
            offset = (text_line_boxes[0, 3] - text_line_boxes[0, 1]) * 0.5
            # 使用一元线性函数求文本行左右两边高度边界
            lt_y, rt_y = self.fit_y(text_line_boxes[:, 1], text_line_boxes[:, 0], x_min - offset, x_max + offset)
            lb_y, rb_y = self.fit_y(text_line_boxes[:, 1], text_line_boxes[:, 2], x_min - offset, x_max + offset)

            # 文本行的得分为所有文本框得分的均值
            score = scores[list(tp_indices)].sum() / float(len(tp_indices))
            # 文本行坐标
            text_lines[index, 0] = x_min
            text_lines[index, 1] = lt_y
            text_lines[index, 2] = x_max
            text_lines[index, 3] = rt_y
            text_lines[index, 4] = x_max
            text_lines[index, 5] = rb_y
            text_lines[index, 6] = x_min
            text_lines[index, 7] = lb_y
        # 裁剪到图像尺寸内
        text_lines = clip_boxes(text_lines, im_size)

        return text_lines.astype(int)

    def get_text_lines_vertical(self, text_proposals, scores, im_size):
        """
        获取竖排文本行
        其中最后记录每行的四角点坐标时，相当于把图像逆时针转90度后再看。即 lt_x 不再是该行左上角，而是右上角；rt_x 不再是右上角，而是右下角，以此类推.
        :param text_proposals: 文本框，[n,(y1,x1,y2,x2)]
        :param scores: 文本框得分，[n]
        :param im_size: 图像尺寸,tuple(H,W,C)
        :return: 文本行，边框和得分,numpy数组   [m,(y1,x1,y2,x2,score)]
        """
        tp_groups = self.group_text_proposals(text_proposals, scores, im_size)
        text_lines = np.zeros((len(tp_groups), 9), np.float32)
        # print("len(tp_groups):{}".format(len(tp_groups)))
        # 逐个文本行处理
        for index, tp_indices in enumerate(tp_groups):
            text_line_boxes = text_proposals[list(tp_indices)]
            # 宽度方向最小值和最大值
            y_min = np.min(text_line_boxes[:, 0])
            y_max = np.max(text_line_boxes[:, 2])
            # 文本框宽度的一半
#             offset = (text_line_boxes[0, 3] - text_line_boxes[0, 1]) * 0.5
            offset = (text_line_boxes[0, 2] - text_line_boxes[0, 0]) * 0.5
#             # 使用一元线性函数求文本行左右两边高度边界
#             lt_x, rt_x = self.fit_y(text_line_boxes[:, 0], text_line_boxes[:, 1], y_min - offset, y_max + offset)
#             lb_x, rb_x = self.fit_y(text_line_boxes[:, 0], text_line_boxes[:, 3], y_min - offset, y_max + offset)
            # 【2019-10-18】回归一条中间线，然后扩到足够的宽度
            center_l_x, center_r_x = self.fit_y(text_line_boxes[:, 0], 0.5 * (text_line_boxes[:, 1] + text_line_boxes[:, 3]), y_min - offset, y_max + offset)
            line_height = np.max(text_line_boxes[:, 3] - text_line_boxes[:, 1])
            lt_x = center_l_x + 0.5 * line_height
            rt_x = center_r_x + 0.5 * line_height
            lb_x, rb_x = lt_x - line_height, rt_x - line_height
#             lt_x, rt_x = self.fit_y(text_line_boxes[:, 0], text_line_boxes[:, 1], y_min - offset, y_max + offset)
#             lb_x, rb_x = self.fit_y(text_line_boxes[:, 0], text_line_boxes[:, 3], y_min - offset, y_max + offset)

            # 文本行的得分为所有文本框得分的均值
            score = scores[list(tp_indices)].sum() / float(len(tp_indices))
            # 文本行坐标
            text_lines[index, 0] = lt_x
            text_lines[index, 1] = y_min
            text_lines[index, 2] = rt_x
            text_lines[index, 3] = y_max
            text_lines[index, 4] = rb_x
            text_lines[index, 5] = y_max
            text_lines[index, 6] = lb_x
            text_lines[index, 7] = y_min
            text_lines[index, 8] = score
        # 裁剪到图像尺寸内
        text_lines = clip_boxes(text_lines, im_size)

        return text_lines.astype(int)
