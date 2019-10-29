"""
Copyright (c) 2019-present NAVER Corp.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import torch.nn as nn

from modules.transformation import TPS_SpatialTransformerNetwork
from modules.feature_extraction import  ResNet_FeatureExtractor
from modules.sequence_modeling import BidirectionalLSTM, BidirectionalGRU
from modules.prediction import Attention


class Model(nn.Module):

    def __init__(self, opt):
        super(Model, self).__init__()
        self.opt = opt
        self.stages = {'Trans': opt.Transformation, 'Feat': opt.FeatureExtraction,
                       'Seq': opt.SequenceModeling, 'Pred': opt.Prediction}

        """ Transformation """
        if opt.Transformation == 'TPS':
            self.Transformation = TPS_SpatialTransformerNetwork(
                F=opt.num_fiducial, I_size=(opt.imgH, opt.imgW), I_r_size=(opt.imgH, opt.imgW), I_channel_num=opt.input_channel)
        else:
            print('No Transformation module specified')

        """ FeatureExtraction """
        if opt.FeatureExtraction == 'VGG':
            self.FeatureExtraction = VGG_FeatureExtractor(opt.input_channel, opt.output_channel)
        elif opt.FeatureExtraction == 'RCNN':
            self.FeatureExtraction = RCNN_FeatureExtractor(opt.input_channel, opt.output_channel)
        elif opt.FeatureExtraction == 'ResNet' or 'SEResNet' == opt.FeatureExtraction:
            self.FeatureExtraction = ResNet_FeatureExtractor(opt.input_channel, opt.output_channel, opt)
        elif 'SEResNetXt' in opt.FeatureExtraction:
            opt.output_channel = 2048
            self.FeatureExtraction = ResNet_FeatureExtractor(opt.input_channel, opt.output_channel, opt)            
        else:
            raise Exception('No FeatureExtraction module specified')
        self.FeatureExtraction_output = opt.output_channel  # int(imgH/16-1) * 512
        self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((None, 1))  # Transform final (imgH/16-1) -> 1

        if opt.dropout > 0:
            self.dropout = nn.Dropout(opt.dropout)
        else:
            self.dropout = None
        """ Sequence modeling"""
        if opt.SequenceModeling == 'BiLSTM':
            self.SequenceModeling = nn.Sequential(
                BidirectionalLSTM(self.FeatureExtraction_output, opt.hidden_size, opt.hidden_size, opt.rnnlayers, opt.rnndropout),
                BidirectionalLSTM(opt.hidden_size, opt.hidden_size, opt.hidden_size, opt.rnnlayers, opt.rnndropout))
            self.SequenceModeling_output = opt.hidden_size
        elif opt.SequenceModeling == 'BiGRU':
            self.SequenceModeling = nn.Sequential(
                BidirectionalGRU(self.FeatureExtraction_output, opt.hidden_size, opt.hidden_size, opt.rnnlayers, opt.rnndropout),
                BidirectionalGRU(opt.hidden_size, opt.hidden_size, opt.hidden_size, opt.rnnlayers, opt.rnndropout))
            self.SequenceModeling_output = opt.hidden_size
        else:
            print('No SequenceModeling module specified')
            self.SequenceModeling_output = self.FeatureExtraction_output

        """ Prediction """
        if opt.Prediction == 'CTC':
            self.Prediction = nn.Linear(self.SequenceModeling_output, opt.num_class + 1)
        elif opt.Prediction == 'Attn':
            self.Prediction = Attention(opt, self.SequenceModeling_output, opt.hidden_size, opt.num_class + 2)
        elif opt.Prediction == 'CTC_Attn':
            self.Prediction_ctc = nn.Linear(self.SequenceModeling_output, opt.num_class + 1)
            self.Prediction_attn = Attention(opt, self.SequenceModeling_output, opt.hidden_size, opt.num_class + 2)
        else:
            raise Exception('Prediction is neither CTC or Attn')

    def forward(self, input, text, is_train=True):
        """ Transformation stage """
        if not self.stages['Trans'] == "None":
            input = self.Transformation(input)

        """ Feature extraction stage """
        visual_feature = self.FeatureExtraction(input)
        visual_feature = self.AdaptiveAvgPool(visual_feature.permute(0, 3, 1, 2))  # [b, c, h, w] -> [b, w, c, h]
        visual_feature = visual_feature.squeeze(3)

        if self.dropout is not None:
            visual_feature = self.dropout(visual_feature)
            
        """ Sequence modeling stage """
        if self.stages['Seq'] == 'BiLSTM':
            contextual_feature = self.SequenceModeling(visual_feature)
        elif self.stages['Seq'] == 'BiGRU':
            contextual_feature = self.SequenceModeling(visual_feature)            
        else:
            contextual_feature = visual_feature  # for convenience. this is NOT contextually modeled by BiLSTM

        """ Prediction stage """
        if self.stages['Pred'] == 'CTC':
            prediction = self.Prediction(contextual_feature.contiguous())
        elif self.stages['Pred'] == 'Attn':
            prediction = self.Prediction(contextual_feature.contiguous(), text, is_train, batch_max_length=self.opt.batch_max_length)
        elif self.stages['Pred'] == 'CTC_Attn':
            prediction1 = self.Prediction_ctc(contextual_feature.contiguous())
            prediction2 = self.Prediction_attn(contextual_feature.contiguous(), text, is_train, batch_max_length=self.opt.batch_max_length)
            prediction = prediction1,prediction2
        return prediction
