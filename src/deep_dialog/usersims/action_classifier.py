# coding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import os
import errno
import re
import time
import tempfile
import codecs
import json
import subprocess
import random
import argparse
import logging


class ClassifyLayer(nn.Module):
    def __init__(self, input_size, num_tags, label2id, label_pad='<pad>', use_cuda=False):
        super(ClassifyLayer, self).__init__()
        self.use_cuda = use_cuda
        self.num_tags = num_tags
        self.label2id = label2id
        self.label_pad = label_pad
        # print('debug:', input_size, type(input_size), num_tags, type(num_tags))
        self.hidden2tag = nn.Linear(in_features=input_size, out_features=num_tags)
        self.logsoftmax = nn.LogSoftmax(dim=2)
        tag_weights = torch.ones(num_tags)
        tag_weights[label2id[label_pad]] = 0
        self.criterion = nn.NLLLoss(tag_weights)

    def forward(self, x, y):
        """
        :param x: torch.Tensor (batch_size, seq_len, n_in)
        :param y: torch.Tensor (batch_size, seq_len)
        :return:
        """
        tag_scores = self.hidden2tag(x)
        if self.training:
            tag_scores = self.logsoftmax(tag_scores)
        if self.label2id[self.label_pad] == 0:
            _, tag_result = torch.max(tag_scores[:, :, 1:], 2)  # block <pad> label as predict output
        else:
            _, tag_result = torch.max(tag_scores, 2)  # give up to block <pad> label for efficiency
        tag_result.add_(1)
        if self.training:
            return tag_result, self.criterion(tag_scores.view(-1, self.num_tags), Variable(y).view(-1))
        else:
            return tag_result, torch.FloatTensor([0.0])

    def get_probs(self, x):
        tag_scores = self.hidden2tag(x)
        if self.training:
            tag_scores = self.logsoftmax(tag_scores)

        return tag_scores