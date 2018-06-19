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
import numpy as np

import sys
sys.path.append("..")
import dialog_config


# LOG_PATH = "E:\Projects\Research\TaskOrientedDialogue\data\TC-bot\log\no_nlg_no_nlu.log"


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


class MultiLableClassifyLayer(nn.Module):
    def __init__(self, input_size, hidden_size, num_tags, opt, use_cuda=False):
        super(MultiLableClassifyLayer, self).__init__()

        self.main = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_tags)
        )
        self.criterion = nn.MultiLabelSoftMarginLoss()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_tags = num_tags
        self.use_cuda = use_cuda
        self.opt = opt

    def forward(self, input, golden=None):
        # print('DEBUG:', input, golden)
        input, golden = Variable(input.float()), Variable(golden.float())
        pred = self.main(input)  # For each pos, > 0 for positive tag, < 0 for negative tag
        # For each pos, 1 for positive tag, 0 for negative tag
        classify_results = [[int(pred_j > 0) for pred_j in pred_i] for pred_i in pred.data]
        if self.training:
            return classify_results, self.criterion(pred, golden)
        else:
            return classify_results, torch.FloatTensor([0.0])


class LSTM_MultiLabelClassifier(nn.Module):
    def __int__(self, input_size, hidden_size, num_tags, opt, use_cuda=False):

        self.encoder = nn.LSTM(
            input_size=input_size, hidden_size=hidden_size, num_layers=opt.depth,
            batch_first=True, dropout=opt.dropout, bidirectional=True
        )
        encoder_output_size = 2 * opt.hidden_dim  # because of the bi-directional
        self.classify_layer = MultiLableClassifyLayer(input_size, hidden_size * 2, num_tags, opt, use_cuda)

        self.criterion = nn.MultiLabelSoftMarginLoss()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_tags = num_tags
        self.use_cuda = use_cuda
        self.opt = opt

    def forward(self, batch_x, batch_y):
        batch_size = batch_x.size(0)
        output, hidden = self.encoder(batch_x)
        output, loss = self.classify_layer(output, batch_y)
        return output, loss


def test_MultiLableClassifyLayer():
    """
    Unit test for multilabel ClassifyLayer
    :return:
    """
    # Make fake dataset
    # (1, 0) => target labels 1 0 1
    # (0, 1) => target labels 0 1 0
    # (1, 1) => target labels 0 0 1
    train = []
    labels = []
    for i in range(10000):
        category = (np.random.choice([0, 1]), np.random.choice([0, 1]))
        if category == (1, 0):
            train.append([np.random.uniform(0.1, 1), 0])
            labels.append([1, 0, 1])
        if category == (0, 1):
            train.append([0, np.random.uniform(0.1, 1)])
            labels.append([0, 1, 0])
        if category == (0, 0):
            train.append([np.random.uniform(0.1, 1), np.random.uniform(0.1, 1)])
            labels.append([0, 0, 1])

    # Training process
    nlabel = len(labels[0])  # => 3
    input_size = len(train[0])
    classifier = MultiLableClassifyLayer(input_size=input_size, hidden_size=64, num_tags=nlabel)

    optimizer = optim.Adam(classifier.parameters())

    # Training
    classifier.train()
    epochs = 5
    for epoch in range(epochs):
        losses = []
        for i, sample in enumerate(train):
            inputv = Variable(torch.FloatTensor(sample)).view(1, -1)
            labelsv = Variable(torch.FloatTensor(labels[i])).view(1, -1)

            output, loss = classifier(inputv)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.data.mean())
        print('[%d/%d] Loss: %.3f' % (epoch + 1, epochs, np.mean(losses)))
    # Testing
    print('Start Testing')
    classifier.eval()
    input_sample = [float(a) for a in raw_input().split()]
    while input_sample:
        input_sample = [float(a) for a in raw_input().split()]
        input_sample_v = Variable(torch.FloatTensor(input_sample)).view(1, -1)
        output = classifier(input_sample_v)
        print('pred resultis', output)


if __name__ == "__main__":
    test_MultiLableClassifyLayer()
