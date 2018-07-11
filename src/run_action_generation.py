# coding: utf-8
from __future__ import print_function, division
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

# from classifier import MultiLableClassifyLayer
# sys.path.append("..")
# import dialog_config
from deep_dialog import dialog_config
from deep_dialog.usersims.action_generation import history_based_classification, one_turn_classification, \
    seq2seq_action_generation, seq2seq_att_action_generation, state2seq_action_generation

logging.basicConfig(filename='', format='%(asctime)-15s %(levelname)s: %(message)s', level=logging.INFO)

PROJECT_DIR = dialog_config.PROJECT_DIR


def main():
    cmd = argparse.ArgumentParser()

    # running mode
    cmd.add_argument('-sm', '--select_model', type=str, help='select target model to run', choices=[
        'otc', 'one_turn_c',
        'mtc', 'multi_turn_c',
        'ssg', 'seq2seq_gen',
        'ssag', 'seq2seq_att_gen',
        'sv2s', 'state_v2seq',
    ])
    model_opt = cmd.parse_args(sys.argv[1: 3])

    if model_opt.select_model == 'otc' or model_opt.select_model == 'one_turn_c':
        DATA_MARK = dialog_config.DATA_MARK[0]  # 'extracted_no_nlg_no_nlu'
    elif model_opt.select_model == 'mtc' or model_opt.select_model == 'multi_turn_c':
        DATA_MARK = dialog_config.DATA_MARK[1]  # 'extracted_no_nlg_no_nlu_lstm'
    elif model_opt.select_model == 'ssg' or model_opt.select_model == 'seq2seq_gen':
        DATA_MARK = dialog_config.DATA_MARK[2]  # 'extracted_no_nlg_no_nlu_seq2seq'
    elif model_opt.select_model == 'ssag' or model_opt.select_model == 'seq2seq_att_gen':
        DATA_MARK = dialog_config.DATA_MARK[3]  # 'extracted_no_nlg_no_nlu_seq2seq_att'
    elif model_opt.select_model == 'sv2s' or model_opt.select_model == 'state_v2seq':
        DATA_MARK = dialog_config.DATA_MARK[4]  # 'extracted_no_nlg_no_nlu_state_v2seq'
    else:
        raise TypeError("Invalid choice for model")


    # define path
    cmd.add_argument('--train_path', help='the path to the training file.', default= '{0}TaskOrientedDialogue/data/TC-bot/data/{1}/{1}.train.json'.format(PROJECT_DIR, DATA_MARK))
    cmd.add_argument('--dev_path', help='the path to the validation file.', default='{0}TaskOrientedDialogue/data/TC-bot/data/{1}/{1}.dev.json'.format(PROJECT_DIR, DATA_MARK))
    cmd.add_argument('--test_path', help='the path to the testing file.', default='{0}TaskOrientedDialogue/data/TC-bot/data/{1}/{1}.test.json'.format(PROJECT_DIR, DATA_MARK))
    cmd.add_argument('--dict_path', help='the path to the full dict file.', default='{0}TaskOrientedDialogue/data/TC-bot/data/{1}/{1}.dict.json'.format(PROJECT_DIR, DATA_MARK))
    cmd.add_argument("--model", help="path to save model", default='{0}TaskOrientedDialogue/data/TC-bot/data/{1}/'.format(PROJECT_DIR, DATA_MARK))
    cmd.add_argument("--model_name", help="name to save model", default='model.pkl')
    cmd.add_argument('--output', help='The path to the output file.', default='{0}TaskOrientedDialogue/data/TC-bot/data/{1}/{1}.output.json'.format(PROJECT_DIR, DATA_MARK))
    # cmd.add_argument("--script", required=True, help="The path to the evaluation script: ./eval/conlleval.pl")
    # cmd.add_argument("--word_embedding", type=str, default='',
    #                  help="pass a path to word vectors from file(not finished), empty string to load from pytorch-nlp")

    # environment setting
    cmd.add_argument('--seed', default=1, type=int, help='the random seed.')
    cmd.add_argument('-gpu', '--gpu', default=-1, type=int, help='use id of gpu, -1 if cpu.')
    cmd.add_argument('--debug', action='store_true', help='run in debug mode')

    # define detail
    cmd.add_argument('--encoder', default='lstm', choices=['lstm', 'gru'],
                     help='the type of encoder: valid options=[lstm, gru]')
    cmd.add_argument('--decoder', default='lstm', choices=['lstm', 'gru'],
                     help='the type of encoder: valid options=[lstm, gru]')
    cmd.add_argument('--classifier', default='vanilla', choices=['vanilla'],
                     help='The type of classifier: valid options=[vanilla]')
    cmd.add_argument('--optimizer', default='adam', choices=['sgd', 'adam'],
                     help='the type of optimizer: valid options=[sgd, adam]')

    cmd.add_argument("--batch_size", "--batch", type=int, default=64, help='the batch size.')
    cmd.add_argument("--hidden_dim", "--hidden", type=int, default=128, help='the hidden dimension.')
    cmd.add_argument("--max_epoch", type=int, default=30, help='the maximum number of iteration.')
    cmd.add_argument("--word_dim", type=int, default=300, help='the input dimension.')
    cmd.add_argument("--dropout", type=float, default=0.3, help='the dropout rate')
    cmd.add_argument("--depth", type=int, default=2, help='the depth of lstm')
    cmd.add_argument('--max_len', type=int,default=50, help='max length for sentence')
    cmd.add_argument('--use_attention', action='store_true', help='use attention in decoding')
    cmd.add_argument('--direction', type=str, default='bi', help='bi to use bidirectional encoder, si or else single')
    cmd.add_argument("--lr", type=float, default=0.01, help='the learning rate.')
    cmd.add_argument("--lr_decay", type=float, default=0, help='the learning rate decay.')
    cmd.add_argument("--clip_grad", type=float, default=5, help='the tense of clipped grad.')
    cmd.add_argument("--teacher_forcing_ratio", type=float, default=0.5, help='the teacher forcing ratio for seq2seq training')
    cmd.add_argument('--embedded_v_size', type=int, default=200, help='set embedded vector size for state2seq')
    opt = cmd.parse_args()

    print(opt)
    torch.manual_seed(opt.seed)
    random.seed(opt.seed)
    if opt.gpu >= 0:
        torch.cuda.set_device(opt.gpu)
        if opt.seed > 0:
            torch.cuda.manual_seed(opt.seed)

    if opt.select_model == 'otc' or opt.select_model == 'one_turn_c':
        print('============== Start one_turn_classification ==============')
        one_turn_classification(opt)
    elif opt.select_model == 'mtc' or opt.select_model == 'multi_turn_c':
        print('============== Start history based classification ==============')
        history_based_classification(opt)
    elif opt.select_model == 'ssg' or opt.select_model == 'seq2seq_gen':
        print('============== Start seq2seq action generation ==============')
        seq2seq_action_generation(opt)
    elif opt.select_model == 'ssag' or model_opt.select_model == 'seq2seq_att_gen':
        print('============== Start seq2seq_att action generation ==============')
        seq2seq_att_action_generation(opt)
    elif model_opt.select_model == 'sv2s' or model_opt.select_model == 'state_v2seq':
        print('============== Start state_v2seq action generation ==============')
        state2seq_action_generation(opt)
    # setting logging
    # DEBUG = False
    # DEBUG = True
    # if DEBUG or opt.debug:
    #     logging.basicConfig(format='%(asctime)-15s %(levelname)s: %(message)s', level=logging.DEBUG)
    # else:
    #     logging.basicConfig(format='%(asctime)-15s %(levelname)s: %(message)s', level=logging.INFO)


if __name__ == '__main__':
    main()
