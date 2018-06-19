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

from classifier import MultiLableClassifyLayer
sys.path.append("..")
import dialog_config

logging.basicConfig(filename='', format='%(asctime)-15s %(levelname)s: %(message)s', level=logging.INFO)

# PROJECT_DIR = 'E:/Projects/Research/'  # for Windows
PROJECT_DIR = '/users4/ythou/Projects/'  # for hpc
# PROJECT_DIR = 'E:/Projects/Research/'  # for tencent linux

DATA_MARK = 'extracted_no_nlg_no_nlu_lstm '
# DATA_MARK = 'extracted_no_nlg_no_nlu'


def get_f1(pred_tags_lst, golden_tags_lst):
    """
    get sentence level f score
    :param pred_tags_lst: list of one hot alike tags i.e. [[1, 0, 1]]
    :param golden_tags_lst: list of one hot alike tags i.e. [[1, 0, 1]]
    :return: precision, recall, f1
    """
    tp, fp, fn = 0, 0, 0
    # print(len(pred_tags_lst), len(pred_tags_lst[0]), golden_tags_lst.shape)
    for pred_tags, golden_tags in zip(pred_tags_lst, golden_tags_lst):
        if len(pred_tags) != len(golden_tags):
            logging.error('Unmatched tags: \npred:{}{}\ngold:{}{}'.format(
                len(pred_tags), pred_tags, len(golden_tags), golden_tags)
            )
            raise RuntimeError
        for pred_t, gold_t in zip(pred_tags, golden_tags):
            if pred_t == 1:
                if pred_t == gold_t:
                    tp += 1
                elif gold_t == 0:
                    fp += 1
                else:
                    raise RuntimeError
            elif pred_t == 0:
                if pred_t == gold_t:
                    pass
                elif gold_t == 1:
                    fn += 1
                else:
                    raise RuntimeError
    if tp == 0:
        precision = 0
        recall = 0
        f1 = 0
    else:
        precision = 1.0 * tp / (tp + fp)
        recall = 1.0 * tp / (tp + fn)
        f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1,


def vector2state(state_vector, full_dict):
    state_v_component = [
        'goal_inform_slots_v',
        'goal_request_slots_v',
        'history_slots_v',  # informed slots, 1 informed, 0 irrelevant, -1 not informed,
        'rest_slots_v',  # remained request slots, 1 for remained, 0 irrelevant, -1 for already got,
        'system_diaact_v',  # system diaact,
        'system_inform_slots_v',  # inform slots of sys response,
        'system_request_slots_v',  # request slots of sys response,
        'consistency_v',  # for each position, -1 inconsistent, 0 irrelevent or not requested, 1 consistent,
        'dialog_status_v'  # -1, 0, 1 for failed, no outcome, success,
    ]

    id2sys_inform_slot = full_dict['id2sys_inform_slot']
    id2sys_request_slot = full_dict['id2sys_request_slot']
    id2user_inform_slot = full_dict['id2user_inform_slot']
    id2user_request_slot = full_dict['id2user_request_slot']
    id2diaact = full_dict['id2diaact']

    vector_corresponding_dict_lst = [
        id2user_inform_slot, id2user_request_slot, id2user_inform_slot, id2user_request_slot,
        id2diaact, id2sys_inform_slot, id2sys_request_slot, id2user_inform_slot,
        {-1: 'fail', 0: 'no_outcome', 1: 'success'}
    ]
    ret = {}
    start_idx = 0
    for name, id2item in zip(state_v_component, vector_corresponding_dict_lst):
        component_v = state_vector[start_idx: start_idx + len(id2item)]
        ret[name] = zip(id2item.values(), component_v)
    return ret


def vector2action(action_vector, full_dict):
    id2diaact = full_dict['id2diaact']
    id2user_inform_slot = full_dict['id2user_inform_slot']
    id2user_request_slot = full_dict['id2user_request_slot']
    diaact_end = len(id2diaact)
    inform_slot_end = len(id2user_request_slot) + diaact_end
    diaact_v = action_vector[: diaact_end]
    inform_slots_v = action_vector[diaact_end: inform_slot_end]
    request_slots_v = action_vector[inform_slot_end: ]

    diaact = ''
    inform_slots = []
    request_slots = []
    for ind, item in enumerate(diaact_v):
        if item == 1:
            if not diaact:
                diaact = id2diaact[str(ind)]
            else:
                print('Warning: multi-action predicted')
                # print(len(action_vector), action_vector, diaact_end, id2diaact)
                # raise RuntimeError
    for ind, item in enumerate(inform_slots_v):
        if item == 1:
            inform_slots.append(id2user_inform_slot[str(ind)])
    for ind, item in enumerate(request_slots_v):
        if item == 1:
            request_slots.append(id2user_request_slot[str(ind)])

    ret = {
        'diaact': diaact,
        'inform_slots': inform_slots,
        'request_slots': request_slots,
    }
    return ret


def eval_model(model, valid_x, valid_y, id2label, opt):

    if opt.output is not None:
        output_path = opt.output
        output_file = open(output_path, 'w')
    else:
        output_file = None
    model.eval()

    all_preds = []
    for x, y in zip(valid_x, valid_y):
        output, loss = model.forward(x, y)
        output_data = output
        for s, pred_a, gold_a in zip(x, output_data, y):
            log = {
                'state': vector2state(s, id2label),
                'pred': vector2action(pred_a, id2label),
                'gold': vector2action(gold_a, id2label),
            }
            if output_file:
                output_file.write(json.dumps(log))
        all_preds.extend(output_data)

    output_file.close()
    valid_y = torch.cat(valid_y)  # re-form batches into one
    precision, recall, f1 = get_f1(pred_tags_lst=all_preds, golden_tags_lst=valid_y)
    return precision, recall, f1


def train_model(epoch, model, optimizer,
                train_x, train_y,
                valid_x, valid_y,
                test_x, test_y,
                ix2label, best_valid, test_f1_score):
    model.train()
    opt = model.opt

    total_loss = 0.0
    cnt = 0
    start_time = time.time()

    lst = list(range(len(train_x)))
    random.shuffle(lst)
    train_x = [train_x[l] for l in lst]
    train_y = [train_y[l] for l in lst]

    for x, y in zip(train_x, train_y):
        cnt += 1
        model.zero_grad()
        _, loss = model.forward(x, y)
        total_loss += loss.data[0]
        n_tags = len(train_y[0]) * len(x)
        loss.backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), opt.clip_grad)
        optimizer.step()
        if cnt * opt.batch_size % 1024 == 0:
            logging.info("Epoch={} iter={} lr={:.6f} train_ave_loss={:.6f} time={:.2f}s".format(
                epoch, cnt, optimizer.param_groups[0]['lr'],
                1.0 * loss.data[0] / n_tags, time.time() - start_time
            ))
            start_time = time.time()

    dev_precision, dev_recall, dev_f1_score = eval_model(model, valid_x, valid_y, ix2label, opt)
    logging.info("Epoch={} iter={} lr={:.6f} train_loss={:.6f} valid_acc={:.6f}".format(
        epoch, cnt, optimizer.param_groups[0]['lr'], total_loss, dev_f1_score))

    if dev_f1_score > best_valid:
        torch.save(
            {
                'state_dict': model.state_dict(),
                'param':
                {
                    'input_size': model.input_size,
                    'hidden_size': model.hidden_size,
                    'num_tags': model.num_tags,
                    'opt': model.opt,
                    'use_cuda': model.use_cuda
                }
            },
            os.path.join(opt.model, 'model.pkl'
        ))
        best_valid = dev_f1_score
        test_precision, test_recall, test_f1_score = eval_model(model, test_x, test_y, ix2label, opt)
        logging.info("New record achieved!")
        logging.info("Epoch={} iter={} lr={:.6f} test_precision={:.6f}, test_recall={:.6f}, test_f1={:.6f}".format(
            epoch, cnt, optimizer.param_groups[0]['lr'], test_precision, test_recall, test_f1_score))
    return best_valid, test_f1_score


def create_one_batch(x, y, use_cuda=False):
    batch_size = len(x)
    lens = [len(xi) for xi in x]
    max_len = max(lens)  # useless for current situation

    batch_x = torch.LongTensor(x)
    batch_y = torch.LongTensor(y)
    if use_cuda:
        batch_x = batch_x.cuda()
        batch_y = batch_y.cuda()
    return batch_x, batch_y, lens


def create_batches(x, y, batch_size, sort=True, shuffle=True, use_cuda=False):
    lst = list(range(len(x)))
    if shuffle:
        random.shuffle(lst)
    if sort:
        lst = sorted(lst, key=lambda i: -len(x[i]))

    x = [x[i] for i in lst]
    y = [y[i] for i in lst]

    nbatch = (len(x) - 1) // batch_size + 1  # subtract 1 fist to handle situation: len(x) // batch_size == 0
    batches_x, batches_y = [], []

    for i in range(nbatch):
        start_id, end_id = i * batch_size, (i + 1) * batch_size
        bx, by, _ = create_one_batch(x[start_id: end_id], y[start_id: end_id], use_cuda)

        batches_x.append(bx)
        batches_y.append(by)

    if sort:
        pos_lst = list(range(nbatch))
        random.shuffle(pos_lst)

        batches_x = [batches_x[i] for i in pos_lst]
        batches_y = [batches_y[i] for i in pos_lst]

    logging.info("{} batches, batch size: {}".format(nbatch, batch_size))
    return batches_x, batches_y


def one_turn_classification(opt):
    use_cuda = opt.gpu >= 0 and torch.cuda.is_available()
    with open(opt.train_path, 'r') as train_file, \
            open(opt.dev_path, 'r') as dev_file, \
            open(opt.test_path, 'r') as test_file, \
            open(opt.dict_path, 'r') as dict_file:
        logging.info('Start loading data from:\ntrain:{}\ndev:{}\ntest:{}\ndict:{}\n'.format(
            opt.train_path, opt.dev_path, opt.test_path, opt.dict_path
        ))
        train_data = json.load(train_file)
        dev_data = json.load(dev_file)
        test_data = json.load(test_file)
        full_dict = json.load(dict_file)

        # TODO: change full dict to a whole dict
        logging.info('Finish data loading.')
        print('Finish  data loading!!!!!!!!!!')
        # unpack data
        train_input, train_label, train_turn_id = zip(* train_data)
        dev_input, dev_label, dev_turn_id = zip(* dev_data)
        test_input, test_label, test_turn_id = zip(* test_data)
        train_x, train_y = create_batches(train_input, train_label, opt.batch_size, use_cuda=use_cuda)
        dev_x, dev_y = create_batches(dev_input, dev_label, opt.batch_size, use_cuda=use_cuda)
        test_x, test_y = create_batches(test_input, test_label, opt.batch_size, use_cuda=use_cuda)

        input_size = len(train_input[0])
        num_tags = len(train_label[0])
        classifier = MultiLableClassifyLayer(input_size=input_size, hidden_size=opt.hidden_dim, num_tags=num_tags,
                                             opt=opt, use_cuda=use_cuda)

        optimizer = optim.Adam(classifier.parameters())

        best_valid, test_result = -1e8, -1e8
        for epoch in range(opt.max_epoch):
            best_valid, test_result = train_model(
                epoch=epoch,
                model=classifier,
                optimizer=optimizer,
                train_x=train_x, train_y=train_y,
                valid_x=dev_x, valid_y=dev_y,
                test_x=test_x, test_y=test_y,
                ix2label=full_dict, best_valid=best_valid, test_f1_score=test_result
            )
            if opt.lr_decay > 0:
                optimizer.param_groups[0]['lr'] *= opt.lr_decay  # there is only one group, so use index 0
            # logging.info('Total encoder time: {:.2f}s'.format(model.eval_time / (epoch + 1)))
            # logging.info('Total embedding time: {:.2f}s'.format(model.emb_time / (epoch + 1)))
            # logging.info('Total classify time: {:.2f}s'.format(model.classify_time / (epoch + 1)))
        logging.info("best_valid_f1: {:.6f}".format(best_valid))
        logging.info("test_f1: {:.6f}".format(test_result))


def transform_data_into_history_style(data, turn_ids, history_turns=5):
    ret = []
    history = []
    for item, turn_id in zip(data, turn_ids):
        if turn_id == 0:
            history = [item] * (history_turns - 1)  # pad empty history
        elif len(history) < 4:
            history = [item] * (history_turns - 1)  # deal with in-complete dialogue
        history.append(item)  # add current state vector
        sample = history[-history_turns:]
        ret.append(sample)
    return ret


def history_based_classification(opt):
    use_cuda = opt.gpu >= 0 and torch.cuda.is_available()
    with open(opt.train_path, 'r') as train_file, \
            open(opt.dev_path, 'r') as dev_file, \
            open(opt.test_path, 'r') as test_file, \
            open(opt.dict_path, 'r') as dict_file:
        logging.info('Start loading data from:\ntrain:{}\ndev:{}\ntest:{}\ndict:{}\n'.format(
            opt.train_path, opt.dev_path, opt.test_path, opt.dict_path
        ))
        train_data = json.load(train_file)
        dev_data = json.load(dev_file)
        test_data = json.load(test_file)
        full_dict = json.load(dict_file)

        # TODO: change full dict to a whole dict
        logging.info('Finish data loading.')
        print('Finish  data loading!!!!!!!!!!')
        # unpack data
        train_input, train_label, train_turn_id = zip(*train_data)
        dev_input, dev_label, dev_turn_id = zip(*dev_data)
        test_input, test_label, test_turn_id = zip(*test_data)

        # stack history
        train_input = transform_data_into_history_style(train_input, train_turn_id)
        dev_input = transform_data_into_history_style(dev_input, dev_turn_id)
        test_input = transform_data_into_history_style(test_input, test_turn_id)

        train_x, train_y = create_batches(train_input, train_label, opt.batch_size, use_cuda=use_cuda)
        dev_x, dev_y = create_batches(dev_input, dev_label, opt.batch_size, use_cuda=use_cuda)
        test_x, test_y = create_batches(test_input, test_label, opt.batch_size, use_cuda=use_cuda)

        input_size = len(train_input[0])
        num_tags = len(train_label[0])
        classifier = MultiLableClassifyLayer(input_size=input_size, hidden_size=opt.hidden_dim, num_tags=num_tags,
                                             opt=opt, use_cuda=use_cuda)

        optimizer = optim.Adam(classifier.parameters())

        best_valid, test_result = -1e8, -1e8
        for epoch in range(opt.max_epoch):
            best_valid, test_result = train_model(
                epoch=epoch,
                model=classifier,
                optimizer=optimizer,
                train_x=train_x, train_y=train_y,
                valid_x=dev_x, valid_y=dev_y,
                test_x=test_x, test_y=test_y,
                ix2label=full_dict, best_valid=best_valid, test_f1_score=test_result
            )
            if opt.lr_decay > 0:
                optimizer.param_groups[0]['lr'] *= opt.lr_decay  # there is only one group, so use index 0
            # logging.info('Total encoder time: {:.2f}s'.format(model.eval_time / (epoch + 1)))
            # logging.info('Total embedding time: {:.2f}s'.format(model.emb_time / (epoch + 1)))
            # logging.info('Total classify time: {:.2f}s'.format(model.classify_time / (epoch + 1)))
        logging.info("best_valid_f1: {:.6f}".format(best_valid))
        logging.info("test_f1: {:.6f}".format(test_result))


def main():
    cmd = argparse.ArgumentParser()

    # running mode
    cmd.add_argument('-otc', '--one_turn_c', action='store_true', help='run train and test at the same time')
    # cmd.add_argument('-tt', '--train_and_test', action='store_true', help='run train and test at the same time')

    # define path
    cmd.add_argument('--train_path', help='the path to the training file.', default= '{0}TaskOrientedDialogue/data/TC-bot/data/{1}/{1}.train.json'.format(PROJECT_DIR, DATA_MARK))
    cmd.add_argument('--dev_path', help='the path to the validation file.', default='{0}TaskOrientedDialogue/data/TC-bot/data/{1}/{1}.dev.json'.format(PROJECT_DIR, DATA_MARK))
    cmd.add_argument('--test_path', help='the path to the testing file.', default='{0}TaskOrientedDialogue/data/TC-bot/data/{1}/{1}.test.json'.format(PROJECT_DIR, DATA_MARK))
    cmd.add_argument('--dict_path', help='the path to the full dict file.', default='{0}TaskOrientedDialogue/data/TC-bot/data/{1}/{1}.dict.json'.format(PROJECT_DIR, DATA_MARK))
    cmd.add_argument("--model", help="path to save model", default='{0}TaskOrientedDialogue/data/TC-bot/data/{1}/'.format(PROJECT_DIR, DATA_MARK))
    # cmd.add_argument('--train_path', required=True, help='the path to the training file.', default= '{0}/TaskOrientedDialogue/data/TC-bot/data/{1}/{1}.train.json'.format(PROJECT_DIR, DATA_MARK))
    # cmd.add_argument('--dev_path', required=True, help='the path to the validation file.', default='{0}/TaskOrientedDialogue/data/TC-bot/data/{1}/{1}.dev.json'.format(PROJECT_DIR, DATA_MARK))
    # cmd.add_argument('--test_path', required=True, help='the path to the testing file.', default='{0}/TaskOrientedDialogue/data/TC-bot/data/{1}/{1}.test.json'.format(PROJECT_DIR, DATA_MARK))
    # cmd.add_argument('--dict_path', required=True, help='the path to the full dict file.', default='{0}/TaskOrientedDialogue/data/TC-bot/data/{1}/{1}.dict.json'.format(PROJECT_DIR, DATA_MARK))
    # cmd.add_argument("--model", required=True, help="path to save model", default='{0}/TaskOrientedDialogue/data/TC-bot/data/{1}/'.format(PROJECT_DIR, DATA_MARK))
    cmd.add_argument('--output', help='The path to the output file.', default='{0}TaskOrientedDialogue/data/TC-bot/data/{1}/{1}.output.json'.format(PROJECT_DIR, DATA_MARK))
    # cmd.add_argument("--script", required=True, help="The path to the evaluation script: ./eval/conlleval.pl")
    # cmd.add_argument("--word_embedding", type=str, default='',
    #                  help="pass a path to word vectors from file(not finished), empty string to load from pytorch-nlp")

    # environment setting
    cmd.add_argument('--seed', default=1, type=int, help='the random seed.')
    cmd.add_argument('--gpu', default=-1, type=int, help='use id of gpu, -1 if cpu.')
    cmd.add_argument('--debug', action='store_true', help='run in debug mode')

    # define detail
    cmd.add_argument('--encoder', default='lstm', choices=['lstm'],
                     help='the type of encoder: valid options=[lstm]')
    cmd.add_argument('--classifier', default='vanilla', choices=['vanilla'],
                     help='The type of classifier: valid options=[vanilla]')
    cmd.add_argument('--optimizer', default='adam', choices=['sgd', 'adam'],
                     help='the type of optimizer: valid options=[sgd, adam]')

    cmd.add_argument("--batch_size", "--batch", type=int, default=64, help='the batch size.')
    cmd.add_argument("--hidden_dim", "--hidden", type=int, default=128, help='the hidden dimension.')
    cmd.add_argument("--max_epoch", type=int, default=30, help='the maximum number of iteration.')
    cmd.add_argument("--word_dim", type=int, default=300, help='the input dimension.')
    cmd.add_argument("--dropout", type=float, default=0.5, help='the dropout rate')
    cmd.add_argument("--depth", type=int, default=2, help='the depth of lstm')
    cmd.add_argument("--lr", type=float, default=0.01, help='the learning rate.')
    cmd.add_argument("--lr_decay", type=float, default=0, help='the learning rate decay.')
    cmd.add_argument("--clip_grad", type=float, default=5, help='the tense of clipped grad.')

    opt = cmd.parse_args()

    print(opt)
    torch.manual_seed(opt.seed)
    random.seed(opt.seed)
    if opt.gpu >= 0:
        torch.cuda.set_device(opt.gpu)
        if opt.seed > 0:
            torch.cuda.manual_seed(opt.seed)

    if opt.one_turn_c:
        print('Start one_turn_classification')
        one_turn_classification(opt)

    # setting logging
    # DEBUG = False
    # DEBUG = True
    # if DEBUG or opt.debug:
    #     logging.basicConfig(format='%(asctime)-15s %(levelname)s: %(message)s', level=logging.DEBUG)
    # else:
    #     logging.basicConfig(format='%(asctime)-15s %(levelname)s: %(message)s', level=logging.INFO)


if __name__ == '__main__':
    main()
