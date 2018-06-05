# coding:utf-8
from __future__ import print_function
import argparse
import os
import json
import nlg_tool
import random


DATA_DIR = '/users4/ythou/Projects/TaskOrientedDialogue/code/TC-Bot/src/deep_dialog/data/'
SPLIT_DATA = [0.8, 0.1, 0.1]


def tsv_output(data, path):
    with open(path, 'w') as writer:
        for sample in data:
            writer.write('{0}\t{1}\n'.format(sample[0], sample[1]))


def src_tgt_output(data, path):
    with open(path + '-src', 'w') as src_writer, open(path + '-tgt', 'w') as tgt_writer:
        for sample in data:
            src_writer.write('{0}\n'.format(sample[0]))
            tgt_writer.write('{0}\n'.format(sample[1]))


def get_seq_action_data(input_path, train_path, dev_path, test_path, data_format, split_data=None):
    '''

    :param input_path:
    :param train_path:
    :param dev_path:
    :param test_path:
    :param data_format:
    :param split_data: None for not split data, pass list to give split rate,eg , [0.8, 0.1, 01]
    :return:
    '''
    with open(input_path, 'r') as reader:
        json_data = json.load(reader)
    all_data = []
    for dia_act in json_data["dia_acts"]:
        for condition in json_data["dia_acts"][dia_act]:
            source = ["$" + dia_act + "$"]
            target = ' '.join(nlg_tool.treebank_tokenizer(condition['nl']['usr']))

            for request_slot in condition['request_slots']:
                source.append('request')
                source.append('$' + request_slot + '$')
            for inform_slot in condition['inform_slots']:
                source.append('inform')
                source.append('$' + inform_slot + '$')

            source = ' '.join(source)
            all_data.append([source, target])
    print(train_path, len(all_data))
    # print(all_data, train_path, len(all_data))
    if split_data:
        random.shuffle(all_data)
        train_num = int(len(all_data) * split_data[0])
        dev_num = int(len(all_data) * split_data[1])
        test_num = int(len(all_data) * split_data[2])
        train_data = all_data[: train_num]
        dev_data = all_data[train_num: train_num + dev_num]
        test_data = all_data[train_num + dev_num:]
    else:
        train_data = all_data[:]
        dev_data = []
        test_data = []
    print('data set stats:', len(train_data), len(dev_data), len(test_data))
    if data_format == data_format == 'tsv':
        tsv_output(train_data, train_path)
        tsv_output(dev_data, dev_path)
        tsv_output(test_data, test_path)
    elif data_format == 'src-tgt':
        src_tgt_output(train_data, train_path)
        src_tgt_output(dev_data, dev_path)
        src_tgt_output(test_data, test_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, default=DATA_DIR + "dia_act_nl_pairs.v6.json",
                        help="file path for raw data file")
    parser.add_argument("--train_path", type=str, default=DATA_DIR + "nlg/train.txt", help="file path for train path")
    parser.add_argument("--dev_path", type=str, default=DATA_DIR + "nlg/dev.txt", help="file path for dev path")
    parser.add_argument("--test_path", type=str, default=DATA_DIR + "nlg/test.txt", help="file path for test path")
    parser.add_argument('--source_format', type=str, default='seq_action',
                        help='select data\'s source end format', choices=['seq_action'])
    parser.add_argument('--data_format', type=str, default='tsv',
                        help='select data\'s source end format', choices=['tsv', 'src-tgt'])
    parser.add_argument('--split_data', action='store_true', help='select whether to use ')

    opt = parser.parse_args()

    if opt.source_format == 'seq_action':
        get_seq_action_data(
            input_path=opt.input_path,
            train_path=opt.train_path,
            dev_path=opt.dev_path,
            test_path=opt.test_path,
            data_format=opt.data_format,
            split_data=SPLIT_DATA if opt.split_data else None,
        )
    else:
        raise NotImplementedError
