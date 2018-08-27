# coding: utf-8
from __future__ import print_function, division
import re
import os


def get_score(file_name, log_dir):
    file_mark = file_name.replace('.log', '')
    with open(log_dir + file_name, 'r')as reader:
        results = reader.read()
        valid_score = re.findall('INFO: best_valid_f1: (.*?)\n', results)
        test_score = re.findall('INFO: test_f1: (.*?)\n', results)
        if not valid_score:
            valid_score = 'N/A'
        else:
            valid_score = valid_score[0]
        if not test_score:
            test_score = 'N/A'
        else:
            test_score = test_score[0]
    return file_mark, valid_score, test_score


def format_output(all_result):
    best_test = 0
    best_valid = 0
    best_setting = ''
    print('\t'.join(['setting', 'best valid', 'test']))
    for result in all_result:
        # if result[2] != 'N/A' and float(result[2]) > best_test:
        if result[1] != 'N/A' and float(result[1]) > best_valid:
            best_test = float(result[2])
            best_valid = float(result[1])
            best_setting = result[0]
        print('\t'.join(result))
    print('Best Setting', best_setting, 'Best valid', best_valid, 'Test', best_test)


def collect_tune_result():
    log_dir = './tune/'
    all_file_name = os.listdir(log_dir)
    all_results = []
    for name in all_file_name:
        all_results.append(get_score(file_name=name, log_dir=log_dir))
    format_output(all_results)


def get_cross_score(file_name, log_dir):
    with open(log_dir + file_name, 'r') as reader:
        results = reader.read()
        success_rate = re.findall('Progress: 500 / 500, Success rate: (.*?) Avg reward:', results)
        reward = re.findall('Progress: 500 / 500, Success rate: .*? Avg reward: (.*?) Avg turns:', results)
        turns = re.findall('Progress: 500 / 500, Success rate: .*? Avg reward: .*? Avg turns: (.*?)\n', results)

        if not success_rate:
            success_rate = 'N/A'
        else:
            success_rate = success_rate[0]
        if not reward:
            reward = 'N/A'
        else:
            reward = reward[0]
        if not turns:
            turns = 'N/A'
        else:
            turns = turns[0]
        return [file_name, success_rate, reward, turns]


def format_output_cross(all_result):
    for result in all_result:
        print(result)


def collect_cross_eval_result():
    log_dir = './cross_eval/'
    all_file_name = sorted(os.listdir(log_dir))
    all_results = []
    for name in all_file_name:
        all_results.append(get_cross_score(file_name=name, log_dir=log_dir))
    format_output_cross(all_results)


if __name__ == '__main__':
    # collect_tune_result()
    collect_cross_eval_result()
