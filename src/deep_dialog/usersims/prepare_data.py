# coding: utf-8
from __future__ import print_function, division
import json
import re
import sys
sys.path.append("..")
import dialog_config

LOG_PATH = dialog_config.TRAIN_LOG_PATH
OUTPUT_PATH = dialog_config.EXTRACTED_LOG_DATA_PATH


def process_one_dialogue(raw_data_str):
    # raw_data_str = raw_data_str.replace("New episode, user goal:\n", "")
    if not raw_data_str:
        return {}
    split_pos = raw_data_str.find('Turn 0 ')
    user_goal_str = raw_data_str[:split_pos]
    turns_str = raw_data_str[split_pos:].strip()
    user_goal = json.loads(user_goal_str)
    turns_str_lst = turns_str.split('\n')

    # Process useless lines
    if len(turns_str_lst) == 1:
        if 'Episode: ' in raw_data_str:
            return {}
        else:
            print('Unexpected 1 line situation')
    if len(turns_str_lst) == 0:
        return {}
    turns = []

    # Extract success info
    success = False
    success_judge_line = turns_str_lst[-1]
    if 'Success' in success_judge_line or 'Successful Dialog' in success_judge_line:
        success = True
    elif 'Fail' in success_judge_line or 'Failed Dialog' in success_judge_line:
        success = False
    else:
        print('Error: Wrong lines for sucess info')
        raise RuntimeError

    # Extract turns info
    for ind, turn_str in enumerate(turns_str_lst[:-1]):
        turn_str = re.sub('Turn \d*? ', '', turn_str)  # first turn don't need to
        if ind % 2 == 0:  # action line
            turn_split_pattern = ", inform_slots: |, request_slots: |, request slots: "

            splited_data = re.split(turn_split_pattern, turn_str)
            speaker_and_diaact_str, inform_slots_str, request_slots_str = splited_data
            speaker, diaact = speaker_and_diaact_str.split(': ')
            # print(inform_slots_str)
            # inform_slots = json.loads(inform_slots_str.replace("'", "\""))
            # request_slots = json.loads(request_slots_str.replace("'", "\""))
            inform_slots = eval(inform_slots_str)
            request_slots = eval(request_slots_str)
            turn_id = ind // 2
            # acts always be the first to appear
            item = {
                'turn_id': turn_id,
                'speaker': speaker,
                'diaact': diaact,
                'inform_slots': inform_slots,
                'request_slots': request_slots
            }
            turns.append(item)
        else:  # utterance line
            turn_id = ind // 2
            item = turns[turn_id]
            # check id
            if turn_id != item['turn_id']:
                print('failed to check id')
                raise RuntimeError
            if 'sys: ' in turn_str:
                speaker = 'usr'
                turn_str = turn_str.replace('sys: ', '')
            elif 'usr: ' in turn_str:
                speaker = 'sys'
                turn_str = turn_str.replace('usr: ', '')
            else:
                speaker = ''
            if not speaker and speaker != item['speaker']:
                print('Fail to match speaker')
                raise RuntimeError
            utterance = turn_str.replace('\n', '')
            item['utterance'] = utterance
    ret = {
        'user_goal': user_goal,
        'turns': turns,
        'success': success
    }
    return ret


def split_raw_data(raw_data, split_pattern):
    """
    split raw data
    for future purpose, this function may frequently change
    :param raw_data:
    :return: splited data list
    """
    splited_data = re.split(split_pattern, raw_data)
    return splited_data


def extract_warm_start_data(warm_start_data_str):
    # statics_pattern = "Warm_Start \d*? epochs, success rate .*?\n"
    statics_pattern = r"Warm_Start \d*? epochs, success rate .*?\n,*?Current experience replay buffer size .*?\n"
    warm_start_data_str = re.sub(
        statics_pattern,
        '',
        warm_start_data_str)
    split_pattern = "New episode, user goal:\n"
    splited_warm_start_data = re.split(split_pattern, warm_start_data_str)
    ret = []
    for ind, dialogue_str in enumerate(splited_warm_start_data):
        data_item = process_one_dialogue(dialogue_str)
        if data_item:  # Skip empty case
            ret.append(data_item)
        if ind % 1000 == 0:
            print(ind, 'warm up dialogue processed')
    return ret


def extract_train_data(train_data_str):
    # remove buffer for each epoch
    sub_pattern = r'simulation success rate[\s\S]*?Success rate:[\s\S]*?Success rate:[\s\S]*?Avg turns:.*?\n'
    train_data_str = re.sub(
        sub_pattern,
        '',
        train_data_str,
    )
    train_data_str = re.sub(
        'saved model in .*?\n',
        '',
        train_data_str
    )
    train_data_str = re.sub(
        'Episode: .*?\n',
        '',
        train_data_str
    )
    if 'Success rate:' in train_data_str[-1000:]:
        train_data_str = train_data_str.replace('Success rate: 406 / 500 Avg reward: 50.11 Avg turns: 16.66', '')
        # print(train_data_str[-1000:])
        print("Fix last line's extra info")
    split_pattern = "New episode, user goal:\n"
    splited_train_data = re.split(split_pattern, train_data_str)
    ret = []
    for ind, dialogue_str in enumerate(splited_train_data):
        data_item = process_one_dialogue(dialogue_str)
        if data_item:  # Skip empty case
            ret.append(data_item)
        if ind % 1000 == 0:
            print(ind, 'train dialogue processed')
    return ret


def prepare_dataset(raw_data_path, output_path, raw_data_type='train_log'):
    """
    Extract data from raw data (i.e. training / testing stdout log or structured log) to train user simulator
    :param raw_data_path:
    :param raw_data_type:
    :return:
    """
    with open(raw_data_path, 'r') as reader:
        raw_data = reader.read()
    if raw_data_type == 'train_log':
        # split data
        split_pattern = "warm_start finished, start RL training ...\n|warm_start starting ...\n"
        expr_param_str, warm_start_data_str, train_data_str = split_raw_data(raw_data, split_pattern)

        # extract structured data
        expr_param = json.loads(expr_param_str.replace('Dialog Parameters: ', ''))
        warm_start_data = extract_warm_start_data(warm_start_data_str)
        train_data = extract_train_data(train_data_str)
        with open(output_path, 'w') as writer:
            json.dump(
                {
                    'expr_param': expr_param,
                    'warm_start_data': warm_start_data,
                    'train_data': train_data
                },
                writer,
                indent=2
            )
    else:
        raise NotImplementedError


if __name__ == '__main__':
    prepare_dataset(raw_data_path=LOG_PATH, output_path=OUTPUT_PATH)
