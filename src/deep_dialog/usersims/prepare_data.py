# coding: utf-8
from __future__ import print_function, division
import json
import re
import sys
import os
sys.path.append("..")
import dialog_config
import logging
import copy

LOG_PATH = dialog_config.TRAIN_LOG_PATH
EXTRACTED_LOG_DATA_PATH = dialog_config.EXTRACTED_LOG_DATA_PATH
EXPR_DIR = dialog_config.EXPR_DIR

# DEBUG = False
DEBUG = True

# setting logging
if DEBUG:
    logging.basicConfig(filename='', format='%(asctime)-15s %(levelname)s: %(message)s', level=logging.DEBUG)
else:
    logging.basicConfig(filename='', format='%(asctime)-15s %(levelname)s: %(message)s', level=logging.INFO)


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


def create_one_hot_v(selected_set, reference):
    """

    :param selected_set:
    :param reference: list/dict, is full set list or a item2id dict
    :return:
    """
    v = [0 for i in range(len(reference))]
    if type(reference) == list:
        for x in selected_set:
            v[reference.index(x)] = 1
    elif type(reference) == dict:  # item2id dict
        for x in selected_set:
            v[reference[x]] = 1
    else:
        raise TypeError
    return v


def create_3state_v(active_set, negative_set, reference):
    """

    :param active_set:
    :param negative_set:
    :param reference: dict, item2id dict
    :return:
    """
    v = [0 for i in range(len(reference))]
    active_set = set(active_set)
    negative_set = set(negative_set)
    full_set = set(reference.keys())
    if not active_set.issubset(full_set) or not negative_set.issubset(full_set):
        # print(active_set.issubset(full_set), negative_set.issubset(full_set))
        logging.error("Error: Unexpected set value")
        logging.error("Debug: a - f:{}, n - f:{}".format(active_set - full_set, negative_set - full_set))
        # logging.error("a:{}, n{}, r{}".format(active_set, negative_set, reference))
        raise RuntimeError
    if active_set & negative_set:
        logging.error('Error: Active and negative in the meantime!')
        raise RuntimeError
    for item in reference:
        if item in active_set:
            v[reference[item]] = 1
        elif item in negative_set:
            v[reference[item]] = -1
    return v


def new_state(user_goal, state_v_component, state_dict, last_sys_turn, user_inform_slot2id, user_request_slot2id,
                      sys_inform_slot2id, sys_request_slot2id, diaact2id, dialog_status):
    state_v = []
    new_state_dict = copy.deepcopy(state_dict)
    if not last_sys_turn:  # for first turn
        pass
    else:
        new_state_dict['history_slots_v'] = create_3state_v(
            active_set=state_dict['history_slots'],
            negative_set=set(user_goal['inform_slots'].keys()) - set(state_dict['history_slots']),
            reference=user_inform_slot2id
        )
        new_state_dict['rest_slots_v'] = create_3state_v(
            active_set=state_dict['rest_slots'],
            negative_set= set(user_goal['request_slots'].keys()) - set(state_dict['rest_slots']),
            reference=user_request_slot2id
        )
        new_state_dict['system_diaact_v'] = create_one_hot_v([last_sys_turn['diaact']], diaact2id)
        new_state_dict['system_inform_slots_v'] = create_one_hot_v(last_sys_turn['inform_slots'].keys(), sys_inform_slot2id)
        new_state_dict['system_request_slots_v'] = create_one_hot_v(last_sys_turn['request_slots'].keys, sys_request_slot2id)
        new_state_dict['consistency_v'] = create_3state_v(
            active_set=state_dict['consistent_slots'],
            negative_set=state_dict['inconsistent_slots'],
            reference=user_request_slot2id
        )
        new_state_dict['dialog_status_v'] = [dialog_status]
    for v_name in state_v_component:
        state_v.extend(state_dict[v_name])
    return state_v, new_state_dict


def cook_one_dialogue(dialogue_item, user_inform_slot2id, user_request_slot2id,
                      sys_inform_slot2id, sys_request_slot2id, diaact2id):
    """
    Given One dialogue, split it into turn level samples,
    :return:
    """
    user_goal = dialogue_item['user_goal']
    turns = dialogue_item['turns']
    samples = []
    labels = []

    # initialize state representation
    # Use this vector to keep order
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
    state_dict = {
        # component for vector representation
        'goal_inform_slots_v': create_one_hot_v(user_goal['inform_slots'].keys(), user_inform_slot2id),
        'goal_request_slots_v': create_one_hot_v(user_goal['request_slots'].keys(), user_request_slot2id),
        'history_slots_v': create_3state_v([], user_goal['inform_slots'].keys(), user_inform_slot2id),
        'rest_slots_v': create_3state_v(user_goal['request_slots'].keys(), [], user_request_slot2id),
        'system_diaact_v': [0 for i in range(len(diaact2id))],
        'system_inform_slots_v': [0 for i in range(len(sys_inform_slot2id))],
        'system_request_slots_v': [0 for i in range(len(sys_request_slot2id))],
        'consistency_v': [0 for i in range(len(user_request_slot2id))],
        'dialog_status_v': [0],

        # component for tracking specific slots
        'history_slots': [],  # track informed slots,
        'rest_slots': [user_goal['request_slots'].keys()],  # track remained slots,
        'consistent_slot': [],  # track consistency
        'inconsistent_slot': [],  # track inconsistency
    }

    # Extract samples
    last_sys_turn = None
    for ind, turn in enumerate(turns):
        if turn['speaker'] == 'sys':
            last_sys_turn = turn
            for slot, value in turn['inform_slots'].values():
                # update rest slot
                state_dict['rest_slots'].remove(slot)
                # update consistent slot and inconsistent slot
                if slot in user_goal['request_slot']:
                    if value != user_goal['request_slot'][slot] and slot not in state_dict['inconsistent_slot']:
                        state_dict['inconsistent_slot'].append(slot)
                    if value == user_goal['request_slot'][slot] and slot not in state_dict['consistent_slot']:
                        state_dict['consistent_slot'].append(slot)

        elif turn['speaker'] == 'usr':
            # build state vector and label
            state_v, state_dict = new_state(
                user_goal, state_v_component, state_dict, last_sys_turn,
                user_inform_slot2id, user_request_slot2id,
                sys_inform_slot2id, sys_request_slot2id, diaact2id,
                dialog_status=(int(ind==len(turns)-1))  # Last turn represent finish.
            )
            label = create_one_hot_v([turn['diaact']], diaact2id) + \
                    create_one_hot_v(turn['inform_slots'].keys(), user_inform_slot2id) + \
                    create_one_hot_v(turn['request_slots'].keys(), user_request_slot2id)
            samples.append(state_v)
            labels.append(label)

            if DEBUG:
                logging.debug("DEBUG one turn:state_dict:{},\n turn:{},\n last_sys_turn:{}\n".format(state_dict, turn, last_sys_turn))
            for slot in turn['inform_slots']:
                if slot not in state_dict['history_slots']:
                    state_dict['history_slots'].append(slot)

    return samples, labels


def lst2dict(lst):
    lst = set(lst)
    item2id = dict([(item, ind) for ind, item in enumerate(lst)])
    id2item = dict([(ind, item) for ind, item in enumerate(lst)])
    return item2id, id2item


def dump_data(data_item, source_data_path, output_dir, data_mark):
    soure_data_name = os.path.basename(source_data_path)
    with open(os.path.join(output_dir, soure_data_name + '.' + data_mark + '.json')) as writer:
        json.dump(data_item, writer, indent=2)


def cook_dataset(target_dataset_path, output_dir, split_rate, dump_processed_data=False):
    """
    Given split dataset as a list, cook the data set into vector representation.
    :return:
    """
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    # create dict for slot_type and diaact
    all_user_request_slot = dialog_config.user_request_slots
    all_user_inform_slot = dialog_config.user_inform_slots
    user_inform_slot2id, id2user_inform_slot = lst2dict(all_user_inform_slot)
    user_request_slot2id, id2user_request_slot = lst2dict(all_user_request_slot)
    sys_inform_slot2id, id2sys_inform_slot = lst2dict(dialog_config.sys_inform_slots)
    sys_request_slot2id, id2sys_request_slot = lst2dict(dialog_config.sys_request_slots)
    diaact2id, id2diaact = lst2dict([action['diaact'] for action in dialog_config.feasible_actions])
    # convert data to vector
    all_sample = []
    all_label = []
    with open(target_dataset_path, 'r') as reader:
        data_set = json.load(reader)
    all_data = data_set['warm_start_data'] + data_set['train_data']
    if DEBUG:
        all_data = all_data[:1000]
    for ind, dialog_item in enumerate(all_data):
        if dialog_item['success']:  # Only use successful sample now
            samples, labels = cook_one_dialogue(
                dialog_item,
                user_inform_slot2id,
                user_request_slot2id,
                sys_inform_slot2id,
                sys_request_slot2id,
                diaact2id
            )
            all_sample.extend(samples)
            all_label.extend(labels)
        if ind % 100 == 0:
            logging.info("{0} dialogues processed".format(ind))

    train_end_idx = len(all_sample) * split_rate[0]
    dev_end_idx = len(all_sample) *split_rate[1] + train_end_idx

    # split data
    train_data = all_sample[:train_end_idx]
    dev_data = all_sample[train_end_idx: dev_end_idx]
    test_data = all_sample[dev_end_idx:]

    train_label = all_label[:train_end_idx]
    dev_label = all_label[train_end_idx:dev_end_idx]
    test_label = all_label[dev_end_idx:]

    logging.info('Finish split data: train-{0}, dev-{1}, test-{2}'.
                 format(len(train_data), len(dev_data), len(test_data)))

    all_dict = {
        'sys_inform_slot2id': sys_inform_slot2id,
        'sys_request_slot2id': sys_request_slot2id,
        'user_inform_slot2id': user_inform_slot2id,
        'user_request_slot2id': user_request_slot2id,
        'diaact2id': diaact2id,
        'id2sys_inform_slot': id2sys_inform_slot,
        'id2sys_request_slot': id2sys_request_slot,
        'id2user_inform_slot': id2user_inform_slot,
        'id2user_request_slot': id2user_request_slot,
        'id2diaact': id2diaact,
    }

    # dump data
    if dump_processed_data:
        dump_data(zip(train_data, train_label), target_dataset_path, output_dir, 'train')
        dump_data(zip(dev_data, dev_label), target_dataset_path, output_dir, 'dev')
        dump_data(zip(test_data, test_label), target_dataset_path, output_dir, 'dev')
        dump_data(all_dict, target_dataset_path, output_dir, 'dict')

    return train_data, train_label, dev_data, dev_label, test_data, test_label, all_dict


if __name__ == '__main__':
    # prepare_dataset(raw_data_path=LOG_PATH, output_path=OUTPUT_PATH)
    cook_dataset(
        target_dataset_path=EXTRACTED_LOG_DATA_PATH,
        output_dir=EXPR_DIR,
        split_rate=[0.8, 0.1, 0.1],
        dump_processed_data=True
    )
