# coding: utf-8

"""
Created on June 5, 2018

a action classify based user simulator

-- user_goals_first_turn_template.revised.v1.p: all goals
-- user_goals_first_turn_template.part.movie.v1.p: moviename in goal.inform_slots
-- user_goals_first_turn_template.part.nomovie.v1.p: no moviename in goal.inform_slots

@author: atma
"""
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
import argparse, json, random, copy
import numpy as np
from deep_dialog import dialog_config

from .usersim_rule import RuleSimulator
from .nn_models import ClassifyLayer, MultiLableClassifyLayer
from .action_generation import *
from .prepare_data import *


DATA_MARK = dialog_config.DATA_MARK[0]
EXPR_DIR = dialog_config.EXPR_DIR[DATA_MARK]


class SuperviseUserSimulator(RuleSimulator):
    def __init__(self, movie_dict=None, act_set=None, slot_set=None, start_set=None, params=None, use_cuda=False,
                 model_path=EXPR_DIR + 'model.pkl', dict_path=EXPR_DIR + 'extracted_no_nlg_no_nlu.dict.json',
                 rule_first_turn=False):
        print('Start supervise user simulator')
        # super(SuperviseUserSimulator, self).__init__(movie_dict, act_set, slot_set, start_set, params)
        self.movie_dict = movie_dict
        self.act_set = act_set
        self.slot_set = slot_set
        self.start_set = start_set

        self.rule_first_turn = rule_first_turn
        self.max_turn = params['max_turn']
        self.slot_err_probability = params['slot_err_probability']
        self.slot_err_mode = params['slot_err_mode']
        self.intent_err_probability = params['intent_err_probability']

        self.simulator_run_mode = params['simulator_run_mode']
        self.simulator_act_level = params['simulator_act_level']

        self.learning_phase = params['learning_phase']

        # new things
        self.use_cuda = use_cuda
        self.state_v_component = dialog_config.STATE_V_COMPONENT
        with open(model_path, 'r') as reader:
            saved_model = torch.load(reader)
            # print(saved_model.keys())
            param = saved_model['param']
            print(param)
            self.classifier = MultiLableClassifyLayer(input_size=param['input_size'], hidden_size=param['opt'].hidden_dim,
                                                      num_tags=param['num_tags'], opt=param['opt'], use_cuda=use_cuda)
            self.classifier.load_state_dict(saved_model['state_dict'])
        with open(dict_path, 'r') as reader:
            self.full_dict = json.load(reader)

        self.state_dict = {}

    def get_state_representation(self):
        state_v = []
        for v_name in self.state_v_component:
            state_v.extend(self.state_dict[v_name])
        state_v = [state_v]  # set batch size as 1
        state_v = torch.LongTensor(state_v)
        return state_v

    def initialize_episode(self):
        """ Initialize a new episode (dialog)
        state['history_slots']: keeps all the informed_slots
        state['rest_slots']: keep all the slots (which is still in the stack yet)
        """

        self.state = {}
        self.state['history_slots'] = {}
        self.state['inform_slots'] = {}
        self.state['request_slots'] = {}
        self.state['rest_slots'] = []
        self.state['turn'] = 0

        self.episode_over = False
        self.dialog_status = dialog_config.NO_OUTCOME_YET

        # self.goal =  random.choice(self.start_set)
        self.goal = self._sample_goal(self.start_set)
        self.goal['request_slots']['ticket'] = 'UNK'
        self.constraint_check = dialog_config.CONSTRAINT_CHECK_FAILURE

        """ Debug: build a fake goal mannually """
        # self.debug_falk_goal()

        # sample first action
        self.state_dict = {
            # component for vector representation
            'goal_inform_slots_v': create_one_hot_v(self.goal['inform_slots'].keys(), self.full_dict['user_inform_slot2id']),
            'goal_request_slots_v': create_one_hot_v(self.goal['request_slots'].keys(), self.full_dict['user_request_slot2id']),
            'history_slots_v': create_3state_v([], self.goal['inform_slots'].keys(), self.full_dict['user_inform_slot2id']),
            'rest_slots_v': create_3state_v(self.goal['request_slots'].keys(), [], self.full_dict['user_request_slot2id']),
            'system_diaact_v': [0 for i in range(len(self.full_dict['diaact2id']))],
            'system_inform_slots_v': [0 for i in range(len(self.full_dict['sys_inform_slot2id']))],
            'system_request_slots_v': [0 for i in range(len(self.full_dict['sys_request_slot2id']))],
            'consistency_v': [0 for i in range(len(self.full_dict['user_request_slot2id']))],
            'dialog_status_v': [0],

            # component for tracking specific slots
            'history_slots': [],  # track informed slots,
            'rest_slots': self.goal['request_slots'].keys(),  # track remained slots,
            'consistent_slots': [],  # track consistency
            'inconsistent_slots': [],  # track inconsistency
        }

        state_representation = self.get_state_representation()
        if self.rule_first_turn:
            response_action = self._sample_action()
        else:
            pred_action = self.predict_action(state_representation)
            self.fill_slot_value(pred_action)
            response_action = {}
            response_action['diaact'] = self.state['diaact']
            response_action['inform_slots'] = self.state['inform_slots']
            response_action['request_slots'] = self.state['request_slots']
            response_action['turn'] = self.state['turn']
            response_action['nl'] = ""

            # add NL to dia_act
            self.add_nl_to_action(response_action)

        current_user_turn = {
            "request_slots": response_action['request_slots'],
            "diaact": response_action['diaact'],
            "inform_slots": response_action['inform_slots'],
            "turn_id": self.state['turn'],
            "speaker": "usr",
            "utterance": '',
        }

        # update state_dict: informed slots
        self.state_dict = update_state_dict_slots(
            current_speaker='usr', turn=current_user_turn, user_goal=self.goal, old_state_dict=self.state_dict
        )
        assert (self.episode_over != 1), ' but we just started'
        return response_action

    def predict_action(self, state_representation):
        self.classifier.eval()
        output, loss = self.classifier.forward(state_representation)
        pred_action = vector2action(output[0], self.full_dict)
        return pred_action

    def fill_slot_value(self, pred_action):
        inform_slots, request_slots = {}, {}
        for slot in pred_action['inform_slots']:
            if slot in self.goal['inform_slots']:
                inform_slots[slot] = self.goal['inform_slots'][slot]
        for slot in pred_action['request_slots']:
            request_slots[slot] = 'UNK'

        self.state['diaact'] = pred_action['diaact']
        self.state['inform_slots'] = inform_slots
        self.state['request_slots'] = request_slots

    def detect_finish(self, system_action):
        if system_action['diaact'] == 'thanks':

            self.episode_over = True
            self.dialog_status = dialog_config.SUCCESS_DIALOG

            request_slot_set = copy.deepcopy(self.state['request_slots'].keys())
            if 'ticket' in request_slot_set:
                request_slot_set.remove('ticket')
            rest_slot_set = copy.deepcopy(self.state['rest_slots'])
            if 'ticket' in rest_slot_set:
                rest_slot_set.remove('ticket')

            if len(request_slot_set) > 0 or len(rest_slot_set) > 0:
                self.dialog_status = dialog_config.FAILED_DIALOG

            # check constraint
            if len(self.state_dict['inconsistent_slot']) > 0 or len(self.state_dict['inconsistent_slot']) < len(self.goal['inform_slots']):
                self.dialog_status = dialog_config.FAILED_DIALOG

        else:
            self.episode_over = False
            self.dialog_status = dialog_config.NO_OUTCOME_YET

            # deal with no value match situation
            for slot in system_action['inform_slots']:
                if system_action['inform_slots'][slot] == dialog_config.NO_VALUE_MATCH:
                    self.dialog_status = dialog_config.FAILED_DIALOG
                    self.episode_over = True

    def next(self, system_action, rule_style=True):
        if rule_style:
            return self.rule_next(system_action)
        else:
            self.state['turn'] += 2
            self.detect_finish(system_action)

            sys_act = system_action['diaact']

            if (self.max_turn > 0 and self.state['turn'] > self.max_turn):
                self.dialog_status = dialog_config.FAILED_DIALOG
                self.episode_over = True
            else:
                self.state['history_slots'].update(self.state['inform_slots'])
                self.state['inform_slots'].clear()

            last_sys_turn = {
                "request_slots": system_action['request_slot'],
                "diaact": system_action['diaact'],
                "inform_slots": system_action['inform_slot'],
                "turn_id": self.state['turn'] - 1,
                "speaker": "sys",
                "utterance": '',
            }

            # update state_dict: rest slot, consistency & inconsistency slot
            self.state_dict = update_state_dict_slots(
                current_speaker='sys', turn=last_sys_turn, user_goal=self.goal, old_state_dict=self.state_dict
            )
            # update state_dict: vector
            state_representation, self.state_dict = update_state_dict_vector(
                user_goal=self.goal,
                state_v_component=self.state_v_component,
                state_dict=self.state_dict,
                last_sys_turn=last_sys_turn,
                user_inform_slot2id=self.full_dict['user_inform_slot2id'],
                user_request_slot2id=self.full_dict['user_request_slot2id'],
                sys_inform_slot2id=self.full_dict['sys_inform_slot2id'],
                sys_request_slot2id=self.full_dict['sys_request_slot2id'],
                diaact2id=self.full_dict['diaact2id'],
                dialog_status=0,
            )
            pred_action = self.predict_action(state_representation)
            self.fill_slot_value(pred_action)
            response_action = {}
            response_action['diaact'] = self.state['diaact']
            response_action['inform_slots'] = self.state['inform_slots']
            response_action['request_slots'] = self.state['request_slots']
            response_action['turn'] = self.state['turn']
            response_action['nl'] = ""

            current_user_turn = {
                "request_slots": response_action['request_slot'],
                "diaact": response_action['diaact'],
                "inform_slots": response_action['inform_slot'],
                "turn_id": self.state['turn'],
                "speaker": "usr",
                "utterance": '',
            }

            # update state_dict: informed slots
            self.state_dict = update_state_dict_slots(
                current_speaker='usr', turn=current_user_turn, user_goal=self.goal, old_state_dict=self.state_dict
            )

            # add NL to dia_act
            self.add_nl_to_action(response_action)
            return response_action, self.episode_over, self.dialog_status

    def rule_next(self, system_action):
        """ Generate next User Action based on last System Action """

        self.state['turn'] += 2
        self.episode_over = False
        self.dialog_status = dialog_config.NO_OUTCOME_YET

        sys_act = system_action['diaact']

        if (self.max_turn > 0 and self.state['turn'] > self.max_turn):
            self.dialog_status = dialog_config.FAILED_DIALOG
            self.episode_over = True
            self.state['diaact'] = "closing"
        else:
            self.state['history_slots'].update(self.state['inform_slots'])
            self.state['inform_slots'].clear()

            if sys_act == "inform":
                self.response_inform(system_action)
            elif sys_act == "multiple_choice":
                self.response_multiple_choice(system_action)
            elif sys_act == "request":
                self.response_request(system_action)
            elif sys_act == "thanks":
                self.response_thanks(system_action)
            elif sys_act == "confirm_answer":
                self.response_confirm_answer(system_action)
            elif sys_act == "closing":
                self.episode_over = True
                self.state['diaact'] = "thanks"


        self.corrupt(self.state)

        response_action = {}
        response_action['diaact'] = self.state['diaact']
        response_action['inform_slots'] = self.state['inform_slots']
        response_action['request_slots'] = self.state['request_slots']
        response_action['turn'] = self.state['turn']
        response_action['nl'] = ""

        # add NL to dia_act
        self.add_nl_to_action(response_action)
        return response_action, self.episode_over, self.dialog_status
