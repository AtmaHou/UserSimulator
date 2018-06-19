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
from .classifier import ClassifyLayer, MultiLableClassifyLayer
from .action_generation import *
from .prepare_data import *

LOG_PATH = dialog_config.EXTRACTED_LOG_DATA_PATH
EXPR_DIR = dialog_config.EXPR_DIR



class SuperviseUserSimulator(RuleSimulator):
    def __init__(self, movie_dict=None, act_set=None, slot_set=None, start_set=None, params=None, use_cuda=False,
                 model_path=EXPR_DIR + 'model.pkl', dict_path=EXPR_DIR + 'extracted_no_nlg_no_nlu.dict.json'):
        super(SuperviseUserSimulator, self).__init__(movie_dict, act_set, slot_set, start_set, params)
        self.use_cuda = use_cuda
        # self.state_representation_size = (
        #     len(dialog_config.user_inform_slots) +  # history slots
        #     len(dialog_config.user_inform_slots) +  # inform slots
        #     len(dialog_config.user_request_slots) +  # request slots
        #     len(dialog_config.user_inform_slots) + # rest slots
        #     len(dialog_config.feasible_action) +  # System action
        #     3  # dialog status
        # )
        # self.action2id = {}
        # self.id2action = {}
        # for ind, action in enumerate(dialog_config.user_feasiable_action):
        #     self.action2id[action] = ind
        #     self.id2action[ind] =action

        # self.action_classifier = ClassifyLayer(
        #     input_size=self.state_representation_size,
        #     num_tags=len(dialog_config.user_feasiable_action),
        #     label2id=self.action2id,
        #     use_cuda=self.use_cuda
        # )
        self.state_v_component = dialog_config.STATE_V_COMPONENT
        with open(model_path, 'r') as reader:
            saved_model = torch.load(reader)
            param = saved_model['param']
            self.classifier = MultiLableClassifyLayer(input_size=param['input_size'], hidden_size=param['hidden_dim'],
                                                      num_tags=param['num_tags'], opt=param['opt'], use_cuda=use_cuda)
            self.classifier.load_state_dict(saved_model['state_dict'])
        with open(dict_path, 'r') as reader:
            self.full_dict = json.load(reader)

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
        state_v = []
        for v_name in self.state_v_component:
            state_v.extend(self.state_dict[v_name])
        user_action = self._sample_action()
        assert (self.episode_over != 1), ' but we just started'
        return user_action

    def predict_action(self, state_representation):
        self.classifier.eval()
        output, loss = self.classifier.forward(state_representation)
        pred_action = vector2action(output, self.full_dict)
        return pred_action

    # def fill_vectors(self, vector, reference, full_set):
    #     for value in reference:
    #         vector[full_set.index(value)] = 1
    #     return vector

    # def prepare_user_state_representation(self, system_action):
    #     # init vectors
    #     history_slots_vector = np.zeros(len(dialog_config.user_inform_slots))
    #     inform_slots_vector = np.zeros(len(dialog_config.user_inform_slots))  # inform slots
    #     request_slots_vector = np.zeros(len(dialog_config.user_request_slots))  # request slots
    #     rest_slots_vector = np.zeros(len(dialog_config.user_inform_slots))  # rest slots
    #     system_action_vector = np.zeros(len(dialog_config.feasible_action))  # system action
    #     dialog_status_vector = np.zeros(1)
    #
    #     # fill the representation
    #     self.fill_vectors(history_slots_vector, self.state['history_slots'].keys, dialog_config.user_inform_slots)
    #     self.fill_vectors(inform_slots_vector, self.state['inform_slots'].keys, dialog_config.user_inform_slots)
    #     self.fill_vectors(request_slots_vector, self.state['request_slots'].keys, dialog_config.user_request_slots)
    #     self.fill_vectors(rest_slots_vector, self.state['rest_slots'], dialog_config.user_inform_slots)
    #     self.fill_vectors(system_action_vector, system_action, dialog_config.user_inform_slots)
    #     dialog_status_vector = self.dialog_status  # -1, 0, 1 failed, no out come, success(the representation here is a little different to TC bot)
    #     raise Exception  # check system action component, change the representation to training data style
    #
    #     # stack vector
    #     state_representation = np.hstack((
    #         history_slots_vector,
    #         inform_slots_vector,
    #         request_slots_vector,
    #         rest_slots_vector,
    #         system_action_vector,
    #         dialog_status_vector
    #     ))
    #     return state_representation

    # def train_and_test_action_classifier(self, LOG_PATH):
    #     with open(LOG_PATH) as reader:
    #         log_data = json.loads(reader)
    #     all_sample = log_data['warm_start_data'] + log_data['train_data']


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

    def next(self, system_action, rule_style=True):
        if rule_style:
            return self.rule_next(system_action)
        else:
            self.state['turn'] += 2
            self.episode_over = False
            self.dialog_status = dialog_config.NO_OUTCOME_YET

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

            state_representation, self.state_dict = new_state(
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
            # TODO inconsistency update
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
