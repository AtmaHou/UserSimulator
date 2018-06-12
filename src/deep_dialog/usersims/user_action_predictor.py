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
from .classifier import ClassifyLayer

LOG_PATH = dialog_config.EXTRACTED_LOG_DATA_PATH


class SuperviseUserSimulator(RuleSimulator):
    def __init__(self, movie_dict=None, act_set=None, slot_set=None, start_set=None, params=None, use_cuda=False):
        super(SuperviseUserSimulator, self).__init__(movie_dict, act_set, slot_set, start_set, params)
        self.use_cuda = use_cuda
        self.state_representation_size = (
            len(dialog_config.user_inform_slots) +  # history slots
            len(dialog_config.user_inform_slots) +  # inform slots
            len(dialog_config.user_request_slots) +  # request slots
            len(dialog_config.user_inform_slots) + # rest slots
            len(dialog_config.feasible_action) +  # System action
            3  # dialog status
        )
        self.action2id = {}
        self.id2action = {}
        for ind, action in enumerate(dialog_config.user_feasiable_action):
            self.action2id[action] = ind
            self.id2action[ind] =action

        self.action_classifier = ClassifyLayer(
            input_size=self.state_representation_size,
            num_tags=len(dialog_config.user_feasiable_action),
            label2id=self.action2id,
            use_cuda=self.use_cuda
        )

    def predict_action(self, state_representation):
        pred_action_id, score_float = self.action_classifier(state_representation)
        pred_action = self.id2action[pred_action_id]
        return pred_action

    def fill_vectors(self, vector, reference, full_set):
        for value in reference:
            vector[full_set.index(value)] = 1
        return vector

    def prepare_user_state_representation(self, system_action):
        # init vectors
        history_slots_vector = np.zeros(len(dialog_config.user_inform_slots))
        inform_slots_vector = np.zeros(len(dialog_config.user_inform_slots))  # inform slots
        request_slots_vector = np.zeros(len(dialog_config.user_request_slots))  # request slots
        rest_slots_vector = np.zeros(len(dialog_config.user_inform_slots))  # rest slots
        system_action_vector = np.zeros(len(dialog_config.feasible_action))  # system action
        dialog_status_vector = np.zeros(1)

        # fill the representation
        self.fill_vectors(history_slots_vector, self.state['history_slots'].keys, dialog_config.user_inform_slots)
        self.fill_vectors(inform_slots_vector, self.state['inform_slots'].keys, dialog_config.user_inform_slots)
        self.fill_vectors(request_slots_vector, self.state['request_slots'].keys, dialog_config.user_request_slots)
        self.fill_vectors(rest_slots_vector, self.state['rest_slots'], dialog_config.user_inform_slots)
        self.fill_vectors(system_action_vector, system_action, dialog_config.user_inform_slots)
        dialog_status_vector = self.dialog_status  # -1, 0, 1 failed, success, no out come
        raise Exception  # check system action component

        # stack vector
        state_representation = np.hstack((
            history_slots_vector,
            inform_slots_vector,
            request_slots_vector,
            rest_slots_vector,
            system_action_vector,
            dialog_status_vector
        ))
        return state_representation

    def train_and_test_action_classifier(self, LOG_PATH):
        with open(LOG_PATH) as reader:
            log_data = json.loads(reader)
        all_sample = log_data['warm_start_data'] + log_data['train_data']


    def fill_slot_value(self, response):
        raise NotImplementedError
        return response

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
            response_action = self.predict_action(system_action)

            self.fill_slot_value(response_action)

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