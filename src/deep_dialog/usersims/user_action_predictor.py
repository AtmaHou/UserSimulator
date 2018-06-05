# coding: utf-8

"""
Created on June 5, 2018

a rule-based user simulator

-- user_goals_first_turn_template.revised.v1.p: all goals
-- user_goals_first_turn_template.part.movie.v1.p: moviename in goal.inform_slots
-- user_goals_first_turn_template.part.nomovie.v1.p: no moviename in goal.inform_slots

@author: xiul, t-zalipt
"""

from .usersim_rule import RuleSimulator
import argparse, json, random, copy

from deep_dialog import dialog_config


class SuperviseUserSimulator(RuleSimulator):
    def __init__(self, movie_dict=None, act_set=None, slot_set=None, start_set=None, params=None):
        super(SuperviseUserSimulator, self).__init__(movie_dict, act_set, slot_set, start_set, params)

    def predict_action(self, state_representation):
        action = None
        return action

    def train_action_classifier(self):
        pass
