
"""
Created on June 7, 2016

a rule-based user simulator

@author: xiul, t-zalipt
"""
from deep_dialog import dialog_config
import random
import copy


class UserSimulator:
    """ Parent class for all user sims to inherit from """

    def __init__(self, movie_dict=None, act_set=None, slot_set=None, start_set=None, params=None):
        """ Constructor shared by all user simulators """
        
        self.movie_dict = movie_dict
        self.act_set = act_set
        self.slot_set = slot_set
        self.start_set = start_set
        
        self.max_turn = params['max_turn']
        self.slot_err_probability = params['slot_err_probability']
        self.slot_err_mode = params['slot_err_mode']
        self.intent_err_probability = params['intent_err_probability']
        

    def initialize_episode(self):
        """ Initialize a new episode (dialog)"""

        print "initialize episode called, generating goal"
        self.goal =  random.choice(self.start_set)
        self.goal['request_slots']['ticket'] = 'UNK'
        episode_over, user_action = self._sample_action()
        assert (episode_over != 1),' but we just started'
        return user_action


    def next(self, system_action):
        pass
    
    
    
    def set_nlg_model(self, nlg_model):
        self.nlg_model = nlg_model  
    
    def set_nlu_model(self, nlu_model):
        self.nlu_model = nlu_model
    
    
    
    def add_nl_to_action(self, user_action):
        """ Add NL to User Dia_Act """
        
        user_nlg_sentence = self.nlg_model.convert_diaact_to_nl(user_action, 'usr')
        user_action['nl'] = user_nlg_sentence
        
        if self.simulator_act_level == 1:
            user_nlu_res = self.nlu_model.generate_dia_act(user_action['nl']) # NLU
            if user_nlu_res != None:
                #user_nlu_res['diaact'] = user_action['diaact'] # or not?
                user_action.update(user_nlu_res)

    # ========= My function =============

    def detect_finish(self, system_action):
        if system_action['diaact'] == 'thanks':

            self.episode_over = True
            self.dialog_status = dialog_config.SUCCESS_DIALOG

            request_slot_set = copy.deepcopy(self.state['request_slots'].keys())
            if 'ticket' in request_slot_set:
                request_slot_set.remove('ticket')
            rest_slot_set = copy.deepcopy(self.state_dict['rest_slots'])
            if 'ticket' in rest_slot_set:
                rest_slot_set.remove('ticket')

            # print('@@@@@@@@@@@@ debug detect finish', self.state_dict)
            if len(request_slot_set) > 0 or len(rest_slot_set) > 0:
                self.dialog_status = dialog_config.FAILED_DIALOG

            # check constraint
            if len(self.state_dict['inconsistent_slots']) > 0 or len(self.state_dict['consistent_slots']) < len(self.goal['inform_slots']):
                self.dialog_status = dialog_config.FAILED_DIALOG

        else:
            self.episode_over = False
            self.dialog_status = dialog_config.NO_OUTCOME_YET

            # deal with no value match situation
            for slot in system_action['inform_slots']:
                if system_action['inform_slots'][slot] == dialog_config.NO_VALUE_MATCH:
                    self.dialog_status = dialog_config.FAILED_DIALOG
                    self.episode_over = True

    def fill_slot_value(self, pred_action):
        inform_slots, request_slots = {}, {}
        for slot in pred_action['inform_slots']:
            if slot in self.goal['inform_slots']:
                inform_slots[slot] = self.goal['inform_slots'][slot]
        for slot in pred_action['request_slots']:
            request_slots[slot] = 'UNK'

        self.state['diaact'] = pred_action['diaact'] if pred_action['diaact'] else 'inform'
        self.state['inform_slots'] = inform_slots
        self.state['request_slots'] = request_slots
