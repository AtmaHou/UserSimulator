'''
Created on May 17, 2016

@author: xiul, t-zalipt
'''

sys_request_slots = ['moviename', 'theater', 'starttime', 'date', 'numberofpeople', 'genre', 'state', 'city', 'zip', 'critic_rating', 'mpaa_rating', 'distanceconstraints', 'video_format', 'theater_chain', 'price', 'actor', 'description', 'other', 'numberofkids']
sys_inform_slots = ['moviename', 'theater', 'starttime', 'date', 'genre', 'state', 'city', 'zip', 'critic_rating', 'mpaa_rating', 'distanceconstraints', 'video_format', 'theater_chain', 'price', 'actor', 'description', 'other', 'numberofkids', 'taskcomplete', 'ticket']

start_dia_acts = {
    #'greeting':[],
    'request':['moviename', 'starttime', 'theater', 'city', 'state', 'date', 'genre', 'ticket', 'numberofpeople']
}

user_request_slots = sys_inform_slots
user_inform_slots = sys_request_slots


################################################################################
# Dialog status
################################################################################
FAILED_DIALOG = -1
SUCCESS_DIALOG = 1
NO_OUTCOME_YET = 0

# Rewards
SUCCESS_REWARD = 50
FAILURE_REWARD = 0
PER_TURN_REWARD = 0

################################################################################
#  Special Slot Values
################################################################################
I_DO_NOT_CARE = "I do not care"
NO_VALUE_MATCH = "NO VALUE MATCHES!!!"
TICKET_AVAILABLE = 'Ticket Available'

################################################################################
#  Constraint Check
################################################################################
CONSTRAINT_CHECK_FAILURE = 0
CONSTRAINT_CHECK_SUCCESS = 1

################################################################################
#  NLG Beam Search
################################################################################
nlg_beam_size = 10

################################################################################
#  run_mode: 0 for dia-act; 1 for NL; 2 for no output
################################################################################
run_mode = 0
auto_suggest = 0

################################################################################
#   A Basic Set of Feasible actions to be Consdered By an RL agent
################################################################################
feasible_actions = [
    ############################################################################
    #   greeting actions
    ############################################################################
    #{'diaact':"greeting", 'inform_slots':{}, 'request_slots':{}},
    ############################################################################
    #   confirm_question actions
    ############################################################################
    {'diaact':"confirm_question", 'inform_slots':{}, 'request_slots':{}},
    ############################################################################
    #   confirm_answer actions
    ############################################################################
    {'diaact':"confirm_answer", 'inform_slots':{}, 'request_slots':{}},
    ############################################################################
    #   thanks actions
    ############################################################################
    {'diaact':"thanks", 'inform_slots':{}, 'request_slots':{}},
    ############################################################################
    #   deny actions
    ############################################################################
    {'diaact':"deny", 'inform_slots':{}, 'request_slots':{}},
]

user_feasible_action = feasible_actions[:]  # copy another version

############################################################################
#   Adding the inform actions
############################################################################
for slot in sys_inform_slots:
    feasible_actions.append({'diaact':'inform', 'inform_slots':{slot:"PLACEHOLDER"}, 'request_slots':{}})
for slot in user_inform_slots:
    user_feasible_action.append({'diaact':'inform', 'inform_slots':{slot:"PLACEHOLDER"}, 'request_slots':{}})

############################################################################
#   Adding the request actions
############################################################################
for slot in sys_request_slots:
    feasible_actions.append({'diaact':'request', 'inform_slots':{}, 'request_slots': {slot: "UNK"}})
for slot in user_request_slots:
    user_feasible_action.append({'diaact':'request', 'inform_slots':{}, 'request_slots': {slot: "UNK"}})

############################################################################
#   File path
############################################################################
LOG_DIR = 'E:/Projects/Research/TaskOrientedDialogue/data/TC-bot/log/'
DATA_DIR = 'E:/Projects/Research/TaskOrientedDialogue/data/TC-bot/data/'

TRAIN_LOG_PATH = LOG_DIR + 'no_nlg_no_nlu.log'
EXTRACTED_LOG_DATA_PATH = LOG_DIR + 'extracted_no_nlg_no_nlu.json'
