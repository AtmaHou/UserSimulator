# coding:utf-8

import argparse, json, copy, os, sys, logging
import cPickle

# default_target_file = './deep_dialog/data/user_goals_all_turns_template.p'
# default_target_file = './deep_dialog/models/nlg/lstm_tanh_relu_[1468202263.38]_2_0.610.p'
default_target_file = './deep_dialog/models/nlu/lstm_[1468447442.91]_39_80_0.921.p'


def pickle2json(target_file_path):
    with open(target_file_path, 'r') as reader,\
            open(target_file_path.replace('.pkl', '_new.json').replace('.pickle', '_new.json').replace('.p', '_new.json'), 'w') as writer:
        data = cPickle.load(reader)
        logging.info(sys.getsizeof(data))
        try:
            json.dump(data, writer, indent=2)
        except TypeError as e:
            print data
            print 'Type Error!', e


def pickle2pickle(target_file_path):
    with open(target_file_path, 'r') as reader,\
            open(target_file_path.replace('.pkl', '_new.pkl').replace('.pickle', '_new.pkl').replace('.p', '_new.pkl'), 'wb') as writer:
        data = cPickle.load(reader)
        logging.info(sys.getsizeof(data))
        cPickle.dump(data, writer)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p2j', type=str, default=default_target_file,
                        help='path to the .p or .pkl or .pickle dictionary file')
    parser.add_argument('-p2p', type=str, default=default_target_file,
                        help='path to the .p or .pkl or .pickle dictionary file')
    args = parser.parse_args()
    if args.p2j:
        pickle2json(args.p2j)
    if args.p2p:
        pickle2pickle(args.p2p)
