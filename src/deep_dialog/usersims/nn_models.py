# coding: utf-8
from __future__ import print_function, division

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
import math
import random
import argparse
import logging
import numpy as np

from torch.optim.lr_scheduler import StepLR

import seq2seq
from seq2seq.trainer import SupervisedTrainer
from seq2seq.models import EncoderRNN, DecoderRNN  # , Seq2seq
from seq2seq.models.baseRNN import BaseRNN
from seq2seq.models.attention import Attention
from seq2seq.loss.loss import Loss
from seq2seq.loss import NLLLoss
from seq2seq.loss import Perplexity
from seq2seq.optim import Optimizer
from seq2seq.dataset import SourceField, TargetField
from seq2seq.evaluator import Predictor
from seq2seq.util.checkpoint import Checkpoint
from nltk.tokenize.treebank import TreebankWordTokenizer

try:
    from deep_dialog import dialog_config
except:
    import sys
    sys.path.append("..")
    import dialog_config

# LOG_PATH = "E:\Projects\Research\TaskOrientedDialogue\data\TC-bot\log\no_nlg_no_nlu.log"


class ClassifyLayer(nn.Module):
    def __init__(self, input_size, num_tags, label2id, label_pad='<pad>', use_cuda=False):
        super(ClassifyLayer, self).__init__()
        self.use_cuda = use_cuda
        self.num_tags = num_tags
        self.label2id = label2id
        self.label_pad = label_pad
        # print('debug:', input_size, type(input_size), num_tags, type(num_tags))
        self.hidden2tag = nn.Linear(in_features=input_size, out_features=num_tags)
        self.logsoftmax = nn.LogSoftmax(dim=2)
        tag_weights = torch.ones(num_tags)
        tag_weights[label2id[label_pad]] = 0
        self.criterion = nn.NLLLoss(tag_weights)

    def forward(self, x, y):
        """
        :param x: torch.Tensor (batch_size, seq_len, n_in)
        :param y: torch.Tensor (batch_size, seq_len)
        :return:
        """
        tag_scores = self.hidden2tag(x)
        if self.training:
            tag_scores = self.logsoftmax(tag_scores)
        if self.label2id[self.label_pad] == 0:
            _, tag_result = torch.max(tag_scores[:, :, 1:], 2)  # block <pad> label as predict output
        else:
            _, tag_result = torch.max(tag_scores, 2)  # give up to block <pad> label for efficiency
        tag_result.add_(1)
        if self.training:
            return tag_result, self.criterion(tag_scores.view(-1, self.num_tags), Variable(y).view(-1))
        else:
            return tag_result, torch.FloatTensor([0.0])

    def get_probs(self, x):
        tag_scores = self.hidden2tag(x)
        if self.training:
            tag_scores = self.logsoftmax(tag_scores)

        return tag_scores


class MultiLableClassifyLayer(nn.Module):
    def __init__(self, input_size, hidden_size, num_tags, opt, use_cuda=False):
        super(MultiLableClassifyLayer, self).__init__()

        self.main = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_tags)
        )
        self.criterion = nn.MultiLabelSoftMarginLoss()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_tags = num_tags
        self.use_cuda = use_cuda
        self.opt = opt

    def forward(self, input, golden=None):
        # print('DEBUG:', input, golden)
        input = Variable(input.float()) if type(input) != Variable else input
        if golden is not None:
            golden = Variable(golden.float()) if type(golden) != Variable else golden
        pred = self.main(input)  # For each pos, > 0 for positive tag, < 0 for negative tag
        # For each pos, 1 for positive tag, 0 for negative tag
        # print('DEBUG!!!!!!!!!!', pred, pred.size())
        classify_results = [[int(pred_j > 0) for pred_j in pred_i] for pred_i in pred.data]
        if self.training:
            return classify_results, self.criterion(pred, golden)
        else:
            return classify_results, torch.FloatTensor([0.0])


class LSTM_MultiLabelClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_tags, opt, use_cuda=False):
        super(LSTM_MultiLabelClassifier, self).__init__()

        self.encoder = nn.LSTM(
            input_size=input_size, hidden_size=hidden_size, num_layers=opt.depth,
            batch_first=True, dropout=opt.dropout, bidirectional=True
        )
        encoder_output_size = 2 * opt.hidden_dim  # because of the bi-directional
        self.classify_layer = MultiLableClassifyLayer(
            input_size=encoder_output_size, hidden_size=hidden_size, num_tags=num_tags,
            opt=opt, use_cuda=use_cuda)

        self.criterion = nn.MultiLabelSoftMarginLoss()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_tags = num_tags
        self.use_cuda = use_cuda
        self.opt = opt

    def forward(self, batch_x, batch_y=None):
        batch_size = batch_x.size(0)
        batch_x = Variable(batch_x.float())
        output, hidden_and_cell = self.encoder(batch_x)
        batch_last_hidden = output[:, -1, :]
        Variable(batch_y.float())
        if self.training:
            output, loss = self.classify_layer(batch_last_hidden, batch_y)
            return output, loss
        else:
            return output


class VanillaEncoder(BaseRNN):
    def __init__(self, input_size, max_len, hidden_size,
                 vocab_size=False, input_dropout_p=0, dropout_p=0,
                 n_layers=1, bidirectional=False, rnn_cell='lstm', variable_lengths=False):
        super(VanillaEncoder, self).__init__(vocab_size, max_len, hidden_size,
                                             input_dropout_p, dropout_p, n_layers, rnn_cell)
        self.variable_lengths = variable_lengths
        self.rnn = self.rnn_cell(input_size, hidden_size, n_layers,
                                 batch_first=True, bidirectional=bidirectional, dropout=dropout_p)

    def forward(self, input_var, input_lengths=None):
        if type(input_var) != Variable:
            input_var = Variable(input_var)
        # print('========= DEBUG =======', input_var)
        output, hidden = self.rnn(input_var)
        return output, hidden


class DecoderRNN(BaseRNN):
    r"""
    Provides functionality for decoding in a seq2seq framework, with an option for attention.

    Args:
        vocab_size (int): size of the vocabulary
        max_len (int): a maximum allowed length for the sequence to be processed
        hidden_size (int): the number of features in the hidden state `h`
        sos_id (int): index of the start of sentence symbol
        eos_id (int): index of the end of sentence symbol
        n_layers (int, optional): number of recurrent layers (default: 1)
        rnn_cell (str, optional): type of RNN cell (default: gru)
        bidirectional (bool, optional): if the encoder is bidirectional (default False)
        input_dropout_p (float, optional): dropout probability for the input sequence (default: 0)
        dropout_p (float, optional): dropout probability for the output sequence (default: 0)
        use_attention(bool, optional): flag indication whether to use attention mechanism or not (default: false)

    Attributes:
        KEY_ATTN_SCORE (str): key used to indicate attention weights in `ret_dict`
        KEY_LENGTH (str): key used to indicate a list representing lengths of output sequences in `ret_dict`
        KEY_SEQUENCE (str): key used to indicate a list of sequences in `ret_dict`

    Inputs: inputs, encoder_hidden, encoder_outputs, function, teacher_forcing_ratio
        - **inputs** (batch, seq_len, input_size): list of sequences, whose length is the batch size and within which
          each sequence is a list of token IDs.  It is used for teacher forcing when provided. (default `None`)
        - **encoder_hidden** (num_layers * num_directions, batch_size, hidden_size): tensor containing the features in the
          hidden state `h` of encoder. Used as the initial hidden state of the decoder. (default `None`)
        - **encoder_outputs** (batch, seq_len, hidden_size): tensor with containing the outputs of the encoder.
          Used for attention mechanism (default is `None`).
        - **function** (torch.nn.Module): A function used to generate symbols from RNN hidden state
          (default is `torch.nn.functional.log_softmax`).
        - **teacher_forcing_ratio** (float): The probability that teacher forcing will be used. A random number is
          drawn uniformly from 0-1 for every decoding token, and if the sample is smaller than the given value,
          teacher forcing would be used (default is 0).

    Outputs: decoder_outputs, decoder_hidden, ret_dict
        - **decoder_outputs** (seq_len, batch, vocab_size): list of tensors with size (batch_size, vocab_size) containing
          the outputs of the decoding function.
        - **decoder_hidden** (num_layers * num_directions, batch, hidden_size): tensor containing the last hidden
          state of the decoder.
        - **ret_dict**: dictionary containing additional information as follows {*KEY_LENGTH* : list of integers
          representing lengths of output sequences, *KEY_SEQUENCE* : list of sequences, where each sequence is a list of
          predicted token IDs }.
    """

    KEY_ATTN_SCORE = 'attention_score'
    KEY_LENGTH = 'length'
    KEY_SEQUENCE = 'sequence'

    def __init__(self, vocab_size, max_len, hidden_size,
            sos_id, eos_id,
            n_layers=1, rnn_cell='gru', bidirectional=False,
            input_dropout_p=0, dropout_p=0, use_attention=False, use_cuda=False):
        super(DecoderRNN, self).__init__(vocab_size, max_len, hidden_size,
                input_dropout_p, dropout_p,
                n_layers, rnn_cell)

        self.bidirectional_encoder = bidirectional
        self.rnn = self.rnn_cell(hidden_size, hidden_size, n_layers, batch_first=True, dropout=dropout_p)

        self.use_cuda = use_cuda
        self.output_size = vocab_size
        self.max_length = max_len
        self.use_attention = use_attention
        self.eos_id = eos_id
        self.sos_id = sos_id

        self.init_input = None

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        if use_attention:
            self.attention = Attention(self.hidden_size)

        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward_step(self, input_var, hidden, encoder_outputs, function):
        batch_size = input_var.size(0)
        output_size = input_var.size(1)
        embedded = self.embedding(input_var)
        embedded = self.input_dropout(embedded)

        # print('======== DEBUG ======== input_var', input_var.shape, '============== embedded\n',
        #  embedded.shape, '=========== hidden\n', [hidden[i].shape for i in range(len(hidden))])
        output, hidden = self.rnn(embedded, hidden)

        attn = None
        if self.use_attention:
            output, attn = self.attention(output, encoder_outputs)

        predicted_softmax = function(self.out(output.contiguous().view(-1, self.hidden_size)), dim=1).view(batch_size, output_size, -1)
        return predicted_softmax, hidden, attn

    def forward(self, inputs=None, encoder_hidden=None, encoder_outputs=None,
                    function=F.log_softmax, teacher_forcing_ratio=0):
        ret_dict = dict()
        if self.use_attention:
            ret_dict[DecoderRNN.KEY_ATTN_SCORE] = list()

        inputs, batch_size, max_length = self._validate_args(inputs, encoder_hidden, encoder_outputs,
                                                             function, teacher_forcing_ratio)
        decoder_hidden = self._init_state(encoder_hidden)

        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

        decoder_outputs = []
        sequence_symbols = []
        lengths = np.array([max_length] * batch_size)

        def decode(step, step_output, step_attn):
            decoder_outputs.append(step_output)
            if self.use_attention:
                ret_dict[DecoderRNN.KEY_ATTN_SCORE].append(step_attn)
            symbols = decoder_outputs[-1].topk(1)[1]
            sequence_symbols.append(symbols)

            eos_batches = symbols.data.eq(self.eos_id)
            if eos_batches.dim() > 0:
                eos_batches = eos_batches.cpu().view(-1).numpy()
                update_idx = ((lengths > step) & eos_batches) != 0
                lengths[update_idx] = len(sequence_symbols)
            return symbols

        # Manual unrolling is used to support random teacher forcing.
        # If teacher_forcing_ratio is True or False instead of a probability, the unrolling can be done in graph
        if use_teacher_forcing:
            decoder_input = inputs[:, :-1]
            decoder_output, decoder_hidden, attn = self.forward_step(decoder_input, decoder_hidden, encoder_outputs,
                                                                     function=function)

            for di in range(decoder_output.size(1)):
                step_output = decoder_output[:, di, :]
                if attn is not None:
                    step_attn = attn[:, di, :]
                else:
                    step_attn = None
                decode(di, step_output, step_attn)
        else:
            decoder_input = inputs[:, 0].unsqueeze(1)
            for di in range(max_length):
                decoder_output, decoder_hidden, step_attn = self.forward_step(decoder_input, decoder_hidden, encoder_outputs,
                                                                         function=function)
                step_output = decoder_output.squeeze(1)
                symbols = decode(di, step_output, step_attn)
                decoder_input = symbols

        ret_dict[DecoderRNN.KEY_SEQUENCE] = sequence_symbols
        ret_dict[DecoderRNN.KEY_LENGTH] = lengths.tolist()

        return decoder_outputs, decoder_hidden, ret_dict

    def _init_state(self, encoder_hidden):
        """ Initialize the encoder hidden state. """
        if encoder_hidden is None:
            return None
        if isinstance(encoder_hidden, tuple):
            encoder_hidden = tuple([self._cat_directions(h) for h in encoder_hidden])
        else:
            encoder_hidden = self._cat_directions(encoder_hidden)
        return encoder_hidden

    def _cat_directions(self, h):
        """ If the encoder is bidirectional, do the following transformation.
            (#directions * #layers, #batch, hidden_size) -> (#layers, #batch, #directions * hidden_size)
        """
        if self.bidirectional_encoder:
            h = torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2)
        return h

    def _validate_args(self, inputs, encoder_hidden, encoder_outputs, function, teacher_forcing_ratio):
        if self.use_attention:
            if encoder_outputs is None:
                raise ValueError("Argument encoder_outputs cannot be None when attention is used.")

        # inference batch size
        if inputs is None and encoder_hidden is None:
            batch_size = 1
        else:
            if inputs is not None:
                batch_size = inputs.size(0)
            else:
                if self.rnn_cell is nn.LSTM:
                    batch_size = encoder_hidden[0].size(1)
                elif self.rnn_cell is nn.GRU:
                    batch_size = encoder_hidden.size(1)

        # set default input and max decoding length
        if inputs is None:
            if teacher_forcing_ratio > 0:
                raise ValueError("Teacher forcing has to be disabled (set 0) when no inputs is provided.")
            inputs = torch.LongTensor([self.sos_id] * batch_size).view(batch_size, 1)
            if torch.cuda.is_available() and self.use_cuda:
                inputs = inputs.cuda()
            max_length = self.max_length
        else:
            max_length = inputs.size(1) - 1  # minus the start of sequence symbol

        return inputs, batch_size, max_length


class Seq2seq(nn.Module):
    """ Standard sequence-to-sequence architecture with configurable encoder
    and decoder.

    Args:
        encoder (EncoderRNN): object of EncoderRNN
        decoder (DecoderRNN): object of DecoderRNN
        decode_function (func, optional): function to generate symbols from output hidden states (default: F.log_softmax)

    Inputs: input_variable, input_lengths, target_variable, teacher_forcing_ratio
        - **input_variable** (list, option): list of sequences, whose length is the batch size and within which
          each sequence is a list of token IDs. This information is forwarded to the encoder.
        - **input_lengths** (list of int, optional): A list that contains the lengths of sequences
            in the mini-batch, it must be provided when using variable length RNN (default: `None`)
        - **target_variable** (list, optional): list of sequences, whose length is the batch size and within which
          each sequence is a list of token IDs. This information is forwarded to the decoder.
        - **teacher_forcing_ratio** (int, optional): The probability that teacher forcing will be used. A random number
          is drawn uniformly from 0-1 for every decoding token, and if the sample is smaller than the given value,
          teacher forcing would be used (default is 0)

    Outputs: decoder_outputs, decoder_hidden, ret_dict
        - **decoder_outputs** (batch): batch-length list of tensors with size (max_length, hidden_size) containing the
          outputs of the decoder.
        - **decoder_hidden** (num_layers * num_directions, batch, hidden_size): tensor containing the last hidden
          state of the decoder.
        - **ret_dict**: dictionary containing additional information as follows {*KEY_LENGTH* : list of integers
          representing lengths of output sequences, *KEY_SEQUENCE* : list of sequences, where each sequence is a list of
          predicted token IDs, *KEY_INPUT* : target outputs if provided for decoding, *KEY_ATTN_SCORE* : list of
          sequences, where each list is of attention weights }.

    """

    def __init__(self, encoder, decoder, decode_function=F.log_softmax, use_cuda=False):
        super(Seq2seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.decode_function = decode_function
        self.use_cuda = use_cuda

    def flatten_parameters(self):
        self.encoder.rnn.flatten_parameters()
        self.decoder.rnn.flatten_parameters()

    def forward(self, input_variable, input_lengths=None, target_variable=None,
                teacher_forcing_ratio=0):
        encoder_outputs, encoder_hidden = self.encoder(input_variable, input_lengths)
        result = self.decoder(inputs=target_variable,
                              encoder_hidden=encoder_hidden,
                              encoder_outputs=encoder_outputs,
                              function=self.decode_function,
                              teacher_forcing_ratio=teacher_forcing_ratio)
        return result


class Seq2SeqActionGenerator(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers, tgt_vocb_size, max_len, dropout_p,
                 sos_id, eos_id, token2id, id2token, opt,
                 bidirectional=False, use_attention=False, input_variable_lengths=False,
                 use_cuda=False
                 ):
        super(Seq2SeqActionGenerator, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.tgt_vocb_size = tgt_vocb_size
        self.max_len = max_len
        self.dropout_p = dropout_p
        self.sos_id, self.eos_id = sos_id, eos_id
        self.token2id = token2id
        self.id2token = id2token
        self.bidirectional = bidirectional
        self.use_attention = use_attention
        self.opt = opt
        self.use_cuda = use_cuda

        # self.encoder = nn.LSTM(
        #     input_size=input_size, hidden_size=hidden_size, num_layers=opt.depth,
        #     batch_first=True, dropout=dropout_p, bidirectional=bidirectional
        # )
        # self.encoder = EncoderRNN(vocab_size=input_size, max_len=max_len, hidden_size=hidden_size,
        #                      bidirectional=bidirectional, variable_lengths=input_variable_lengths)
        self.encoder = VanillaEncoder(
            input_size=input_size, max_len=max_len, hidden_size=hidden_size, input_dropout_p=dropout_p,
            dropout_p=dropout_p, n_layers=n_layers, bidirectional=bidirectional, rnn_cell=opt.encoder,
            variable_lengths=False
        )
        self.decoder = DecoderRNN(
            vocab_size=tgt_vocb_size, max_len=max_len, hidden_size=hidden_size * 2 if bidirectional else hidden_size,
            eos_id=eos_id, sos_id=sos_id, n_layers=n_layers, rnn_cell=opt.decoder,
            bidirectional=bidirectional, input_dropout_p=dropout_p, dropout_p=dropout_p, use_attention=use_attention,
            use_cuda=self.use_cuda,
        )
        self.seq2seq = Seq2seq(self.encoder, self.decoder)
        if use_cuda:
            self.seq2seq.cuda()

    def forward(self, batch_x, batch_y=None, teacher_forcing_ratio=0, output_token=False, pad_token='<PAD>'):
        input_var = batch_x if type(batch_x) == Variable else Variable(batch_x)
        input_var = input_var.float()
        if batch_y is not None:
            batch_y = batch_y if type(batch_y) == Variable else Variable(batch_y)

        decoder_outputs, decoder_hidden, other = self.seq2seq(input_variable=input_var, target_variable=batch_y,
                                                              teacher_forcing_ratio=teacher_forcing_ratio)
        # ''' prepare var for loss computing '''
        # target_var = []
        # for label in batch_y:
        #     tmp_var = []
        #     for id in label:
        #         token_v = [0 for i in range(len(self.token2id))]
        #         token_v[id] = 1
        #         tmp_var.append(token_v)
        #     target_var.append(tmp_var)
        # target_var = torch.LongTensor(target_var)

        ''' decode prediction vector as result '''
        pred_tokens = []
        preds = []
        for ind in range(len(batch_x)):
            length = other['length'][ind]
            tgt_id_seq = [other['sequence'][di][ind].data[0] for di in range(length)]
            tgt_seq = [self.id2token[int(tok)] for tok in tgt_id_seq]
            preds.append(tgt_id_seq)
            pred_tokens.append(tgt_seq)
        if self.training:
            '''computing loss '''
            # loss = Perplexity()
            weight = torch.FloatTensor([1 for i in range(len(self.token2id))])
            if self.use_cuda:
                weight = weight.cuda()
            loss = NLLLoss(weight=weight, mask=self.token2id[pad_token], size_average=True)

            # loss = nn.NLLLoss(weight=weight, size_average=size_average)

            for step, step_output in enumerate(decoder_outputs):
                batch_size = batch_x.size(0)
                ''' Select out current step's token. EOS is not included in step_output, so + 1 step for target '''
                pred_token_distribute_batch = step_output.contiguous().view(batch_size, -1)
                target_token_id_v_batch = batch_y[:, step + 1]
                loss.eval_batch(pred_token_distribute_batch, target_token_id_v_batch)
            if output_token:
                return preds, loss.acc_loss, pred_tokens
            else:
                return preds, loss.acc_loss
        else:
            if output_token:
                return preds, pred_tokens
            else:
                return preds


class StateEncoder(nn.Module):
    def __init__(self, slot_num, diaact_num, embedded_v_size, state_v_component, max_len, hidden_size,
                 vocab_size=False, input_dropout_p=0, dropout_p=0,
                 n_layers=1, bidirectional=False, rnn_cell='lstm', variable_lengths=False, use_cuda=False):
        super(StateEncoder, self).__init__()
        self.variable_lengths = variable_lengths
        self.state_v_component = state_v_component
        self.bidirectional = bidirectional
        self.n_layers = n_layers
        self.n_directions = 2 if self.bidirectional else 1
        self.embedded_v_size = hidden_size  # should be force to same with hidden size, as it imitates rnn output
        self.use_cuda = use_cuda

        self.diaact_embedding = nn.Linear(diaact_num, self.embedded_v_size)
        # In my neural model, I actually use the union set of all possible slots
        self.slot_embedding = nn.Linear(slot_num, self.embedded_v_size)
        # Status has 1 dim only, -1, 0, 1 for failed, no outcome, success,
        self.dialog_status_embedding = nn.Linear(1, self.embedded_v_size)

        # h_n: (num_layers * num_directions, batch, hidden_size)
        # c_n: (num_layers * num_directions, batch, hidden_size)
        self.h_encoder = nn.Linear(
            self.embedded_v_size * len(self.state_v_component),
            hidden_size * n_layers * 2 if self.bidirectional else hidden_size * n_layers
        )
        self.c_encoder = nn.Linear(
            self.embedded_v_size * len(self.state_v_component),
            hidden_size * n_layers * 2 if self.bidirectional else hidden_size * n_layers
        )
        self.h_relu = nn.ReLU()
        self.c_relu = nn.ReLU()

    def forward(self, batch_x, input_lengths=None):
        batch_size = len(batch_x)

        def get_batch_component(batch, component_idx):
            '''
            list could not proceed multi-dim slice, so write this
            :param batch: iterable batch
            :param component_idx:
            :return: torch variable
            '''
            tmp = []
            for sample in batch:
                tmp.append(sample[component_idx])
            tmp = torch.LongTensor(tmp).cuda() if self.use_cuda else torch.LongTensor(tmp)
            return Variable(tmp).float()

        ''' Extract State Representation '''
        if len(self.state_v_component) != 9:
            raise RuntimeError('Wrong Component! The state v component is hard coded as here and vector2state()')
        goal_inform_slots_v = get_batch_component(batch_x, 0)
        goal_request_slots_v = get_batch_component(batch_x, 1)
        history_slots_v = get_batch_component(batch_x, 2)  # informed slots, 1 informed, 0 irrelevant, -1 not informed,
        rest_slots_v = get_batch_component(batch_x, 3)  # remained request slots, 1 for remained, 0 irrelevant, -1 for already got,
        system_diaact_v = get_batch_component(batch_x, 4)  # system diaact,
        system_inform_slots_v = get_batch_component(batch_x, 5)  # inform slots of sys response,
        system_request_slots_v = get_batch_component(batch_x, 6)  # request slots of sys response,
        consistency_v = get_batch_component(batch_x, 7)  # for each pos, -1 inconsistent, 0 irrelevent or not requested, 1 consistent,
        dialog_status_v = get_batch_component(batch_x, 8)  # -1, 0, 1 for failed, no outcome, success,

        ''' Get embeddings of state component '''
        goal_inform_slots_embedding = self.slot_embedding(goal_inform_slots_v)
        goal_request_slots_embedding = self.slot_embedding(goal_request_slots_v)
        history_slots_embedding = self.slot_embedding(history_slots_v)
        rest_slots_embedding = self.slot_embedding(rest_slots_v)
        system_diaact_embedding = self.diaact_embedding(system_diaact_v)
        system_inform_slots_embedding = self.slot_embedding(system_inform_slots_v)
        system_request_slots_embedding = self.slot_embedding(system_request_slots_v)
        consistency_embedding = self.slot_embedding(consistency_v)
        dialog_status_embedding = self.dialog_status_embedding(dialog_status_v)

        ''' Send embeddings to encoder '''
        all_embeddings = [
            goal_inform_slots_embedding,
            goal_request_slots_embedding,
            history_slots_embedding,
            rest_slots_embedding,
            system_diaact_embedding,
            system_inform_slots_embedding,
            system_request_slots_embedding,
            consistency_embedding,
            dialog_status_embedding
        ]
        concated_state = torch.cat(
            all_embeddings,
            1
        )
        h_n = self.h_relu(self.h_encoder(concated_state))
        c_n = self.c_relu(self.c_encoder(concated_state))

        ''' Split hidden state to fit LSTM output shape'''
        # h_n: (num_layers * num_directions, batch, hidden_size)
        # c_n: (num_layers * num_directions, batch, hidden_size)
        h_n = h_n.view(batch_size, self.n_layers * self.n_directions, -1)
        c_n = c_n.view(batch_size, self.n_layers * self.n_directions, -1)
        # convert to batch second
        h_n = torch.transpose(h_n, 0, 1)
        c_n = torch.transpose(c_n, 0, 1)

        hidden = (h_n.contiguous(), c_n.contiguous())

        ''' Build output to imitate rnn'''
        # outputs: (batch, seq_len, hidden_size * num_directions)
        for ei in all_embeddings:
            ei.unsqueeze_(0)  # add one extra dimension
        output = torch.cat(all_embeddings, 0)
        output = torch.transpose(output, 0, 1).contiguous()  # convert to batch first to fit decoder's need
        # print(output.size(), h_n.size(), c_n.size())
        return output, hidden


class State2Seq(nn.Module):
    def __init__(self, slot_num, diaact_num, embedded_v_size, state_v_component,
                 hidden_size, n_layers, tgt_vocb_size, max_len, dropout_p,
                 sos_id, eos_id, token2id, id2token, opt,
                 bidirectional=False, use_attention=False, input_variable_lengths=False,
                 use_cuda=False
                 ):
        super(State2Seq, self).__init__()
        self.input_size = slot_num * 6 + diaact_num * 2 + 1
        self.slot_num = slot_num
        self.diaact_num = diaact_num
        self.embedded_v_size = embedded_v_size
        self.state_v_component = state_v_component
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.tgt_vocb_size = tgt_vocb_size
        self.max_len = max_len
        self.dropout_p = dropout_p
        self.sos_id, self.eos_id = sos_id, eos_id
        self.token2id = token2id
        self.id2token = id2token
        self.bidirectional = bidirectional  # bidirectional should forced to be false here
        self.use_attention = use_attention
        self.opt = opt
        self.use_cuda = use_cuda

        self.encoder = StateEncoder(
            slot_num=slot_num, diaact_num=diaact_num,embedded_v_size=embedded_v_size, state_v_component=state_v_component,
            max_len=max_len, hidden_size=hidden_size, input_dropout_p=dropout_p,
            dropout_p=dropout_p, n_layers=n_layers, bidirectional=bidirectional, rnn_cell=opt.encoder,
            variable_lengths=False, use_cuda=use_cuda,
        )
        self.decoder = DecoderRNN(
            vocab_size=tgt_vocb_size, max_len=max_len, hidden_size=hidden_size * 2 if bidirectional else hidden_size,
            eos_id=eos_id, sos_id=sos_id, n_layers=n_layers, rnn_cell=opt.decoder,
            bidirectional=bidirectional, input_dropout_p=dropout_p, dropout_p=dropout_p, use_attention=use_attention,
            use_cuda=self.use_cuda,
        )
        self.seq2seq = Seq2seq(self.encoder, self.decoder)
        if use_cuda:
            self.seq2seq.cuda()

    def forward(self, batch_x, batch_y=None, teacher_forcing_ratio=0, output_token=False, pad_token='<PAD>'):
        # input_var = batch_x if type(batch_x) == Variable else Variable(batch_x)
        # input_var = input_var.float()
        if batch_y is not None:
            if type(batch_y) == list:
                batch_y = torch.LongTensor(batch_y).cuda() if self.use_cuda else torch.LongTensor(batch_y)
            batch_y = batch_y if type(batch_y) == Variable else Variable(batch_y)

        ''' input will be changed to variable by state_encoder's forward function '''
        decoder_outputs, decoder_hidden, other = self.seq2seq(input_variable=batch_x, target_variable=batch_y,
                                                              teacher_forcing_ratio=teacher_forcing_ratio)

        ''' decode prediction vector as result '''
        pred_tokens = []
        preds = []
        for ind in range(len(batch_x)):
            length = other['length'][ind]
            tgt_id_seq = [other['sequence'][di][ind].data[0] for di in range(length)]
            tgt_seq = [self.id2token[int(tok)] for tok in tgt_id_seq]
            preds.append(tgt_id_seq)
            pred_tokens.append(tgt_seq)
        if self.training:
            '''computing loss '''
            # loss = Perplexity()
            weight = torch.FloatTensor([1 for i in range(len(self.token2id))])
            if self.use_cuda:
                weight = weight.cuda()
            loss = NLLLoss(weight=weight, mask=self.token2id[pad_token], size_average=True)

            # loss = nn.NLLLoss(weight=weight, size_average=size_average)

            for step, step_output in enumerate(decoder_outputs):
                try:
                    batch_size = batch_x.size(0)
                except AttributeError:
                    batch_size = len(batch_x)
                ''' Select out current step's token. EOS is not included in step_output, so + 1 step for target '''
                pred_token_distribute_batch = step_output.contiguous().view(batch_size, -1)
                target_token_id_v_batch = batch_y[:, step + 1]
                loss.eval_batch(pred_token_distribute_batch, target_token_id_v_batch)
            if output_token:
                return preds, loss.acc_loss, pred_tokens
            else:
                return preds, loss.acc_loss
        else:
            if output_token:
                return preds, pred_tokens
            else:
                return preds


def test_MultiLableClassifyLayer():
    """
    Unit test for multilabel ClassifyLayer
    :return:
    """
    # Make fake dataset
    # (1, 0) => target labels 1 0 1
    # (0, 1) => target labels 0 1 0
    # (1, 1) => target labels 0 0 1
    train = []
    labels = []
    for i in range(10000):
        category = (np.random.choice([0, 1]), np.random.choice([0, 1]))
        if category == (1, 0):
            train.append([np.random.uniform(0.1, 1), 0])
            labels.append([1, 0, 1])
        if category == (0, 1):
            train.append([0, np.random.uniform(0.1, 1)])
            labels.append([0, 1, 0])
        if category == (0, 0):
            train.append([np.random.uniform(0.1, 1), np.random.uniform(0.1, 1)])
            labels.append([0, 0, 1])

    # Training process
    nlabel = len(labels[0])  # => 3
    input_size = len(train[0])
    classifier = MultiLableClassifyLayer(input_size=input_size, hidden_size=64, num_tags=nlabel)

    optimizer = optim.Adam(classifier.parameters())

    # Training
    classifier.train()
    epochs = 5
    for epoch in range(epochs):
        losses = []
        for i, sample in enumerate(train):
            inputv = Variable(torch.FloatTensor(sample)).view(1, -1)
            labelsv = Variable(torch.FloatTensor(labels[i])).view(1, -1)

            output, loss = classifier(inputv)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.data.mean())
        print('[%d/%d] Loss: %.3f' % (epoch + 1, epochs, np.mean(losses)))
    # Testing
    print('Start Testing')
    classifier.eval()
    input_sample = [float(a) for a in raw_input().split()]
    while input_sample:
        input_sample = [float(a) for a in raw_input().split()]
        input_sample_v = Variable(torch.FloatTensor(input_sample)).view(1, -1)
        output = classifier(input_sample_v)
        print('pred resultis', output)


if __name__ == "__main__":
    test_MultiLableClassifyLayer()

