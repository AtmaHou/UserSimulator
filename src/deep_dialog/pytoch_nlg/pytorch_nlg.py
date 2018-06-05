# coding: utf-8
import logging
import argparse
import random

import torch
import torchtext

from torch.optim.lr_scheduler import StepLR

import seq2seq
from seq2seq.trainer import SupervisedTrainer
from seq2seq.models import EncoderRNN, DecoderRNN, Seq2seq
from seq2seq.loss import Perplexity
from seq2seq.optim import Optimizer
from seq2seq.dataset import SourceField, TargetField
from seq2seq.evaluator import Predictor
from seq2seq.util.checkpoint import Checkpoint
from nltk.tokenize.treebank import TreebankWordTokenizer


DATA_DIR = '/users4/ythou/Projects/TaskOrientedDialogue/code/TC-Bot/src/deep_dialog/data/'


def treebank_tokenizer(sentence, max_length=0):
    """
    Tokenize and truncate sentence
    :param sentence: str, a sentence string
    :param max_length: int, max token included in the result, 0 for unlimited
    :return: list, a list of token
    """
    # split 's but also split <>, wait to use in further work
    t = TreebankWordTokenizer()
    word_lst = t.tokenize(sentence.lower().replace("$", "_B_"))
    # word_lst = t.tokenize(sentence.lower().replace("<", "LAB_").replace(">", "_RAB"))
    ret = []
    for w in word_lst:
         ret.append(w.replace("_B_", "$"))
         # ret.append(w.replace("LAB_", "<").replace("_RAB", ">"))
    if max_length > 0:
        return ret[: max_length]
    else:
        return ret


def data_loader(target_file_path):
    pass


def offline_training(opt, traget_file_path):

    # Prepare dataset with torchtext
    src = SourceField(tokenize=treebank_tokenizer)
    tgt = TargetField(tokenize=treebank_tokenizer)

    def sample_filter(sample):
        """ sample example for future purpose"""
        return True

    train = torchtext.data.TabularDataset(
        path=opt.train_path, format='tsv',
        fields=[('src', src), ('tgt', tgt)],
        filter_pred=sample_filter
    )
    dev = torchtext.data.TabularDataset(
        path=opt.dev_path, format='tsv',
        fields=[('src', src), ('tgt', tgt)],
        filter_pred=sample_filter
    )
    test = torchtext.data.TabularDataset(
        path=opt.dev_path, format='tsv',
        fields=[('src', src), ('tgt', tgt)],
        filter_pred=sample_filter
    )
    src.build_vocab(train, max_size=opt.src_vocab_size)
    tgt.build_vocab(train, max_size=opt.tgt_vocab_size)
    input_vocab = src.vocab
    output_vocab = tgt.vocab

    # NOTE: If the source field name and the target field name
    # are different from 'src' and 'tgt' respectively, they have
    # to be set explicitly before any training or inference
    # seq2seq.src_field_name = 'src'
    # seq2seq.tgt_field_name = 'tgt'

    # Prepare loss
    weight = torch.ones(len(tgt.vocab))
    pad = tgt.vocab.stoi[tgt.pad_token]
    if opt.loss == 'perplexity':
        loss = Perplexity(weight, pad)
    else:
        raise TypeError

    seq2seq = None
    optimizer = None
    if not opt.resume:
        # Initialize model
        encoder = EncoderRNN(
            vocab_size=len(src.vocab),
            max_len=opt.max_length,
            hidden_size=opt.hidden_size,
            input_dropout_p=opt.intput_dropout_p,
            dropout_p=opt.dropout_p,
            n_layers=opt.n_layers,
            bidirectional=opt.bidirectional,
            rnn_cell=opt.rnn_cell,
            variable_lengths=True,
            embedding=input_vocab.vectors if opt.use_pre_trained_embedding else None,
            update_embedding=opt.update_embedding
        )
        decoder = DecoderRNN(
            vocab_size=len(tgt.vocab),
            max_len=opt.max_length,
            hidden_size=opt.hidden_size * 2 if opt.bidirectional else opt.hidden_size,
            sos_id=tgt.sos_id,
            eos_id=tgt.eos_id,
            n_layers=opt.n_layers,
            rnn_cell=opt.rnn_cell,
            bidirectional=opt.bidirectional,
            input_dropout_p=opt.input_dropout_p,
            dropout_p=opt.dropout_p,
            use_attention=opt.use_attention
        )
        seq2seq = Seq2seq(encoder=encoder, decoder=decoder)
        if opt.gpu >= 0 and torch.cuda.is_available():
            seq2seq.cuda()

        for param in seq2seq.parameters():
            param.data.uniform_(-0.08, 0.08)
    # train
    trainer = SupervisedTrainer(
        loss=loss,
        batch_size=opt.batch_size,
        checkpoint_every=opt.checkpoint_every,
        print_every=opt.print_every,
        expt_dir=opt.expt_dir
    )
    seq2seq = trainer.train(
        model=seq2seq,
        data=train,
        num_epochs=opt.epochs,
        resume=opt.resume,
        dev_data=dev,
        optimizer=optimizer,
        teacher_forcing_ratio=opt.teacher_forcing_rate
    )




def online_training():
    pass


def test(opt, test_path):
    if opt.load_checkpoint is not None:
        # load model
        logging.info(
            "loading check point from {}".format(
                os.path.join(opt.expt_dir, Checkpoint.CHECKPOINT_DIR_NAME, opt.load_checkpoint)
            )
        )
        checkpoint_path = os.path.join(opt.expt_dir, Checkpoint.CHECKPOINT_DIR_NAME, opt.load_checkpoint)
        checkpoint = Checkpoint.load(checkpoint_path)
        seq2seq = checkpoint.model
        input_vocab = checkpoint.input_vocab
        output_vocab = checkpoint.output_vocab

        # Prepare predictor
        predictor = Predictor(seq2seq, input_vocab, output_vocab)

        with open(test_path, 'r') as reader, open(test_path + '_pred', 'w') as writer:
            for line in reader:
                source = treebank_tokenizer(line.split("\t")[0])
                writer.write(generate(source, predictor) + '\n')


def generate(input_seq, predictor):
    return predictor.predict(input_seq)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # file path
    parser.add_argument("--train_path", type=str, default=DATA_DIR + "nlg/train.txt", help="file path for train path")
    parser.add_argument("--dev_path", type=str, default=DATA_DIR + "nlg/dev.txt", help="file path for dev path")
    parser.add_argument("--test_path", type=str, default=DATA_DIR + "nlg/test.txt", help="file path for test path")
    parser.add_argument('--expt_dir', action='store', dest='expt_dir', default='./experiment',
                        help='Path to experiment directory. If load_checkpoint, then this directory be provided')
    parser.add_argument('--raw_path', type=str, default=DATA_DIR + "dia_act_nl_pairs.v6.json",
                        help="file path for raw data file")
    parser.add_argument('--pre_trained_embedding_path', type=str, help='path to a pre_trained embedding file')


    # train style setting
    parser.add_argument('--load_checkpoint', action='store', dest='load_checkpoint',
                        help='The name of the checkpoint to load, usually an encoded time string')
    parser.add_argument('--source_format', type=str, default='seq_action', help='select data\'s source end format',
                        choices=['seq_action'])
    parser.add_argument('--checkpoint_every', type=int, default=50, help='number of batches to checkpoint after')
    parser.add_argument('--print_every', type=int, default=10, help='number of batches to print after')
    parser.add_argument('--resume', action='store_true', help='use the model loaded from the latest checkpoint')


    # tuning setting
    parser.add_argument('--random_seed', type=int, default=0, help='set random seed for experiment')
    parser.add_argument('--max_length', type=int, default=50, help='max sentence length for encoder & decoder')
    parser.add_argument('--tgt_vocab_size', type=int, default=50000,
                        help='max source vocab size for encoder(the actual size will be no more than this)')
    parser.add_argument('--src_vocab_size', type=int, default=50000,
                        help='max target vocab size for decoder(the actual size will be no more than this)')
    parser.add_argument('--hidden_size', type=int, default=128, help='hidden size for RNN')
    parser.add_argument('--loss', type=str, default='perplexity', choices=['perplexity'], help='set loss type')
    parser.add_argument('--train_epoch', type=int, default=15, help='set training epoch')
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam'], help='set optimizer(do nothing now)')
    parser.add_argument('--bidirectional', action='store_true', help='set rnn encoder type of direction')
    parser.add_argument('--input_dropout_p', type=float, default=0.0, help='dropout probability for the input sequence')
    parser.add_argument('--dropout_p', type=float, default=0.2, help='dropout probability for the rnn cell')
    parser.add_argument('--n_layers', type=int, default=2, help='the layer num of rnn cell')
    parser.add_argument('--rnn_cell', type=str, default='lstm', choices=['gru', 'lstm'], help='set rnn type')
    parser.add_argument('--use_pre_trained_embedding', action=True, help='use pre-trained embedding to init encoder')
    parser.add_argument('--update_embedding', action='store_true', help='to update embedding during training')
    parser.add_argument('--use_attention', action='store_true', help='use attention during decoding')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size for training')
    parser.add_argument('--teacher_forcing_rate', type=float, default=0.5, help='set rate for teacher forcing')

    # gpu setting
    parser.add_argument('--gpu_id', type=int, default=-1, help='set gpu id, -1 for not use gpu')

    opt = parser.parse_args()

    LOG_FORMAT = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
    logging.basicConfig(format=LOG_FORMAT, level=getattr(logging, opt.log_level.upper()))
    logging.info(opt)

    # set random seed
    random.seed(opt.random_seed)
    torch.manual_seed(opt.random_seed)

    # set gpu
    if opt.gpu >= 0 and torch.cuda.is_available():
        torch.cuda.set_device(opt.gpu)
