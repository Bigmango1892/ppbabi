#!/usr/bin/env python
# coding:utf8

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import time
import yutils
import numpy


torch.manual_seed(123456)


class BLSTM(nn.Module):
    """
        Implementation of BLSTM Concatenation for sentiment classification task
    """

    def __init__(self, embeddings, input_dim, hidden_dim, num_layers, output_dim, max_len=40, dropout=0.5):
        super(BLSTM, self).__init__()

        self.emb = nn.Embedding(num_embeddings=embeddings.size(0),
                                embedding_dim=embeddings.size(1),
                                padding_idx=0)
        self.emb.weight = nn.Parameter(embeddings)

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # sen encoder
        self.sen_len = max_len
        self.sen_rnn = nn.LSTM(input_size=input_dim,
                               hidden_size=hidden_dim,
                               num_layers=num_layers,
                               dropout=dropout,
                               batch_first=True,
                               bidirectional=True)

        self.output = nn.Linear(2 * self.hidden_dim, output_dim)

    def bi_fetch(self, rnn_outs, seq_lengths, batch_size, max_len):
        rnn_outs = rnn_outs.view(batch_size, max_len, 2, -1)

        # (batch_size, max_len, 1, -1)
        fw_out = torch.index_select(rnn_outs, 2, Variable(torch.LongTensor([0])).cuda())
        fw_out = fw_out.view(batch_size * max_len, -1)
        bw_out = torch.index_select(rnn_outs, 2, Variable(torch.LongTensor([1])).cuda())
        bw_out = bw_out.view(batch_size * max_len, -1)

        batch_range = Variable(torch.LongTensor(range(batch_size))).cuda() * max_len
        batch_zeros = Variable(torch.zeros(batch_size).long()).cuda()

        fw_index = batch_range + seq_lengths.view(batch_size) - 1
        fw_out = torch.index_select(fw_out, 0, fw_index)  # (batch_size, hid)

        bw_index = batch_range + batch_zeros
        bw_out = torch.index_select(bw_out, 0, bw_index)

        outs = torch.cat([fw_out, bw_out], dim=1)
        return outs

    def forward(self, sen_batch, sen_lengths, sen_mask_matrix):
        """
        :param sen_batch: (batch, sen_length), tensor for sentence sequence
        :param sen_lengths:
        :param sen_mask_matrix:
        :return:
        """

        ''' Embedding Layer | Padding | Sequence_length 40'''
        sen_batch = self.emb(sen_batch)

        batch_size = len(sen_batch)

        ''' Bi-LSTM Computation '''
        sen_outs, _ = self.sen_rnn(sen_batch.view(batch_size, -1, self.input_dim))
        sen_rnn = sen_outs.contiguous().view(batch_size, -1, 2 * self.hidden_dim)  # (batch, sen_len, 2*hid)

        ''' Fetch the truly last hidden layer of both sides
        '''
        sentence_batch = self.bi_fetch(sen_rnn, sen_lengths, batch_size, self.sen_len)  # (batch_size, 2*hid)

        representation = sentence_batch
        out = self.output(representation)
        out_prob = F.softmax(out.view(batch_size, -1))

        return out_prob


def train(model, training_data, args, optimizer, criterion):
    model.train()

    batch_size = args.batch_size

    sentences, sentences_seqlen, sentences_mask, labels = training_data

    # print batch_size, len(sentences), len(labels)

    assert batch_size == len(sentences) == len(labels)

    ''' Prepare data and prediction'''
    sentences_, sentences_seqlen_, sentences_mask_ = \
        var_batch(args, batch_size, sentences, sentences_seqlen, sentences_mask)
    labels_ = Variable(torch.LongTensor(labels))
    if args.cuda:
        labels_ = labels_.cuda()

    assert len(sentences) == len(labels)

    model.zero_grad()
    probs = model(sentences_, sentences_seqlen_, sentences_mask_)
    loss = criterion(probs.view(len(labels_), -1), labels_)

    loss.backward()
    optimizer.step()


def test(model, dataset, args, data_part="test"):
    """
    :param model:
    :param args:
    :param dataset:
    :param data_part:
    :return:
    """

    tvt_set = dataset[data_part]
    tvt_set = yutils.YDataset(tvt_set["xIndexes"],
                              tvt_set["yLabels"],
                              to_pad=True, max_len=args.sen_max_len)

    test_set = tvt_set
    sentences, sentences_seqlen, sentences_mask, labels = test_set.next_batch(len(test_set))

    assert len(test_set) == len(sentences) == len(labels)

    tic = time.time()

    model.eval()
    ''' Prepare data and prediction'''
    batch_size = len(sentences)
    sentences_, sentences_seqlen_, sentences_mask_ = \
        var_batch(args, batch_size, sentences, sentences_seqlen, sentences_mask)

    probs = model(sentences_, sentences_seqlen_, sentences_mask_)

    _, pred = torch.max(probs, dim=1)

    if args.cuda:
        pred = pred.view(-1).cpu().data.numpy()
    else:
        pred = pred.view(-1).data.numpy()

    tit = time.time() - tic
    print
    "  Predicting {:d} examples using {:5.4f} seconds".format(len(test_set), tit)

    labels = numpy.asarray(labels)
    ''' log and return prf scores '''
    accuracy = test_prf(pred, labels)

    return accuracy


def cal_prf(pred, right, gold, formation=True, metric_type=""):
    """
    :param pred: predicted labels
    :param right: predicting right labels
    :param gold: gold labels
    :param formation: whether format the float to 6 digits
    :param metric_type:
    :return: prf for each label
    """
    ''' Pred: [0, 2905, 0]  Right: [0, 2083, 0]  Gold: [370, 2083, 452] '''
    num_class = len(pred)
    precision = [0.0] * num_class
    recall = [0.0] * num_class
    f1_score = [0.0] * num_class

    for i in xrange(num_class):
        ''' cal precision for each class: right / predict '''
        precision[i] = 0 if pred[i] == 0 else 1.0 * right[i] / pred[i]

        ''' cal recall for each class: right / gold '''
        recall[i] = 0 if gold[i] == 0 else 1.0 * right[i] / gold[i]

        ''' cal recall for each class: 2 pr / (p+r) '''
        f1_score[i] = 0 if precision[i] == 0 or recall[i] == 0 \
            else 2.0 * (precision[i] * recall[i]) / (precision[i] + recall[i])

        if formation:
            precision[i] = precision[i].__format__(".6f")
            recall[i] = recall[i].__format__(".6f")
            f1_score[i] = f1_score[i].__format__(".6f")

    ''' PRF for each label or PRF for all labels '''
    if metric_type == "macro":
        precision = sum(precision) / len(precision)
        recall = sum(recall) / len(recall)
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    elif metric_type == "micro":
        precision = 1.0 * sum(right) / sum(pred) if sum(pred) > 0 else 0
        recall = 1.0 * sum(right) / sum(gold) if sum(recall) > 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1_score
