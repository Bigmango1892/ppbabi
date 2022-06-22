import random
import numpy as np
import input_layer
import torch
import torch.nn as nn
import torch.utils.data as Data
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

# 设置
torch.manual_seed(1)

# 添加文本起止符作为文本列表中的哨兵元
START_TAG = '<START>'
STOP_TAG = '<STOP>'
# 嵌入特征向量维度D（文中设置为100）
EMBEDDING_DIM = 100
# 隐藏层维度（文中设置为100）
HIDDEN_DIM = 100
# 批大小（文中设置为20）
BATCH_SIZE = 20
# 学习率（文中设置为0.001)
LEARNING_RATE = 0.1
# epoch（文中设置为）
EPOCH = 100


def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    if idx.shape == torch.Size([1]):
        return idx.item()
    else:
        return idx


def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)


# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec):
    max_score = vec[range(vec.size()[0]), argmax(vec)]
    max_score_broadcast = max_score.view(-1, 1).expand(vec.size()[0], vec.size()[1])
    return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast), 1))


def batch_length_pad(sentences):
    pad_datas = pad_sequence(sentences, batch_first=True, padding_value=0)
    # 这个应该是batch size first
    # print(pad_datas.shape)
    return pad_datas


# TODO
def batch_pack_padded(sentences, sentences_padded):
    batch_len = torch.tensor([len(t) for t in sentences])
    pack_sample = pack_padded_sequence(sentences_padded, batch_len, enforce_sorted=False, batch_first=True)
    return pack_sample


class BiLSTM_CRF_no_batch(nn.Module):
    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim, batch_size=BATCH_SIZE):
        super(BiLSTM_CRF_no_batch, self).__init__()
        # 设置模型参数
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)

        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2, num_layers=1, bidirectional=True)

        # Maps the output of the LSTM into tag space.
        self.hidden2tag = nn.Linear(hidden_dim + 3, self.tagset_size)

        # Matrix of transition parameters.  Entry i,j is the score of
        # transitioning *to* i *from* j.
        self.transitions = nn.Parameter(torch.randn(self.tagset_size, self.tagset_size))

        # These two statements enforce the constraint that we never transfer
        # to the start tag, and we never transfer from the stop tag
        self.transitions.data[tag_to_ix[START_TAG], :] = -10000
        self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000

        self.hidden = self.init_hidden()

    def init_hidden(self):
        return torch.randn(2, 1, self.hidden_dim // 2), torch.randn(2, 1, self.hidden_dim // 2)

    def _get_lstm_features(self, sentence_onehot):
        # Embedding layer -- input: sentence code; output: embedded matrix
        # view方法重新整合tensor维度，前两参数为维度，-1表示自动匹配
        print(self.word_embeds(sentence_onehot).shape)
        embeds = self.word_embeds(sentence_onehot).view(len(sentence_onehot), 1, -1)
        # LSTM layer -- input: eigenvalue, hidden layers matrix; output: LSTM out, new hidden layers matrix
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        lstm_out = lstm_out.view(len(sentence_onehot), self.hidden_dim)
        return lstm_out, embeds.view(len(sentence_onehot), self.hidden_dim)

    def _feat_splice(self, sentence, embeds):
        eig = torch.cat((embeds, sentence.pos.view(-1, 1), sentence.con.view(-1, 1), sentence.seg.view(-1, 1)), 1)
        split_feat = self.hidden2tag(eig)
        return split_feat

    def _forward_alg(self, feats):
        # TODO 查找CRF实现
        # Do the forward algorithm to compute the partition function
        init_alphas = torch.full((1, self.tagset_size), -10000.)
        # START_TAG has all the score.
        init_alphas[0][self.tag_to_ix[START_TAG]] = 0.

        # Wrap in a variable so that we will get automatic backprop
        forward_var = init_alphas

        # Iterate through the sentence
        for feat in feats:
            alphas_t = []  # The forward tensors at this timestep
            for next_tag in range(self.tagset_size):
                # broadcast the emission score: it is the same regardless of the previous tag
                emit_score = feat[next_tag].view(1, -1).expand(1, self.tagset_size)
                # the i-th entry of trans_score is the score of transitioning to next_tag from i
                trans_score = self.transitions[next_tag].view(1, -1)
                # The i-th entry of next_tag_var is the value for the edge (i -> next_tag) before we do log-sum-exp
                next_tag_var = forward_var + trans_score + emit_score
                # The forward variable for this tag is log-sum-exp of all the scores.
                alphas_t.append(log_sum_exp(next_tag_var).view(1))
            forward_var = torch.cat(alphas_t).view(1, -1)
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        alpha = log_sum_exp(terminal_var)
        return alpha

    def _score_sentence(self, feats, tags):
        # Gives the score of a provided tag sequence
        score = torch.zeros(1)
        tags = torch.cat([torch.tensor([self.tag_to_ix[START_TAG]], dtype=torch.long), tags])
        for i, feat in enumerate(feats):
            score = score + self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self.transitions[self.tag_to_ix[STOP_TAG], tags[-1]]
        return score

    def loss_function(self, sentence, tags):
        lstm_feats, embeds = self._get_lstm_features(sentence.onehot)
        splice_feats = self._feat_splice(sentence, embeds)
        forward_score = self._forward_alg(splice_feats)
        gold_score = self._score_sentence(splice_feats, tags)
        return forward_score - gold_score

    # TODO 添加其他特征值进入模型

    def _viterbi_decode(self, feats):
        backpointers = []

        # Initialize the viterbi variables in log space
        init_vvars = torch.full((1, self.tagset_size), -10000.)
        init_vvars[0][self.tag_to_ix[START_TAG]] = 0

        # forward_var at step i holds the viterbi variables for step i-1
        forward_var = init_vvars
        for feat in feats:
            bptrs_t = []  # holds the backpointers for this step
            viterbivars_t = []  # holds the viterbi variables for this step

            for next_tag in range(self.tagset_size):
                # next_tag_var[i] holds the viterbi variable for tag i at the
                # previous step, plus the score of transitioning
                # from tag i to next_tag.
                # We don't include the emission scores here because the max
                # does not depend on them (we add them in below)
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
            # Now add in the emission scores, and assign forward_var to the set
            # of viterbi variables we just computed
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)

        # Transition to STOP_TAG
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        # Follow the back pointers to decode the best path.
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        # Pop off the start tag (we dont want to return that to the caller)
        start = best_path.pop()
        assert start == self.tag_to_ix[START_TAG]  # Sanity check
        best_path.reverse()
        return path_score, best_path

    def forward(self, sentence):  # don't confuse this with _forward_alg above.
        lstm_feats, embeds = self._get_lstm_features(sentence.onehot)
        splice_feats = self._feat_splice(sentence, embeds)
        score, tag_seq = self._viterbi_decode(splice_feats)
        return tag_seq


class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim, batch_size=BATCH_SIZE):
        super(BiLSTM_CRF, self).__init__()
        # 设置模型参数
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)
        self.batch_size = batch_size

        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        # 这里batch_first要加吗？
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2, num_layers=1, bidirectional=True)

        # Maps the output of the LSTM into tag space.
        self.hidden2tag = nn.Linear(2 * hidden_dim + 3, self.tagset_size)

        # Matrix of transition parameters.  Entry i,j is the score of
        # transitioning *to* i *from* j.
        self.transitions = nn.Parameter(torch.randn(self.tagset_size, self.tagset_size))

        # These two statements enforce the constraint that we never transfer
        # to the start tag, and we never transfer from the stop tag
        self.transitions.data[tag_to_ix[START_TAG], :] = -10000
        self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000

        self.hidden = None

    def init_hidden(self, batch_size=20):
        return torch.randn(2, batch_size, self.hidden_dim // 2), torch.randn(2, batch_size, self.hidden_dim // 2)

    def _get_lstm_features(self, sentence_onehot):
        # self.hidden = self.init_hidden()
        # view是不对的：
        # embeds = self.word_embeds(sentence_onehot).view(len(sentence_onehot[0]), len(sentence_onehot), -1)
        # 不用pack_padded：
        # sentence_onehot_pad = batch_length_pad(sentence_onehot)
        # embeds = self.word_embeds(sentence_onehot_pad).transpose(0,1)
        # 用pack_padded，那这样后面所有的batch_size都要放在最前面了，batch_first，因为不能transpose
        sentence_onehot_pad = batch_length_pad(sentence_onehot)
        embeds = self.word_embeds(sentence_onehot_pad)
        embeds = batch_pack_padded(sentence_onehot, embeds)

        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        return lstm_out, embeds

    def _feat_splice(self, sentence, sentence_one_hot, lstm_feats, embeds):
        sentence_pos = batch_length_pad([t.pos for t in sentence])
        sentence_con = batch_length_pad([t.con for t in sentence])
        sentence_seg = batch_length_pad([t.seg for t in sentence])
        lstm_feats_split, _ = pad_packed_sequence(lstm_feats, batch_first=True)
        lstm_feats_split = lstm_feats_split.transpose(0, 1)
        eig = torch.cat((lstm_feats_split, embeds, sentence_pos.view(-1, len(sentence), 1),
                         sentence_con.view(-1, len(sentence), 1), sentence_seg.view(-1, len(sentence), 1)), 2)
        hidden_feat = self.hidden2tag(eig)
        split_feat = torch.tanh(hidden_feat)
        split_feat = batch_pack_padded(sentence_one_hot, split_feat.transpose(0, 1))
        split_feat, _ = pad_packed_sequence(split_feat, batch_first=True)
        split_feat = split_feat.transpose(0, 1)
        return split_feat

    def _forward_alg(self, lstm_feats, batch_len):
        # Do the forward algorithm to compute the partition function
        init_alphas = torch.full((BATCH_SIZE, self.tagset_size), -10000.)
        # START_TAG has all the score.
        for i in range(BATCH_SIZE):
            init_alphas[i][self.tag_to_ix[START_TAG]] = 0.

        # Wrap in a variable so that we will get automatic backprop
        forward_var = init_alphas

        # Iterate through the sentence
        for i, length in enumerate(batch_len[1:]):
            for feat in lstm_feats[batch_len[i]: length, :BATCH_SIZE - i, :]:
                alphas_t = []  # The forward tensors at this timestep
                for next_tag in range(self.tagset_size):
                    # broadcast the emission score: it is the same regardless of the previous tag
                    emit_score = feat[:, next_tag].view(-1, 1).expand(feat.size()[0], self.tagset_size)
                    # the i-th entry of trans_score is the score of transitioning to next_tag from i
                    trans_score = self.transitions[next_tag].view(1, -1).expand(feat.size()[0], self.tagset_size)
                    # The i-th entry of next_tag_var is the value for the edge (i -> next_tag) before we do log-sum-exp
                    next_tag_var = forward_var[: BATCH_SIZE - i] + trans_score + emit_score
                    # The forward variable for this tag is log-sum-exp of all the scores.
                    alphas_t.append(log_sum_exp(next_tag_var).view(-1, 1))
                forward_var[: BATCH_SIZE - i] = torch.cat(alphas_t, 1)
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]].expand(BATCH_SIZE, self.tagset_size)
        alpha = log_sum_exp(terminal_var)
        return alpha

    def _score_sentence(self, feats, tags, batch_len):
        # Gives the score of a provided tag sequence
        score = torch.zeros(BATCH_SIZE)
        tags = [torch.cat([torch.tensor([self.tag_to_ix[START_TAG]], dtype=torch.long), tag,
                           torch.zeros(feats.size()[0] - tag.size()[0])]).view(1, -1) for tag in tags]
        tags = torch.cat(tags).long()
        batch_len_with_start = [x+1 for x in batch_len]
        batch_len_with_start[0] = 0
        for i, length in enumerate(batch_len_with_start[1:]):
            for j, feat in enumerate(feats[batch_len_with_start[i]: length, :BATCH_SIZE - i, :]):
                pos = batch_len_with_start[i] + j
                score[:BATCH_SIZE - i] = score[:BATCH_SIZE - i] \
                    + self.transitions[tags[:BATCH_SIZE - i, pos + 1], tags[:BATCH_SIZE - i, pos]] \
                    + feat[range(BATCH_SIZE - i), tags[:BATCH_SIZE - i, pos + 1]]
        score = score + self.transitions[self.tag_to_ix[STOP_TAG], tags[range(BATCH_SIZE), batch_len[-1:0:-1]]]
        return score

    def loss_function(self, sentence, tags, batch_len):
        self.hidden = self.init_hidden()
        sentence_onehot = [t.onehot for t in sentence]
        # sentence_onehot = batch_pack_padded(sentence_onehot_list, sentence_onehot)
        lstm_feats, embeds = self._get_lstm_features(sentence_onehot)
        embeds, _ = pad_packed_sequence(embeds, batch_first=True)
        embeds = embeds.transpose(0, 1)
        splice_feats = self._feat_splice(sentence, sentence_onehot, lstm_feats, embeds)
        forward_score = self._forward_alg(splice_feats, batch_len)
        gold_score = self._score_sentence(splice_feats, tags, batch_len)
        return torch.sum(forward_score - gold_score)

    def _viterbi_decode(self, feats):
        backpointers = []

        # Initialize the viterbi variables in log space
        init_vvars = torch.full((1, self.tagset_size), -10000.)
        init_vvars[0][self.tag_to_ix[START_TAG]] = 0

        # forward_var at step i holds the viterbi variables for step i-1
        forward_var = init_vvars
        for feat in feats:
            bptrs_t = []  # holds the backpointers for this step
            viterbivars_t = []  # holds the viterbi variables for this step

            for next_tag in range(self.tagset_size):
                # next_tag_var[i] holds the viterbi variable for tag i at the
                # previous step, plus the score of transitioning
                # from tag i to next_tag.
                # We don't include the emission scores here because the max
                # does not depend on them (we add them in below)
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
            # Now add in the emission scores, and assign forward_var to the set
            # of viterbi variables we just computed
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)

        # Transition to STOP_TAG
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        # Follow the back pointers to decode the best path.
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        # Pop off the start tag (we dont want to return that to the caller)
        start = best_path.pop()
        assert start == self.tag_to_ix[START_TAG]  # Sanity check
        best_path.reverse()
        return path_score, best_path

    def forward(self, sentence):  # don't confuse this with _forward_alg above.
        # Get the emission scores from the BiLSTM
        self.hidden = self.init_hidden(1)
        embeds = self.word_embeds(sentence.onehot).view(len(sentence.onehot), 1, -1)
        lstm_feats, self.hidden = self.lstm(embeds, self.hidden)
        eig = torch.cat((lstm_feats[:, 0, :], embeds[:, 0, :], sentence.pos.view(-1, 1),
                         sentence.con.view(-1, 1), sentence.seg.view(-1, 1)), 1)
        hidden_feat = self.hidden2tag(eig)
        splice_feats = torch.tanh(hidden_feat)
        # Find the best path, given the features.
        score, tag_seq = self._viterbi_decode(splice_feats)
        # return score, tag_seq
        return tag_seq


FLAG = True
if __name__ == "__main__":
    # 导入数据与数据预处理
    data, characters, data_feature = input_layer.preprocess('../BIO_data/data_jd.txt')
    with open('../BIO_data/data_bio.txt', 'r', ) as f:
        result = f.read().strip('\n').split(sep='\n')
    with open('../BIO_data/data_jd.txt', 'r', encoding='utf-8') as f:
        data_raw = f.read().strip('\n').split(sep='\n')

    #  取50个
    result = result[:25]
    data = data[:25]
    data_feature = data_feature[:25]
    data_raw = data_raw[:25]

    # 随机化数据集并选取前80%作训练集，后20%作测试集
    random.seed(836)
    random.shuffle(data)
    random.seed(836)
    random.shuffle(result)
    random.seed(836)
    random.shuffle(data_raw)
    train_data = data[:int(len(data) * 0.8)]
    train_data_raw = data_raw[:int(len(data) * 0.8)]
    train_result = result[:int(len(data) * 0.8)]
    test_data = data[int(len(data) * 0.8):]
    test_result = result[int(len(data) * 0.8):]

    # 搭建模型
    tag_to_ix = {"B": 0, "I": 1, "O": 2, START_TAG: 3, STOP_TAG: 4}
    if FLAG:
        model = BiLSTM_CRF(vocab_size=len(characters), tag_to_ix=tag_to_ix, embedding_dim=EMBEDDING_DIM,
                           hidden_dim=HIDDEN_DIM)
    else:
        model = BiLSTM_CRF_no_batch(vocab_size=len(characters), tag_to_ix=tag_to_ix, embedding_dim=EMBEDDING_DIM,
                                    hidden_dim=HIDDEN_DIM)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    print('=============开始BiLSTM+CRF模型的训练=============')
    for epoch in range(EPOCH):
        print('%4d' % epoch, end='')
        # 检查用
        # def show_batch():
        #     for epoch in range(1):
        #         for step, batch in enumerate(train_data_loader):
        #             # training
        #             # print("steop:{}, batch_y:{}".format(step, batch))
        #             item = batch[0]
        #             print(item)
        #             print(train_data_raw.index(item))
        # show_batch()
        if not FLAG:
            for i in range(len(train_data)):
                model.zero_grad()
                tag_to_ix_new = {"B-AbilityTag": 0, "I-AbilityTag": 1, "O": 2, START_TAG: 3, STOP_TAG: 4,
                                 'B-LevelTag': 2,
                                 'I-LevelTag': 2}
                tags = train_result[i].split(' ')
                targets = torch.tensor([tag_to_ix_new[t] for t in tags], dtype=torch.long)

                loss = model.loss_function(train_data[i], targets)
                # loss.backward()
                # optimizer.step()
        else:
            train_data_loader = Data.DataLoader(dataset=train_data_raw,
                                                batch_size=BATCH_SIZE,
                                                shuffle=True, )
            for step, batch in enumerate(train_data_loader):
                # batch: list[str]
                # Step 0. 找对应的index，整理出batch_feature和batch_result，格式不变
                # 为了后续的pad和pack，这里必须要先对Batch内的sample进行长度排序
                batch.sort(key=lambda t: len(t), reverse=True)
                batch_len = [len(t) for t in batch]
                batch_index = [train_data_raw.index(t) for t in batch]
                batch_feature = [train_data[t] for t in batch_index]
                batch_result = [train_result[t] for t in batch_index]

                # Step 1. Remember that Pytorch accumulates gradients.
                # We need to clear them out before each instance
                model.zero_grad()

                # Step 2. Get our inputs ready for the network, that is,
                # turn them into Tensors of word indices.
                tag_to_ix_new = {"B-AbilityTag": 0, "I-AbilityTag": 1, "O": 2, START_TAG: 3, STOP_TAG: 4,
                                 'B-LevelTag': 2,
                                 'I-LevelTag': 2}
                batch_tags = [t.split(' ') for t in batch_result]
                batch_targets = [torch.tensor([tag_to_ix_new[t] for t in tags], dtype=torch.long) for tags in
                                 batch_tags]

                # Step 3. Run our forward pass.
                batch_len.append(0)
                batch_len.reverse()
                loss = model.loss_function(batch_feature, batch_targets, batch_len)
                print('%10.2f' % loss.item(), end='')

                # Step 4. Compute the loss, gradients, and update the parameters by
                # calling optimizer.step()
                loss.backward()
                optimizer.step()

            print()

    # 保存训练好的模型
    output_path = 'ner_trained_model.cpt'
    torch.save(model, output_path)
    print('=============训练结束，保存训练好的模型=============\n\n')

    # # 加载训练好的模型
    # print('=============加载训练好的模型，进行测试=============')
    # model_path = 'ner_trained_model.cpt'
    # # ?
    # trained_ner_model = torch.load(model_path)
    # # model.eval() ?
    # with torch.no_grad():
    #     precheck_sent = test_data[0]
    #     print('训练后模型的预测：' + str(model(precheck_sent)))
