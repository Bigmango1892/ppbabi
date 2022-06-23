import random
import input_layer
import torch
import torch.nn as nn
import xlwt

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
LEARNING_RATE = 0.001
# epoch（文中设置为）
EPOCH = 100


def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return idx.item()


def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)


# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))


class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim):
        super(BiLSTM_CRF, self).__init__()
        # 设置模型参数
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)

        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
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

        self.hidden = self.init_hidden()

    def init_hidden(self):
        return torch.randn(2, 1, self.hidden_dim // 2), torch.randn(2, 1, self.hidden_dim // 2)

    def _get_lstm_features(self, sentence_onehot):
        self.hidden = self.init_hidden()
        # Embedding layer -- input: sentence code; output: embedded matrix
        # view方法重新整合tensor维度，前两参数为维度，-1表示自动匹配
        embeds = self.word_embeds(sentence_onehot).view(len(sentence_onehot), 1, -1)
        # LSTM layer -- input: eigenvalue, hidden layers matrix; output: LSTM out, new hidden layers matrix
        # 之后重新整合lstm_out维度
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        lstm_out = lstm_out.view(len(sentence_onehot), self.hidden_dim)
        return lstm_out, embeds.view(len(sentence_onehot), self.hidden_dim)

    def _feat_splice(self, sentence, lstm_feats, embeds):
        eig = torch.cat((lstm_feats, embeds,
                         sentence.pos.view(-1, 1), sentence.con.view(-1, 1), sentence.seg.view(-1, 1)), 1)
        hidden_feat = self.hidden2tag(eig)
        split_feat = torch.tanh(hidden_feat)
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
        splice_feats = self._feat_splice(sentence, lstm_feats, embeds)
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
        # Get the emission scores from the BiLSTM
        lstm_feats, embeds = self._get_lstm_features(sentence.onehot)
        splice_feats = self._feat_splice(sentence, lstm_feats, embeds)
        # Find the best path, given the features.
        score, tag_seq = self._viterbi_decode(splice_feats)
        # return score, tag_seq
        return tag_seq


# # 导入数据与数据预处理
data, characters = input_layer.preprocess('../BIO_data/data_jd.txt')
with open('../BIO_data/data_bio.txt', 'r') as f:
    result = f.read().strip('\n').split(sep='\n')
# # 随机化数据集并选取前80%作训练集，后20%作测试集
random.seed(836)
random.shuffle(data)
random.seed(836)
random.shuffle(result)
train_data = data[:int(len(data) * 0.4)]
train_result = result[:int(len(data) * 0.4)]
test_data = data[int(len(data) * 0.4):]
test_result = result[int(len(data) * 0.4):]
#
# # 搭建模型
# tag_to_ix = {"B": 0, "I": 1, "O": 2, START_TAG: 3, STOP_TAG: 4}
# model = BiLSTM_CRF(vocab_size=len(characters), tag_to_ix=tag_to_ix, embedding_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_DIM)
# optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
#
# print('=============开始BiLSTM+CRF模型的训练=============')
# for epoch in range(EPOCH):
#     for i in range(len(train_data)):
#         # Step 1. Remember that Pytorch accumulates gradients.
#         # We need to clear them out before each instance
#         model.zero_grad()
#
#         # Step 2. Get our inputs ready for the network, that is,
#         # turn them into Tensors of word indices.
#         tags = train_result[i].split(' ')
#         tag_to_ix_new = {"B-AbilityTag": 0, "I-AbilityTag": 1, "O": 2, START_TAG: 3, STOP_TAG: 4, 'B-LevelTag': 2, 'I-LevelTag': 2}
#         targets = torch.tensor([tag_to_ix_new[t] for t in tags], dtype=torch.long)
#
#         # Step 3. Run our forward pass.
#         loss = model.loss_function(train_data[i], targets)
#
#         # Step 4. Compute the loss, gradients, and update the parameters by
#         # calling optimizer.step()
#         loss.backward()
#         optimizer.step()
#
#     print(f'模型训练第{epoch}轮的Loss值为：{loss[0]}')
#
# # 保存训练好的模型
# output_path = 'ner_trained_model.cpt'
# torch.save(model, output_path)
# print('=============训练结束，保存训练好的模型=============\n\n')

# 加载训练好的模型
print('=============加载训练好的模型，进行测试=============')
# 导入测试数据
data, characters = input_layer.preprocess('../BIO_data/data_jd.txt')
with open('../BIO_data/data_bio.txt', 'r') as f:
    result = f.read().strip('\n').split(sep='\n')
test_data = data
test_result = result
model_path = 'ner_trained_model.cpt'
trained_ner_model = torch.load(model_path)
tag_to_ix_new = {"B-AbilityTag": 0, "I-AbilityTag": 1, "O": 2, START_TAG: 3, STOP_TAG: 4, 'B-LevelTag': 2, 'I-LevelTag': 2}
with torch.no_grad():
    f = xlwt.Workbook('encoding = utf-8')
    sheet1 = f.add_sheet('sheet1', cell_overwrite_ok=True)
    for i in range(len(test_data)):
        print(i)
        precheck_sent = test_data[i]
        test_text = precheck_sent.text

        test_pred = trained_ner_model(precheck_sent)
        test_pred_1 = [k for k, j in enumerate(test_pred) if j == 1]
        test_pred_0 = [k for k, j in enumerate(test_pred) if j == 0]
        words = [test_text[t] for t in sorted(test_pred_1 + test_pred_0)]
        pivot = [k for k in [sorted(test_pred_1 + test_pred_0).index(t) for t in test_pred_0]]
        if len(pivot) == 1:
            test_pred_words= [''.join(words[pivot[-1]:])]
        elif len(pivot) == 0:
            test_pred_words = []
        else:
            test_pred_words = [''.join(words[pivot[k]:pivot[k+1]]) for k in range(len(pivot)-1)]
            test_pred_words.append(''.join(words[pivot[-1]:]))

        tags = test_result[i].split(' ')
        # print(tags)
        test_gt = [tag_to_ix_new[t] for t in tags]
        test_gt_0 = [k for k, j in enumerate(test_gt) if j == 0]
        test_gt_1 = [k for k, j in enumerate(test_gt) if j == 1]
        test_gt_01 = [k for k, j in enumerate(test_gt) if j == 1 or j == 0]
        words = [test_text[t] for t in sorted(test_gt_1 + test_gt_0)]
        pivot = [k for k in [sorted(test_gt_1 + test_gt_0).index(t) for t in test_gt_0]]
        if len(pivot) == 1:
            test_gt_words = [''.join(words[pivot[-1]:])]
        elif len(pivot) == 0:
            test_gt_words = []
            # print(test_text)
            # print(tags)
            # print(test_pred_words)
        else:
            test_gt_words = [''.join(words[pivot[k]:pivot[k + 1]]) for k in range(len(pivot) - 1)]
            test_gt_words.append(''.join(words[pivot[-1]:]))

        sheet1.write(i, 0, ','.join(test_pred_words))
        sheet1.write(i, 1, ','.join(test_gt_words))
        sheet1.write(i, 2, test_text)
    f.save('test_result_all.xls')

        # print('训练后模型的预测：' + str(trained_ner_model(precheck_sent)))
