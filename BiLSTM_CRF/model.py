import pandas as pd
import input_layer
import torch
import torch.nn as nn

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
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)

        # Matrix of transition parameters.  Entry i,j is the score of
        # transitioning *to* i *from* j.
        self.transitions = nn.Parameter(
            torch.randn(self.tagset_size, self.tagset_size))

        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag
        self.transitions.data[tag_to_ix[START_TAG], :] = -10000
        self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000

        self.hidden = self.init_hidden()

    def forward(self):
        pass


# 数据与模型准备
data = input_layer.preprocess('./jd_sample.csv', '工作内容（总的）')
model = BiLSTM_CRF(1, data, 1, 1)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

print('=============开始BiLSTM+CRF模型的训练=============')
for epoch in range(EPOCH):
    for sentence, tags in train_data:
        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        model.zero_grad()

        # Step 2. Get our inputs ready for the network, that is,
        # turn them into Tensors of word indices.

        # Step 3. Run our forward pass.
        loss = model.forward()

        loss.backward()
