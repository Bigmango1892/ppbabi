import numpy as np
import math

key_word = '用户运营'


def length_fun(n: int):
    return n


def position_fun(p: float):
    return p

with open('../testing_data/result/{}.txt'.format(key_word), 'r') as f:
    abilities = [x.split(',') for x in f.read().strip('\n').split('\n')]

count_pos = []
for words in abilities:
    item = []
    for i, word in enumerate(words):
        word = word.replace('编程', '').replace('能力', '').lower()
        if len(words) == 1:
            item.append(0)
        else:
            item.append(i / (len(words) - 1))
    count_pos.append(item)

count_index = {}
for words, poses in zip(abilities, count_pos):
    for word, pos in zip(words, poses):
        if word not in count_index:
            count_index[word] = []
        count_index[word].append(pos)

count_mean = {key: np.mean(value) for key, value in count_index.items()}
count_len = {key: len(value) for key, value in count_index.items()}
results = {key: length_fun(count_len[key]) * position_fun(1 - count_mean[key]) for key in count_len.keys()}
index = np.argsort(list(results.values()))[-1::-1]
for i in range(20):
    print(list(results.keys())[index[i]], list(results.values())[index[i]])
print()