import numpy as np


def _idf_list(word: str):
    if word in IDF:
        return IDF[word]
    else:
        return 0


def _calc_idf(word_list: list):
    return [[_idf_list(word) for word in item] for item in word_list]


def _calc_weight(word_list: list):
    title_pos = []
    for pos in range(len(word_list)):
        if len(''.join(word_list[pos])) < 8:
            title_pos.append(pos)
    title_pos.append(len(word_list))
    weight = []
    for i in range(len(title_pos) - 1):
        weight.append(0)
        weight.extend(_activation_func(title_pos[i+1] - title_pos[i] - 1))
    return weight


def _activation_func(n: int):
    alpha = 1
    result = []
    tmp = 1
    for i in range(n):
        result.append(tmp)
        tmp = tmp * alpha
    return result


def jdeig(word_list: list):
    idf = _calc_idf(word_list)
    weight = _calc_weight(word_list)
    for i in range(len(weight)):
        idf[i] = list(np.array(idf[i]) * weight[i])
    return idf


with open('./ppbabi/renew/idf.txt', 'r', encoding='utf8') as f:
    IDF = {x.split(sep=' ')[0]: float(x.split(sep=' ')[1].strip('\n')) for x in f.readlines()}