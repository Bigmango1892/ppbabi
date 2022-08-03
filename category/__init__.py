from LAC import LAC
import pickle
from fuzzywuzzy import fuzz
import os

module_path = os.path.dirname(os.path.realpath(__file__))
# 全局变量
lac = LAC(mode='rank')
with open(module_path + '/category_separated.pkl', 'rb') as f:
    my_format = pickle.load(f)


def clean(word):
    start = word.find('（')
    if start != -1:
        end = word.find('）')
        sub = word[start:end + 1]
        word = word.replace(sub, '')
    start = word.find('【')
    if start != -1:
        end = word.find('】')
        sub = word[start:end + 1]
        word = word.replace(sub, '')
    black_list = ['*', '师', '经理', '助理', '专员', '主管', '经纪人', '高级', '实习', '管培生', '管理培训生', '校招', '春招']
    for i in black_list:
        word = word.replace(i, '')
    return word


def lac_clean(record):
    rank_result = lac.run(record)
    pivot = [t for t in range(len(rank_result[0])) if rank_result[1][t] != 'LOC' and rank_result[1][t] != 'w']
    one_word, k = [], 0
    while k < len(pivot):
        one_word.append(rank_result[0][pivot[k]])
        k += 1
    return ''.join(one_word)


def match_one(tb_match, category=my_format, threshold=50):
    max_score, best_match = 0, None
    ci = clean(tb_match)
    if len(ci) == 0:
        return None
    ci = lac_clean(ci)
    for key, words in category.items():
        for word in words:
            score = fuzz.ratio(word, ci)
            if score > max_score:
                best_match = key
                max_score = score
    return best_match if max_score >= threshold else None


if __name__ == "__main__":
    match = match_one('硬件开发工程师')
    print(match)
