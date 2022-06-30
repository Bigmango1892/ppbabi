import numpy as np
import pickle
import pandas as pd


def reset_words(words: list, polymerize: dict):
    polymerize_set = set([x for i in polymerize.values() for x in i])
    for i in range(len(words)):
        for j in range(len(words[i])):
            if words[i][j] in polymerize_set:
                for _key, _value in polymerize.items():
                    if words[i][j] in _value:
                        words[i][j] = _key
                        break


if __name__ == '__main__':
    titles = pd.read_csv('./job_titles.csv')
    with open('words_all.data', 'rb') as f:
        words = pickle.load(f)
    with open('polymerize0630.data', 'rb') as f:
        polymerize = pickle.load(f)
    reset_words(words, polymerize)

    groups = titles.groupby(by='行业').groups
    group_len = {key: len(value) for key, value in groups.items()}
    group_count = {}
    for key, value in groups.items():
        tf_count = {}
        for pos in value:
            for word in words[pos]:
                tf_count[word] = tf_count.get(word, 0) + 1
        group_count[key] = tf_count

    word_dic = list(set([j for i in words for j in i]))
    words_count = [[dic.get(word, 0) for dic in group_count.values()] for word in word_dic]
    # word_len_factor = [sum(count) for count in words_count]
    words_count_normalized = [[n/list(group_len.values())[i] for i, n in enumerate(count)] for count in words_count]
    std_factor = {key: np.std(count)/np.mean(count) for key, count in zip(word_dic, words_count_normalized)}
    with open('./std_factor0630.data', 'wb') as f:
        pickle.dump(std_factor, f)
