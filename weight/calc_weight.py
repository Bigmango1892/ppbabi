import numpy as np
import pickle


def calc_std():
    with open('words_reseted.data', 'rb') as f:
        words = pickle.load(f)
    category = ['行业', '岗位大类', '岗位细分']

    for cate in category:
        with open('../JD_category/JD按{}分类.txt'.format(cate), 'r', encoding='utf8') as f:
            groups = {x.split(':')[0]: [int(y) for y in x.split(':')[1].strip('[]').split(', ')] for x in
                      f.read().split('\n')}
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
        std_factor = {}
        for key, count in zip(word_dic, words_count_normalized):
            if sum(count) == 0:
                std_factor[key] = 0
            else:
                std_factor[key] = np.std(count) / np.mean(count)
        with open('./std_factor0630_{}.data'.format(cate), 'wb') as f:
            pickle.dump(std_factor, f)
