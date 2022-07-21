import numpy as np
import pickle
import pandas as pd
import pymysql


def load_data():
    columns = ["工种", "岗系", "岗位名称目录", "能力关键词（同义替换）"]
    connect = pymysql.connect(
        host='192.168.3.4',
        user='root',
        port=3306,
        db='paperball_db'
    )
    cursor = connect.cursor()
    ext = '''SELECT 工种,岗系,岗位名称目录,能力关键词（同义替换） FROM `shixiseng_school_recruitment_JD` 
    WHERE `能力关键词（同义替换）` IS NOT NULL AND `工种` IS NOT NULL'''
    cursor.execute(ext)
    result = cursor.fetchall()
    df = pd.DataFrame(data=result, columns=columns)
    df['能力关键词（同义替换）'] = df['能力关键词（同义替换）'].apply(lambda x: x.split(';'))
    return df


def calc_std():
    data = load_data()
    category = ['工种', '岗系', '岗位名称目录']

    for cate in category:
        groups = data.groupby(by=cate).groups
        group_len = {key: len(value) for key, value in groups.items()}
        group_count = {}
        for key, value in groups.items():
            tf_count = {}
            for pos in value:
                for word in data.loc[pos, '能力关键词（同义替换）']:
                    tf_count[word] = tf_count.get(word, 0) + 1
            group_count[key] = tf_count

        word_dic = list(set([j for i in range(len(data)) for j in data.loc[i, '能力关键词（同义替换）']]))
        words_count = [[dic.get(word, 0) for dic in group_count.values()] for word in word_dic]
        # word_len_factor = [sum(count) for count in words_count]
        words_count_normalized = [[n/list(group_len.values())[i] for i, n in enumerate(count)] for count in words_count]
        std_factor = {}
        for key, count in zip(word_dic, words_count_normalized):
            if sum(count) == 0:
                std_factor[key] = 0
            else:
                std_factor[key] = np.std(count) / np.mean(count)
        with open('./std_factor/std_factor0721_{}.data'.format(cate), 'wb') as f:
            pickle.dump(std_factor, f)


if __name__ == '__main__':
    calc_std()
