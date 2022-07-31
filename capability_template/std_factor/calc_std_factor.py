import numpy as np
import pickle
import pandas as pd
import pymysql
import time
# 输出标准差因子，存储于./std_factor/中. 输入可为文件目录，或为 'mysql' 后加数据库内容，具体参数为pymysql.connect参数
# 输出为std_factor_{time_code}_{category}.pkl
# 如：std_factor_20220727_岗系.pkl


def calc_std(path: str, connect: pymysql.connect):
    if path == 'mysql':
        data = load_data_from_mysql(connect)
    else:
        data = load_data_from_csv(path)
    category = ['工种', '岗系', '岗位名称目录']
    date_code = time.strftime("%Y%m%d", time.localtime(time.time()))

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
        with open('./data/std_factor_{}_{}.pkl'.format(date_code, cate), 'wb') as f:
            pickle.dump(std_factor, f)


def load_data_from_mysql(connect: pymysql.connect):
    columns = ["工种", "岗系", "岗位名称目录", "能力关键词（同义替换）"]
    cursor = connect.cursor()
    ext = '''SELECT 工种,岗系,岗位名称目录,能力关键词（同义替换） FROM `shixiseng_school_recruitment_JD` 
    WHERE `能力关键词（同义替换）` IS NOT NULL AND `工种` IS NOT NULL'''
    cursor.execute(ext)
    result = cursor.fetchall()
    cursor.close()
    df = pd.DataFrame(data=result, columns=columns)
    df['能力关键词（同义替换）'] = df['能力关键词（同义替换）'].apply(lambda x: x.split(';'))
    return df


def load_data_from_csv(path: str):
    columns = ["工种", "岗系", "岗位名称目录", "能力关键词（同义替换）"]
    df = pd.read_csv(path)[columns]
    df['能力关键词（同义替换）'] = df['能力关键词（同义替换）'].apply(lambda x: x.split(';'))
    return df


if __name__ == '__main__':
    con = pymysql.connect(host='192.168.3.4', port=3306, user='root', db='paperball_db')
    calc_std('mysql', con)  # 第一参数为 'mysql' 或.csv的路径名
    con.close()
