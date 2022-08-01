# 统计能力和能力之间相似程度（拥有能力1后获取能力2的难易程度）
import pickle
import pandas as pd
from sshtunnel import SSHTunnelForwarder
import numpy as np
import pymysql
import pymysql.cursors
import os
import time
from scipy.sparse import coo_matrix


def fetch_data():
    server = SSHTunnelForwarder(
        ssh_address_or_host=('106.52.154.120', 22),
        ssh_username='root',  # 跳转机的用户
        ssh_password='9a93r6a1l//',  # 跳转机的密码
        remote_bind_address=('localhost', 3306))
    server.start()
    connect = pymysql.connect(
        host='127.0.0.1',  # 此处必须是是127.0.0.1
        user='root',
        password='paperball',
        database='paperball_raw_data',
        port=server.local_bind_port,
        cursorclass=pymysql.cursors.DictCursor)
    cursor = connect.cursor()
    sql = "SELECT `能力关键词（同义替换）` FROM `shixiseng_school_recruitment_jd` WHERE `能力关键词（同义替换）` is not null "
    cursor.execute(sql)
    result = cursor.fetchall()
    cursor.close()
    connect.close()
    return [x['能力关键词（同义替换）'].split(';') for x in result]


def calc_ease():
    jd_ability = fetch_data()
    if os.path.exists('dict.txt'):
        with open('dict.txt', 'r') as f:
            words_all = [x.strip('\n').split(': ')[1] for x in f.readlines()]
    else:
        words_all = list(set([j for i in jd_ability for j in i]))

    ease = np.zeros(shape=(len(words_all), len(words_all)))
    for jd_words in jd_ability:
        index = [words_all.index(word) for word in jd_words]
        func1(index, ease)

    return ease, words_all


def func1(index, ease):
    for i_ind in range(len(index)):
        for j_ind in range(i_ind + 1, len(index)):
            ease[index[i_ind], index[j_ind]] = ease[index[i_ind], index[j_ind]] + 1


def func2(index, ease):
    for i_ind in range(len(index)):
        alpha, param = 0.9, 1
        for j_ind in range(i_ind + 1, len(index)):
            ease[index[i_ind], index[j_ind]] = ease[index[i_ind], index[j_ind]] + param
            param *= alpha

# ability_words = words_all
# out_data = []
# for i in range(len(ability_words)):
#     index = np.argsort(ease[i])[-1:-11:-1]
#     for j in range(len(index)-1, -1, -1):
#         if ease[i][index[j]] == 0:
#             index = np.delete(index, j)
#     out_data.append([ability_words[i]])
#     for j in index:
#         out_data[-1].extend([ability_words[j], ease[i, j]])
# columns = ['技能名称']
# for i in range(10):
#     columns.extend(['相似技能{}'.format(i+1), '相似度{}'.format(i+1)])
# out_data = pd.DataFrame(data=out_data, columns=columns)
# out_data.to_csv('相似能力词.csv', index=False)


if __name__ == "__main__":
    ease, words_all = calc_ease()

    # 输出相似度矩阵
    coo_ease = coo_matrix(ease)
    with open('coo_ease_{}.pkl'.format(int(time.time())), 'wb') as f:
        pickle.dump(coo_ease, f)
    if not os.path.exists('dict.txt'):
        with open('dict.txt', 'w') as f:
            print('\n'.join([f'{i}: {word}' for i, word in enumerate(words_all)]), file=f)
