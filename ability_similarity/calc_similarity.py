# 统计能力和能力之间相似程度（拥有能力1后获取能力2的难易程度）
import pickle
import pandas as pd
import numpy as np
import pymysql
import os


def fetch_data():
    connect = pymysql.connect(
        host='192.168.3.4',
        port=3306,
        user='root',
        database='paperball_db'
    )
    cursor = connect.cursor()
    sql = "SELECT `能力关键词（同义替换）` FROM `shixiseng_school_recruitment_jd` WHERE `能力关键词（同义替换）` is not null "
    cursor.execute(sql)
    result = cursor.fetchall()
    cursor.close()
    connect.close()
    return [x[0].split(';') for x in result]


def calc_ease():
    jd_ability = fetch_data()
    if os.path.exists('dict.txt'):
        with open('dict.txt', 'r') as f:
            words_all = f.readlines()
        words_all = [x.split(' ')[0] for x in words_all]
    else:
        words_all = list(set([j for i in jd_ability for j in i]))

    ease = np.zeros(shape=(len(words_all), len(words_all)))
    for i in range(len(jd_ability)):
        index = []
        for word in jd_ability[i]:
            index.append(words_all.index(word))
        func1(index, ease)

    with open('ease.pkl', 'wb') as f:
        pickle.dump(ease, f)
    if not os.path.exists('dict.txt'):
        with open('dict.txt', 'w') as f:
            print('\n'.join([f'{i} {word}' for i, word in enumerate(words_all)]), file=f)


def func1(index, ease):
    for i_ind in range(len(index)):
        for j_ind in range(i_ind + 1, len(index)):
            ease[index[i_ind], index[j_ind]] = ease[index[i_ind], index[j_ind]] + 1


def func2(index, ease):
    for i_ind in range(len(index)):
        for j_ind in range(i_ind + 1, len(index)):
            ease[index[i_ind], index[j_ind]] = ease[index[i_ind], index[j_ind]] + 1

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
    calc_ease()
