import pickle
import numpy as np
import pymysql
import pandas as pd

with open('./polymerize0721_lower.pkl', 'rb') as f:
    poly = pickle.load(f)
poly_set = set([j for i in poly.values() for j in i])


def change_words(string):
    if not isinstance(string, str):
        return string
    words = string.split(';')
    words_changed = []
    for origin_word in words:
        word = origin_word.lower()
        if word not in poly_set:
            words_changed.append(origin_word)
            continue
        for key, value in poly.items():
            if word in value:
                words_changed.append(key)
                continue
    return ';'.join(words_changed)


if __name__ == '__main__':
    connect = pymysql.connect(
        host='192.168.3.4',
        user='root',
        port=3306,
        db='paperball_db'
    )
    cursor = connect.cursor()
    ext = 'SELECT * FROM `shixiseng_school_recruitment_JD`'
    columns = ['工种', '岗系', '岗位名称目录', '岗位具体名称', 'code', '最低薪资', '最高薪资', '资质要求1',
               '资质要求2', '学历要求', '标签', '职位描述', '所属公司', '所属公司_code', '公司介绍', '公司所在行业',
               '公司性质', '公司规模', '公司所在地', '公司标签', '工作内容（总的）', '能力关键词', '能力关键词（同义替换）',
               '简历要求', '截止日期', '职位所在地', '网站链接']
    cursor.execute(ext)
    result = cursor.fetchall()
    df = pd.DataFrame(data=result, columns=columns)
    df['能力关键词（同义替换）'] = df['能力关键词'].apply(change_words)
    df.to_csv('result.csv', index=False)
