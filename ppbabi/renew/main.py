import pandas as pd
from cutwords import jdwords
from calceig import jdeig
import numpy as np


if __name__ == '__main__':
    tmp = 0
    jd_path = './JD.csv'
    output_path = '../jobabi/'
    jd_df = pd.read_csv(jd_path, encoding='utf8')
    for first_dir in set(jd_df['行业']):
        jd_df_1 = jd_df.loc[jd_df['行业'] == first_dir]
        for second_dir in set(jd_df_1['岗位大类']):
            jd_df_2 = jd_df_1.loc[jd_df_1['岗位大类'] == second_dir]
            for third_dir in set(jd_df_2['岗位细分']):
                jd_df_3 = jd_df_2.loc[jd_df_2['岗位细分'] == third_dir]
                # count的位置决定按哪一级目录分
                count = {}
                for describe in jd_df_3['工作内容（总的）']:
                    if pd.notna(describe):
                        jd_words = jdwords(describe)
                        words_eig = jdeig(jd_words)
                        for i in range(len(words_eig)):
                            for j in range(len(words_eig[i])):
                                if words_eig[i][j] != 0:
                                    count[jd_words[i][j]] = count.get(jd_words[i][j], 0) + words_eig[i][j]
                count_sort = np.argsort(np.array(list(count.values())))[-1::-1]
                word_sort = []
                for pos in count_sort:
                    word_sort.append(list(count.keys())[pos])
                print('{} {} {} {}'.format(first_dir, second_dir, third_dir, word_sort))
                tmp = tmp + 1
                if tmp > 2:
                    quit()
