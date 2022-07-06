# 统计能力和能力之间相似程度（拥有能力1后获取能力2的难易程度）
import pickle
import pandas as pd
import numpy as np

with open('../weight/words_reseted.data', 'rb') as f:
    words_reset = pickle.load(f)
df = pd.read_csv('../JD_category/岗位技能词_按岗位大类std_100以上.csv')
words_all_set = set([j for i in words_reset for j in i])
ability_words_set = set()
for i in range(len(df)):
    for j in range(10):
        if pd.notna(df.loc[i, '技能词{}'.format(j+1)]):
            ability_words_set.add(df.loc[i, '技能词{}'.format(j+1)])
ability_words = list(ability_words_set)
words_all = ability_words + list(words_all_set.symmetric_difference(ability_words_set))
with open('ability_words.txt', 'w', encoding='utf8') as f:
    print('\n'.join(ability_words), file=f, end='')
ease = np.zeros(shape=(len(words_all), len(words_all)))

for i in range(len(words_reset)):
    index = []
    for word in words_reset[i]:
        index.append(words_all.index(word))
    for i_ind in range(len(index)):
        for j_ind in range(i_ind + 1, len(index)):
            ease[index[i_ind], index[j_ind]] = ease[index[i_ind], index[j_ind]] + 1

out_data = []
for i in range(len(ability_words)):
    index = np.argsort(ease[i])[-1:-11:-1]
    for j in range(len(index)-1, -1, -1):
        if ease[i][index[j]] == 0:
            index = np.delete(index, j)
    out_data.append([ability_words[i]])
    for j in index:
        out_data[-1].extend([ability_words[j], ease[i, j]])
columns = ['技能名称']
for i in range(10):
    columns.extend(['相似技能{}'.format(i+1), '相似度{}'.format(i+1)])
out_data = pd.DataFrame(data=out_data, columns=columns)
out_data.to_csv('相似能力词.csv', index=False)
