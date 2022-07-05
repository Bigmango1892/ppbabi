import pickle
import weight.analy as analyse
import pandas as pd

with open('JD按岗位细分分类.txt', 'r', encoding='utf8') as f:
    index = {x.split(':')[0]: [int(y) for y in x.split(':')[1].strip('[]').split(', ')] for x in f.read().split('\n')}
with open('../weight/words_reseted.data', 'rb') as f:
    words = pickle.load(f)
length = {key: len(value) for key, value in index.items()}
for key in length:
    if length[key] < 100:
        del index[key]
category = ['行业', '岗位大类', '岗位细分']

for cate in category:
    analyse.std_file = 'std_factor0630_{}.data'.format(cate)
    output_data = []
    for key, value in index.items():
        analyse.key_word = key
        ability = [words[i] for i in value if i < len(words)]
        analyse.abilities = ability
        results, ind = analyse.count_index()
        output_abi = [(list(results.keys())[ind[i]], list(results.values())[ind[i]]) for i in range(10) if i < len(ind)]
        num = len(output_data)
        output_data.append([key])
        for i in range(len(output_abi)):
            output_data[num].extend([output_abi[i][0], output_abi[i][1]])

    columns = ['岗位名称']
    for i in range(10):
        columns.extend(['技能词{}'.format(i+1), '特征值{}'.format(i+1)])
    output_data = pd.DataFrame(data=output_data, columns=columns)
    output_data.to_csv('岗位技能词_按{}std_100以上.csv'.format(cate), index=False)
