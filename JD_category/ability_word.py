import pickle
import weight.analy as analyse
import pandas as pd
from weight.calc_weight import load_data


# df = load_data()
df = pd.read_csv('../按【要求】提取JD_能力词.csv')
df['能力关键词（同义替换）'] = df['能力关键词（同义替换）'].apply(lambda x: x.split(';') if isinstance(x, str) else x)
df = df.loc[pd.notna(df['能力关键词（同义替换）'])].reset_index(drop=True)

index = df.groupby(by='岗位名称目录').groups
words = df['能力关键词（同义替换）'].to_list()
length = {key: len(value) for key, value in index.items()}
for key in length:
    if length[key] < 50:
        del index[key]
category = ['工种', '岗系', '岗位名称目录']

for cate in category:
    analyse.std_file = 'std_factor0721_{}.data'.format(cate)
    output_data = []
    for key, value in index.items():
        analyse.key_word = key
        if key == '商务司机':
            print()
        ability = [words[i] for i in value]
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
    output_data.to_csv('0725岗位技能词_按{}std_50以上_要求.csv'.format(cate), index=False)
