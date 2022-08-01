import pickle
import pandas as pd
import numpy as np


def calc_one_template(abilities: list, std_name: str):
    count_pos = []  # 用以记录每一条JD中能力词的位置信息，从0至1线性递增
    for ability_words in abilities:
        ability_words_len = len(ability_words)
        if ability_words_len == 1:
            count_pos.append([0])
            continue
        item = [i / (ability_words_len - 1) for i in range(len(ability_words))]
        count_pos.append(item)

    count_index = {}    # 用以统计不同词出现的位置信息，即对上述结果按不同的词进行聚合
    for ability_words, poses in zip(abilities, count_pos):
        for word, pos in zip(ability_words, poses):
            if word not in count_index:
                count_index[word] = []
            count_index[word].append(pos)

    count_mean = {key: np.mean(value) for key, value in count_index.items()}    # 位置量的均值
    count_len = {key: len(value) for key, value in count_index.items()}         # 词频
    result = {key: count_len[key] * (1 - count_mean[key]) for key in count_len.keys()}

    with open(f'std_factor/data/{std_name}', 'rb') as f:
        std_factor = pickle.load(f)
    result = {key: value * std_factor[key] for key, value in result.items() if key in std_factor}  # 引入std参量

    keys = list(result.keys())
    values = [result[key] for key in keys]
    index = np.argsort(values)[-1::-1]
    return [(keys[i], values[i]) for i in index]


def calc_all_template(data: pd.DataFrame):
    # 输入data包含 "岗位名称目录" 和 "能力关键词（同义替换）" 两列，按 "岗位名称目录" 计算能力模板
    index = data.groupby(by='岗位名称目录').groups
    length = {key: len(value) for key, value in index.items()}
    for key in length:
        if length[key] < 50:
            del index[key]

    category = ['工种', '岗系', '岗位名称目录']
    template_out = {}
    for cate in category:
        output_data = {}
        for key, value in index.items():
            ability = [data.loc[i, '能力关键词（同义替换）'] for i in value]
            std_file = f'std_factor0721_{cate}.data'
            results = calc_one_template(ability, std_file)
            output_data[key] = results
        template_out[cate] = output_data
    return template_out


if __name__ == "__main__":
    print('#############计算单个模板################')
    template = calc_one_template([['沟通能力'], ['表达能力', '沟通能力', '其他']], 'std_factor0721_岗系.data')
    print(template)

    print('#############计算Dataframe中的模板################')
    data = [['分析师', ['沟通能力', '表达能力']],
            ['分析师', ['沟通', '表达']],
            ['精算师', ['沟通能力', '表达能力']]]
    template = calc_all_template(pd.DataFrame(data=data, columns=['岗位名称目录', '能力关键词（同义替换）']))
    print(template)
