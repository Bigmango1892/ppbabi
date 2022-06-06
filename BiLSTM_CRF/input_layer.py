import numpy as np
import pandas as pd
import jieba

# 读取所有的字符
with open('./characters.txt', 'r', encoding='utf8') as f:
    CHAR = f.read().split(sep='\n')

# 读取词性标注对应表
with open('./posseg.txt', 'r', encoding='utf8') as f:
    POSSEG = {POS.split(sep='\t')[0]: int(POS.split(sep='\t')[2].strip('\n')) for POS in f.readlines()}

# 读取词性与词性标注对应表
with open('./conpos.txt', 'r', encoding='utf8') as f:
    CONPOS = f.read()


def one_hot(describe_text: str, vectorize: bool = False):
    # 输入工作描述内容，输出每个字符的独热码。若vectorize为True则输出为一个np矩阵，每一列为describe_text中每一个字符的独热码；
    # 若vectorize为False则输出为一个list列表，每个元素代表describe_text中每一个字符的独热非0元的位置
    char_pos = [CHAR.index(char) for char in describe_text]
    if not vectorize:
        return char_pos
    oh_code = np.zeros((len(CHAR), len(describe_text)))
    for char in range(len(describe_text)):
        oh_code[char_pos[char], char] = 1
    return oh_code


def calc_seg(text_words: list):
    # 输入工作描述的jieba分词（带词性）结果，输出为每个字符的位置特征seg，其值为每个字符在所在的词语中的相对位置。如：
    # "操作系统"是分词后得到的词，则"操"位置标记为0，"作"位置标记为1，"系"位置标记为2，"统"位置标记为3
    seg_eig = []
    for word, _ in text_words:
        seg_eig.extend(range(len(word)))
    return seg_eig


def calc_pos(text_words: list):
    # 输入工作描述的jieba分词（带词性）结果，输出为每个分词的词性特征pos，其值为每个字符所在词语的词性，具体词性表可见posseg.txt，如：
    # "操作系统我学不会"分词结果为 [("操作系统", 'l'), ('我', 'r'), ('学', 'n'), ('不会', 'v')]，此时在posseg.txt中找到各词性
    # 所对应的数字编号，'l'为16，'r'为31，'n'为20，'v'为45，故pos特征为[16, 16, 16, 16, 31, 20, 45, 45]
    # 注：此函数可能会报错，因为posseg.txt所用为结巴0.39版本词性标注表，现版本为0.42不确定是否添加了新的词性分类，请及时修改posseg.txt
    pos_eig = []
    for word, flag in text_words:
        pos_eig.extend([POSSEG[flag] for _ in range(len(word))])
    return pos_eig


def calc_con(text_words: list, text_length: int):
    # 输入工作描述的jieba分词（带词性）结果，输出为每个分词的上下文特征con，具体标注见论文阐述
    # 由于jieba的词性有多种标注，conpos.txt中记录每种词性对应的jieba标注，其中：
    # 动词、名词栏列出了所有可能的jieba标注，形容词栏除了jieba标注外还需判断'的'是否出现，连接字符栏为需要判断的字符
    con_eig = [0 for _ in range(text_length)]
    i_pos, i_words = 0, 0
    while i_pos < len(text_words):
        if text_words[i_pos].flag in CONPOS['v']:
            if i_pos + 1 < len(text_words) and text_words[i_pos + 1].flag in CONPOS['n']:
                word_len = len(text_words[i_pos].word)
                con_eig[i_words: i_words + word_len] = [1 for _ in range(word_len)]
                i_words = i_words + word_len + len(text_words[i_pos + 1].word)
                i_pos = i_pos + 2
                if i_pos < len(text_words) and text_words[i_pos].flag in CONPOS['n']:
                    word_len = len(text_words[i_pos].word)
                    con_eig[i_words: i_words + word_len] = [4 for _ in range(word_len)]
                    i_words = i_words + word_len
                    i_pos = i_pos + 1
            else:
                i_words = i_words + len(text_words[i_pos].word)
                i_pos = i_pos + 1



# 若直接运行该程序，则为使用jd_sample.csv更新字符表character.txt
# 注：字符表中去除了'\n'字符
if __name__ == "__main__":
    df = pd.read_csv('./jd_sample.csv')
    characters = []
    for i in range(len(df)):
        for c in df.loc[i, '工作内容（总的）']:
            if c not in characters:
                characters.append(c)
    if '\n' in characters:
        characters.remove('\n')
    with open('./characters.txt', 'w', encoding='utf8') as f:
        print('\n'.join(characters), end='', file=f)
