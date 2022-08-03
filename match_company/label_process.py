import pandas as pd
import numpy as np
import networkx as nx
from collections import defaultdict
import cpca

test_data = pd.read_csv('test_data/产品经理.csv', encoding='gbk')

all_label_words_part1 = np.load('origin_labels/all_label_words_part1.npy', allow_pickle= True)
all_label_words_part2 = np.load('origin_labels/all_label_words_part2.npy', allow_pickle= True)
word_dict = {'险':'A','社保':'A','休':'B', '假':'B','体检':'C', '补':'D','津贴':'D','奖':'E', '分红':'E', '福利':'F', '晋升':'G', '带':'H', '培训':'H',
             '指导':'H','导师':'H','餐':'I', '食':'I', '零食':'I', '下午茶':'I', '水果': 'I', '住':'J','公寓':'J','宿舍':'K', '旅游':'K', '团建':'L',
             '活动':'L','薪':'M', '经验':'N'}
#
# '''体检、吃、住、保险、奖金、休、福利'''
def words_type(data):
    label_dict = defaultdict(list)
    for i in range(len(data)):
        for j in word_dict.keys():
            if j in data[i]:
                label_dict[word_dict[j]].append(data[i])
    result = sum([i for i in label_dict.values()], [])
    data = list(set(data).difference(set(result)))
    file_name = open('origin_labels/useful_label.txt', 'w')
    for k, v in label_dict.items():
        v = ','.join(v)
        file_name.write(str(k) + v)
        file_name.write('\n')
    file_name.close()
    df = pd.DataFrame(data, columns=['label_words'])
    df.to_csv('test_data/other_label.csv', index = False)

def collect_all_words(data):
    result = []
    for i in range(len(data)):
        result = list(set(result + list(data.iloc[i])))
    # print(len(result))
    df = pd.DataFrame(result, columns=['label_words'])
    df.to_csv('origin_labels/all_label_words_part2.csv', index = False)
    result = np.array(result)
    np.save('origin_labels/all_label_words_part2.npy', result)

def revise_data(data):
    tag1, tag2 = data.loc[:,['标签']].values, data.loc[:,['公司标签']].values
    tag1 = [tag1[i][0].split(';') for i in range(len(data))]
    tag2_tmp = []
    for i in range(len(tag2)):
        if type(tag2[i][0]) is not float: tag2_tmp.append(tag2[i][0].split(';'))
        else: tag2_tmp.append([])
    tag = [list(set(tag1[i]).union(set(tag2_tmp[i]))) for i in range(len(data))]
    format_dict, format_graph = dict(), nx.Graph()
    company_dict = defaultdict(list)
    f = open('origin_labels/useful_label.txt')
    lines = f.readlines()
    f.close()
    for line in lines:
        format_dict[line[0]] = line[1:-1].split(',')
        label_type = line[0]
        for label in line[1:-1].split(','):
            format_graph.add_edge(label, label_type)
    for i in range(len(data)):
        for j in tag[i]:
            if format_graph.has_node(j):
                company_dict[i].append(j)
    label_dict = defaultdict(list)
    company_names = data['所属公司'].values
    for j in range(14):
        format_set = set(format_graph.neighbors(chr(65+j)))
        for i in range(len(data)):
            if len(set(tag[i]).intersection(format_set)) > 0:
                label_dict[i].append(set(tag[i]).intersection(format_set))
            else: label_dict[i].append(None)
    df = pd.DataFrame(label_dict).T
    df.to_csv('test_data/产品经理_label.csv')

def get_location(data):
    address1, address2 = data.loc[:, ['公司所在地']].values, data.loc[:, ['职位所在地']].values
    # print(address1)
    address_dict = defaultdict(list)
    for i in range(len(data)):
        if list(cpca.transform([address2[i][0]]).iloc[0]) != [None,None,None,None,None]:
            address_dict[i] = list(cpca.transform([address2[i][0]]).iloc[0])[:2]
        else:
            if address1[i] == "全国": address_dict[i] = ["全国", None]
            elif '北京' in address1[i][0] or '上海'in address1[i][0] or '成都'in address1[i][0] or '天津' in address1[i][0]:
                address_dict[i] = [address1[i][0][:2], address1[i][0][:2]]
            elif '/' in address1[i][0]:
                address_dict[i] = address1[i][0].split('/')
            else:address_dict[i] = [address1[i][0], None]
    df = pd.DataFrame(address_dict).T
    df.to_csv('location.csv')

if __name__ == "__main__":
    # collect_all_words(label_data_part2)
    all_label_words = set(all_label_words_part1).union(set(all_label_words_part2))
    words_type(list(all_label_words))
    revise_data(test_data)
