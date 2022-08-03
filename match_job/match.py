import pandas as pd
import numpy as np
import os
import pickle

def get_related(user):
    with open('format_weight/dict.txt',encoding='utf-8') as file:
        content=file.readlines()
    content = [i.strip('\n') for i in content]
    content_list = []
    for i in range(len(content)):
        content_list.append(content[i][len(str(i))+2:])

    f = open('format_weight/coo.pkl', 'rb')
    d = pickle.load(f)
    relate_dict = {}
    for i in user:
        if i not in content_list:
            continue
        else:
            num = content_list.index(i)
        r = np.where(d.row == num)
        c = d.col[np.where(d.row == num)]
        w = d.data[np.where(d.row == num)]
        w = w / max(w)
        for j in range(len(c)):
            abi = content_list[c[j]]
            if abi in relate_dict.keys():
                relate_dict[abi] = max(relate_dict[abi], w[j])
            else:
                relate_dict[abi] = w[j]
    return relate_dict

def match_number(user:list, job_data, hard_skill):
    typedict = {'1': 10, '2': 10, '3': 0.8, '4': 0.5, '5': 0.5, '6': 0.3, '7': 0.3}
    count = {}
    user = set(user)
    relate_abi = get_related(user)


    for i in range(len(job_data)):
        record = job_data.iloc[i].dropna().values
        record_hs = hard_skill.iloc[i].dropna().values.tolist()[1:]
        if len(record_hs) < 1:
            continue

        l = int((len(record) -1) / 3)
        type,wei, req = record[l+1:2*l+1],record[2*l + 1:],record[1:l+1]
        type = [str(int(j)) for j in type]
        for j in range(len(req)):
            if req[j] in record_hs:
                type[j] = '1'

        wei = [float(j) for j in wei]
        s = sum(wei)
        wei = [j/s for j in wei]
        score = 0

        if len(user.union(set(relate_abi.keys())).intersection(set(record_hs))) == 0:
        # if len(user.intersection(set(record_hs))) == 0:
            continue
        for j in range(l):
            skill = req[j]
            if skill in user:
                score += typedict[str(int(type[j]))] * wei[j]
            elif skill in relate_abi:
                score += typedict[str(int(type[j]))] * wei[j] * relate_abi[skill] * 0.5
            else:
                score -= typedict[str(int(type[j]))] * wei[j]
            count[record[0]] = score
    count_sort = sorted(count.items(), key=lambda a: a[1], reverse=True)
    count_top = [i[0] for i in count_sort[:10]]
    return count_top

def get_type(full_record, record):
    typelist = []
    for j in record[1:]:
        index = full_record.index(j)
        if 0 < index < 7:
            typelist.append('1')  # 工具性技能
        elif 6 < index < 13:
            typelist.append('2')  # 知识/思维
        elif 12 < index < 16:
            typelist.append('3')  # 语言
        elif 15 < index < 19:
            typelist.append('4')  # 资质经验
        elif 18 < index < 25:
            typelist.append('5')  # 通用技能
        elif 24 < index < 30:
            typelist.append('6')  # 思想品德
        else:
            typelist.append('7')
    return typelist

def get_weight(record, w, p):
    weightlist, tmp_word, tmp_we = [],[],[]
    for i in range(len(w)):
        tmp_word.append([w[i][k] for k in range(1,21,2)])
        tmp_we.append([w[i][k] for k in range(2, 22, 2)])
    all_words = set(sum(tmp_word,[]))
    for j in record[1:]:
        weight_total = []
        if j in all_words:
            for k in range(len(w)):
                tmp = tmp_word[k]
                if j in tmp: weight_total.append(tmp_we[k][tmp.index(j)])
        else:
            best = p[p.index(j)+1].split('/')
            for k in range(len(w)):
                tmp = tmp_word[k]
                for i in best:
                    if i in tmp: weight_total.append(tmp_we[k][tmp.index(i)])
        weightlist.append(sum(weight_total)/len(weight_total))
    return weightlist

def clean_words():
    format_data = pd.read_csv('format_weight/能力模板.csv', encoding='gbk')
    weight = pd.read_csv('format_weight/原表.csv', encoding='gbk')

    missing = []
    for i in range(len(format_data)):
        tmp_word = []
        record = format_data.iloc[i].dropna().values.tolist()
        name = record[0]
        w = weight.loc[weight['岗位名称'] == name].values.tolist()
        for j in range(len(w)):
            tmp_word.append([w[j][k] for k in range(1, 21, 2)])
        all_word = set(sum(tmp_word,[]))
        tmp_missing = set(record[1:]).difference(all_word)
        missing.append(list(tmp_missing))
    return missing

def get_table():
    format_dict = {}
    format_data = pd.read_csv('format_weight/能力模板.csv', encoding='gbk')
    weight = pd.read_csv('format_weight/原表.csv', encoding='gbk')
    pair = pd.read_csv('format_weight/修改词对照表.csv',encoding='gbk')
    for i in range(len(format_data)):
        full_record = format_data.iloc[i].values.tolist()
        record = format_data.iloc[i].dropna().values.tolist()
        name = record[0]
        w = weight.loc[weight['岗位名称'] == name].values.tolist()
        p = pair.loc[pair['岗位名称']== name].values.tolist()[0]
        typelist = get_type(full_record, record)
        weightlist = get_weight(record, w,p)
        format_dict[name] = record[1:] + typelist + weightlist
    length = max([len(i) for i in format_dict.values()])
    for k, v in format_dict.items():
        if len(v) < length:
            v += [None for _ in range(length-len(v))]
    my_df = pd.DataFrame.from_dict(format_dict, orient = 'index')
    my_df.to_csv('format_weight/权重总表.csv')

if __name__ == '__main__':
    t3 = ['产品需求','互联网产品','组织活动','团队意识','沟通表达','学习能力','创意设计']
    t4 = ['Mysql','EXCEL','office办公软件','数学基础','文献检索','数据分析','SQL','MatLab']
    t5 = ['外语水平','语言沟通能力','学习能力','敏捷的洞察','用户服务意识']
    t6 = ['责任心','三观正','积极向上','亲和力','思想成熟']
    t7 = ['sketch','互联网产品','美术功底','审美力','想象力']
    mydata = pd.read_csv('format_weight/权重总表.csv')
    hard_skill = pd.read_csv('format_weight/能力模板+能力字典 - 岗位核心能力词.csv', encoding='gbk')
    print(match_number(t3, mydata, hard_skill))

