import pandas as pd
import numpy as np
from collections import defaultdict
import cpca

'''生成公司介绍表'''

COM_SIZE = ['2000人以上','500-2000人','150-500人','50-150人','15-50人','少于15人']
COM_RANK = {'TOP20':1,'TOP50':2,'TOP100':3,'TOP150':4}

company_name = pd.read_csv('company_data/校招jd_公司名.csv', encoding='utf-8')
com_jiancheng = pd.read_csv('company_data/公司简称v1.csv', encoding='gbk')
com_recom_data = pd.read_csv('company_data/公司介绍表_原始数据.csv')
big_com_rank_data = pd.read_csv('company_data/公司简称-大类排行-CY表.csv', encoding = 'gbk')

def clean(word):
    word = word.replace('有限公司', '')
    word = word.replace('研究院', '')
    word = word.replace('集团', '')
    word = word.replace('控股集团', '')
    word = word.replace('有限责任公司', '')
    word = word.replace('股份有限公司', '')
    return word

def big_company(data, com_data):
    big_com_name = sum(com_data.loc[:, ['公司简称']].values.tolist(), [])
    all_com_name = sum(data.loc[:, ['所属公司']].values.tolist(), [])
    big_com_dict = defaultdict(list)
    for i in range(len(big_com_name)):
        big_com_name[i] = clean(big_com_name[i])
    for i in all_com_name:
        for j in big_com_name:
            if j in i or i in j:
                big_com_dict[i].append(j)
                big_com_dict[i] = ['/'.join(big_com_dict[i])]
    df = pd.DataFrame(big_com_dict).T
    return  df

def company_recom_chart_step1(data1, data2):
    company_names =data1.index.values.tolist()
    brief_names = data1[0].values.tolist()
    df = None
    com_dict = defaultdict(list)
    for i in range(len(company_names)):
        mask = data2['所属公司'] == company_names[i]
        pos = np.flatnonzero(mask)
        select_data = data2.iloc[pos]#.insert(loc = 1, column = '公司简称', value = brief_names[i])
        select_data.loc[:,'公司简称'] = [brief_names[i] for _ in range(len(select_data))]
        df = pd.concat([select_data,df])
    local = process_location(df)
    del df['公司所在地']
    del df['职位所在地']
    df['公司所在地_1'] = local.loc[:,'公司所在地_1'].values.tolist()
    df['公司所在地_2'] = local.loc[:, '公司所在地_2'].values.tolist()
    return df

def company_recom_chart_step2(data):
    com_quancheng = list(set(data['所属公司'].values.tolist()))
    cluster_data = None
    # for _, df in data.groupby(['公司全称']):
    for _, df in data.groupby(['所属公司']):
        if len(df) == 1: cluster_data = pd.concat([cluster_data,df])
        # 公司全称相同，规模取最高，所在行业取并集，公司性质暂时取并集，JD数量加和，所在地取并集
        else:
            field = df['公司所在行业'].values.tolist()
            field = '/'.join(list(set(sum([i.split('/') for i in field],[]))))
            xingzhi = '/'.join(list(set(df['公司性质'].dropna().values.tolist())))
            loca1 = '/'.join(list(set(df['公司所在地_1'].dropna().values.tolist())))
            loca2 = '/'.join(list(set(df['公司所在地_2'].dropna().values.tolist())))
            jd_count = sum(df['JD数量'].values)
            com_size = list(set(df['公司规模'].dropna().values.tolist()))
            com_size = COM_SIZE[min([COM_SIZE.index(i) for i in com_size])]

            record = df.iloc[0].to_frame().T
            record['公司所在行业'] = field
            record['公司性质'] = xingzhi
            record['公司所在地_1'] = loca1
            record['公司所在地_2'] = loca2
            record['公司规模'] = com_size
            record['JD数量'] = jd_count
            cluster_data = pd.concat([cluster_data, record])
    return cluster_data

# 增加一列，公司排序
def company_recom_chart_step3(data):
    com_jian = data['公司简称'].values.tolist()
    com_jian_format, com_rank = big_com_rank_data['公司简称'].values.tolist(), big_com_rank_data['行业大类公司排名'].values.tolist()
    my_rank = []
    for i in com_jian:
        # 无记录的视为3
        if i not in com_jian_format: my_rank.append(4)
        else: my_rank.append(COM_RANK[com_rank[com_jian_format.index(i)]])
    data['公司排序'] = my_rank
    data.columns = ['公司全称','公司性质','公司规模','公司所在行业','JD数量','公司简称','公司所在地_1','公司所在地_2','公司排序']
    column_order = ['公司简称','公司全称','公司性质','公司规模','公司所在行业','JD数量','公司所在地_1','公司所在地_2','公司排序']
    return data

def process_location(data):
    address1, address2 = data.loc[:, ['公司所在地']].values, data.loc[:, ['职位所在地']].values
    address_dict = defaultdict(list)
    for i in range(len(data)):
        if type(address1[i][0]) == float:
            address_dict[i] = list(cpca.transform([address2[i][0]]).iloc[0])[:2]
        elif address1[i][0] == "全国": address_dict[i] = ["全国", None]
        elif '北京' in address1[i][0] or '上海' in address1[i][0] or '成都' in address1[i][0] or '天津' in address1[i][0]:
            address_dict[i] = [address1[i][0][:2], address1[i][0][:2]]
        elif '/' in address1[i][0]:
            address_dict[i] = address1[i][0].split('/')
        elif None not in list(cpca.transform([address1[i][0]]).iloc[0])[:2]:
            address_dict[i] = list(cpca.transform([address2[i][0]]).iloc[0])[:2]
        else:address_dict[i] = [address1[i][0],list(cpca.transform([address2[i][0]]).iloc[0])[1]]
    df = pd.DataFrame(address_dict).T
    df.columns = ['公司所在地_1','公司所在地_2']
    return df

# 就不聚合了
def small_company(data_all, data_big):
    big_com_names = data_big['公司全称'].values.tolist()
    all_com_names = data_all['所属公司'].values.tolist()
    small_com_names = list(set(all_com_names).difference(set(big_com_names)))
    small_data = None
    for i in small_com_names:
        records = data_all[data_all['所属公司'] == i]
        small_data = pd.concat([small_data, records])
    local = process_location(small_data)
    del small_data['公司所在地']
    del small_data['职位所在地']
    small_data['公司所在地_1'] = local.loc[:, '公司所在地_1'].values.tolist()
    small_data['公司所在地_2'] = local.loc[:, '公司所在地_2'].values.tolist()
    small_data.columns = ['公司全称', '公司性质', '公司规模', '公司所在行业', 'JD数量', '公司所在地_1', '公司所在地_2']
    print(len(small_data))

    small_rank = []
    cluster_data = None
    for _, df in small_data.groupby(['公司全称']):
        if len(df) == 1:
            cluster_data = pd.concat([cluster_data, df])
            small_rank.append(5+COM_SIZE.index(df['公司规模'].values[0]))
        # 公司全称相同，规模取最高，所在行业取并集，公司性质暂时取并集，JD数量加和，所在地取并集
        else:
            field = df['公司所在行业'].values.tolist()
            field = '/'.join(list(set(sum([i.split('/') for i in field], []))))
            xingzhi = '/'.join(list(set(df['公司性质'].dropna().values.tolist())))
            loca1 = '/'.join(list(set(df['公司所在地_1'].dropna().values.tolist())))
            loca2 = '/'.join(list(set(df['公司所在地_2'].dropna().values.tolist())))
            jd_count = sum(df['JD数量'].values)
            com_size = list(set(df['公司规模'].dropna().values.tolist()))
            com_size = [i.replace('—','-') for i in com_size]
            if len(com_size) == 0:
                com_size = None
                small_rank.append(10)
            else:
                com_size = COM_SIZE[min([COM_SIZE.index(i) for i in com_size])]
                small_rank.append(5 + COM_SIZE.index(com_size))


            record = df.iloc[0].to_frame().T
            record['公司所在行业'] = field
            record['公司性质'] = xingzhi
            record['公司所在地_1'] = loca1
            record['公司所在地_2'] = loca2
            record['公司规模'] = com_size
            record['JD数量'] = jd_count

            cluster_data = pd.concat([cluster_data, record])
        cluster_data['公司排序'] = small_rank
    com_jian = [None for _ in range(len(cluster_data))]
    cluster_data.insert(loc=0, column='公司简称', value=com_jian)
    cluster_data.to_csv('company_data/小公司介绍表.csv', index = False)

def merge_small_big():
    s_data = pd.read_csv('company_data/小公司介绍表.csv')
    b_data = pd.read_csv('company_data/大公司介绍表.csv')
    all = pd.concat([b_data,s_data])
    all.to_csv('company_data/all_公司介绍表.csv',index = False)

if __name__ =="__main__":
    com_duiying = big_company(company_name, com_jiancheng)

    # 这三步是给大公司的
    step1 = company_recom_chart_step1(com_duiying, com_recom_data)
    step2 = company_recom_chart_step2(step1)
    big_com = company_recom_chart_step3(step2)

    # 这是小公司的
    small_company(com_recom_data, big_com)

    merge_small_big()