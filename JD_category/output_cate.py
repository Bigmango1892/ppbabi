import pandas as pd

JD = pd.read_csv('../jd_cate.csv')
category = ['行业', '岗位大类', '岗位细分']

for cate in category:
    groups = JD.groupby(by='{}_新分类'.format(cate)).groups
    groups_out = {key: list(value) for key, value in groups.items() if key[-1] != '*'}
    with open('./JD按{}分类.txt'.format(cate), 'w', encoding='utf8') as f:
        for key, value in groups_out.items():
            print(key, value, sep=':', file=f)
