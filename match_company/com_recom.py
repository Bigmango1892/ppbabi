import pandas as pd
import pymysql

'''1. 输入：个人信息、需求的岗位
2. 查找匹配：有这个岗位的公司（连sql查）；公司排序
'''
company_data = pd.read_csv('company_data/all_公司介绍表.csv')

# 输入一个三级目录的岗位
# 返回一个有排序的公司列表
def job_com(target):
    conn = pymysql.connect(
        host="192.168.3.4",
        user ="root",
        password ="",
        database ="paperball_db",
   )
    cursor = conn.cursor()
    company = []
    # for i in job_stack:
    sql = "select distinct `所属公司` from `shixiseng_school_recruitment_jd` where `岗位名称目录` = '%s'" % (target)
    cursor.execute(sql)
    company.append(list(j[0] for j in cursor.fetchall()))
    cursor.close()
    conn.close()
    company = list(set(sum(company,[])))
    print(len(company))
    all_company, all_rank = company_data['公司全称'].values.tolist(),company_data['公司排序'].values.tolist()
    my_rank = []
    for i in company:
        my_rank.append(all_rank[all_company.index(i)])
    sorted_com = sorted(company, key=lambda x: my_rank[company.index(x)])
    return sorted_com

if __name__ == "__main__":
    my_com = job_com('活动运营')
    print(my_com)
