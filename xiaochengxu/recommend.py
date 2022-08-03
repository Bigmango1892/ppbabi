import pymysql
from collections import defaultdict
import numpy as np
import pandas as pd
# select同专业、同等级学校的记录
# 找到薪资高的、去向占比高的；标记的和盲盒有的暂时空着
# 顺序暂时写一个

conn = pymysql.connect(
    host='192.168.3.4',  ####mysql数据库地址可以输入本机IP,可以是localhost
    # host='localhost'
    port=3306,  ####mysql数据库端口号
    user='root',  ####mysql数据库账号
    passwd='',  #####mysql数据库密码
    database='paperball_db')  ####mysql数据库库名

def cal_salary(salary_list):
    salary_list = np.array(salary_list)
    Q = np.percentile(salary_list, (25, 50, 75), interpolation='midpoint')
    xmin = Q[0] - 1 * (Q[2] - Q[0])
    xmax = Q[2] + 1.5 * (Q[2] - Q[0])
    t = len(salary_list)
    #  >= xmin, < Q[0], < Q[2], < xmax
    part1, part2, part3, part4 = salary_list >= xmin, salary_list < Q[0], salary_list < Q[2],salary_list < xmax
    p1 = [i for i in range(len(salary_list)) if part1[i] == True and part2[i] == True]
    p2 = [i for i in range(len(salary_list)) if part2[i] == False and part3[i] == True]
    p3 = [i for i in range(len(salary_list)) if part3[i] == False and part4[i] == True]

    p1 = len(p1)/t
    p2 = len(p2)/t
    p3 = len(p3) / t

    salary = p1 * Q[0] + p2 * Q[1] + p3 * Q[2]

    return salary

# 专业好说；学校呢
def get_record(school_type,major):
    cursor = conn.cursor()
    sql1 = "select `学校名称` from `school_985211` where `标签` = '985'"
    cursor.execute(sql1)
    school_985 = [i[0] for i in cursor.fetchall()]
    sql2 = "select `学校名称` from `school_985211` where `标签` = '211'"
    cursor.execute(sql2)
    school_211 = [i[0] for i in cursor.fetchall()]

    sql3 = "select `教育经历1学校`,`教育经历1学历`,\
           `工作经历1岗位岗位大类`,`工作经历1岗位岗位细分`,`工作经历1月薪`,`工作经历2岗位岗位大类`,`工作经历2岗位岗位细分`,`工作经历2月薪`,\
          `工作经历3岗位岗位大类`,`工作经历3岗位岗位细分`,`工作经历3月薪`,`工作经历4岗位岗位大类`,`工作经历4岗位岗位细分`,`工作经历4月薪`\
           from `zhilian_data` where `教育经历1专业`  LIKE '%{}%'".format(major)
    cursor.execute(sql3)
    record = list(cursor.fetchall())

    sql4 = "select `教育经历2学校`,`教育经历2学历`,\
           `工作经历1岗位岗位大类`,`工作经历1岗位岗位细分`,`工作经历1月薪`,`工作经历2岗位岗位大类`,`工作经历2岗位岗位细分`,`工作经历2月薪`,\
          `工作经历3岗位岗位大类`,`工作经历3岗位岗位细分`,`工作经历3月薪`,`工作经历4岗位岗位大类`,`工作经历4岗位岗位细分`,`工作经历4月薪`\
            from `zhilian_data` where `教育经历2专业` LIKE '%{}%'".format(major)
    cursor.execute(sql4)
    record += list(cursor.fetchall())
    cursor.close()
    if school_type == '985/211':
        for i in record:
            if i[0] not in school_985 and i[0] not in school_211: record.remove(i)
    elif school_type == '普通本科':
        for i in record:
            if i[0] in school_211 or i[0] not in school_211 or i[1] == '大专': record.remove(i)
    else:
        for i in record:
            if i[1] != '大专': record.remove(i)
    return record

def get_recom(record):
    record_salary, record_category, record_list = defaultdict(list), {}, []
    for i in record:
        for j in range(3,14,3):
            if i[j] == None or i[j+1] == None: continue
            if i[j] == '医生助理': continue
            if i[j] not in record_category.keys():
                record_category[i[j]] = i[j-1]
            record_salary[i[j]].append(i[j+1])
    total = len(record)

    for k, v in record_salary.items():
        l = len(v)
        if l / total < 0.01 or l < 4:
            record_list.append([k,record_category[k],0,0,0])
            continue
        else:
            ave = cal_salary(v)
            record_list.append([k, record_category[k], l, 100.0*l/total, ave])
    rec_s = sorted(record_list, key = lambda x: x[4], reverse=True)
    rec_c = sorted(record_list, key = lambda x: x[2], reverse=True)

    return rec_s, rec_c

if __name__ == "__main__":
    record = get_record('普通本科', '旅游管理')
    conn.close()
    get_recom(record)



