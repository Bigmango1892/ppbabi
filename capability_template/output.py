import pandas as pd
from position_capability_template import calc_all_template
import time


def output_template():
    df = pd.read_csv('../按【要求】提取JD_能力词.csv')
    df = df.dropna()
    df['能力关键词（同义替换）'] = df['能力关键词（同义替换）'].apply(lambda x: x.split(';'))
    template = calc_all_template(df)
    date = time.strftime("%m%d", time.localtime(time.time()))
    for cate, sub_temp in template.items():
        table = []
        for job, abilities in sub_temp.items():
            table.append([job])
            for i in range(10):
                if i == len(abilities):
                    break
                table[-1].extend(abilities[i])
        columns = ['岗位名称'] + [j for i in range(10) for j in [f'能力词{i+1}', f'特征值{i+1}']]
        out_df = pd.DataFrame(table, columns=columns)
        out_df.to_csv(f'./result/{date}岗位技能词_按{cate}.csv', index=False, encoding='utf8')


if __name__ == '__main__':
    output_template()
