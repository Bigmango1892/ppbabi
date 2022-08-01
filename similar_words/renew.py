import pandas as pd
import pickle

df = pd.read_csv('./修改词对照表.csv', encoding='gbk')
with open('./polymerize_lower.pkl', 'rb') as f:
    poly = pickle.load(f)

for i in range(len(df)):
    for j in range(8):
        if pd.notna(df.loc[i, '原词{}'.format(j+1)]):
            original = df.loc[i, '原词{}'.format(j+1)].split('/')
            changed_set = set()
            for word in original:
                if word in poly:
                    changed_set = changed_set.union(poly[word])
                    del poly[word]
                else:
                    changed_set.add(word.lower())
            if df.loc[i, '模板词{}'.format(j+1)] in poly:
                changed_set = changed_set.union(poly[df.loc[i, '模板词{}'.format(j+1)]])
            else:
                changed_set.add(df.loc[i, '模板词{}'.format(j+1)].lower())
            poly[df.loc[i, '模板词{}'.format(j+1)]] = changed_set
with open('polymerize_lower.pkl', 'wb') as f:
    pickle.dump(poly, f)
