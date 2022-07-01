import pickle
import weight.analy as analyse
import pandas as pd

with open('JD岗位分类.txt', 'r') as f:
    index = {x.split(':')[0]: [int(y) for y in x.split(':')[1].strip('[]').split(', ')] for x in f.read().split('\n')}
with open('../weight/words_reseted.data', 'rb') as f:
    words = pickle.load(f)

output_data = []
for key, value in index.items():
    analyse.key_word = key
    ability = [words[i] for i in value if i < len(words)]
    analyse.abilities = ability
    results, ind = analyse.count_index()
    output_data.append([key] + [list(results.keys())[ind[i]] for i in range(10) if i < len(ind)])

output_data = pd.DataFrame(data=output_data)
output_data.to_csv('岗位技能词.csv', index=False)
