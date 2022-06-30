import pickle
import numpy as np

if __name__ == '__main__':
    data = []
    for i in range(16):
        data.extend(np.load('/Users/dingyouqian/Desktop/result/result{}.npy'.format(i+1), allow_pickle=True))
    for i in range(len(data)):
        if len(data[i]) > 15:
            data[i] = {}
        elif len(data[i]) > 15:
            print(data[i])
            flag = input()
            if flag == 'd':
                data[i] = {}
    for _ in range(data.count({})):
        data.remove({})

    data_integrate = []
    for _set in data:
        flag = True
        for i in range(len(data_integrate)):
            if not data_integrate[i].isdisjoint(_set):
                data_integrate[i] = data_integrate[i].union(_set)
                flag = False
                break
        if flag:
            data_integrate.append(_set)

    data_dict = {}
    for key_set in data_integrate:
        key = key_set.pop()
        key_set.add(key)
        data_dict[key] = key_set

    with open('./polymerize.data', 'wb') as f:
        pickle.dump(data_dict, f)

