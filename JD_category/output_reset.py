import pickle

with open('../weight/polymerize0630.data', 'rb') as f:
    poly = pickle.load(f)
with open('../weight/words_all.data', 'rb') as f:
    words = pickle.load(f)
poly = {key: {x.lower() for x in value} for key, value in poly.items()}
poly_words_all = [j for i in poly.values() for j in i]

for i in range(len(words)):
    for j in range(len(words[i])):
        word = words[i][j].lower()
        if word in poly_words_all:
            for key, value in poly.items():
                if word in value:
                    words[i][j] = key
                    break
    if i % 1000 == 0:
        print(i)

with open('../weight/words_reseted.data', 'wb') as f:
    pickle.dump(words, f)

