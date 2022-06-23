import numpy as np


def find_tags(jd_text: str, bio: list):
    i = 0
    tags = []
    while i != len(jd_text):
        if bio[i][0] == 'B':
            j = 1
            tag = bio[i][2:]
            while i + j < len(jd_text) and bio[i + j][0] == 'I' and bio[i + j][2:] == tag:
                j = j + 1
            tags.append((jd_text[i:i+j], tag))
            i = i + j
        elif bio[i][0] == 'I':
            pass
        else:
            i = i + 1
    return tags


def find_ability(tags_all: list):
    tags = []
    for word, tag in tags_all:
        if tag == 'AbilityTag':
            tags.append(word)
    return tags


with open('../BIO_data/data_jd.txt', 'r') as f:
    jd_texts = f.read().strip('\n').split('\n')
with open('../BIO_data/data_bio.txt', 'r') as f:
    bio_texts = f.read().strip('\n').split('\n')

if __name__ == '__main__':
    count_index = []
    for jd, bio in zip(jd_texts, bio_texts):
        tags = find_tags(jd, bio.split(' '))
        tags = find_ability(tags)
        item = []
        for i, word in enumerate(tags):
            word = word.replace('编程', '').replace('能力', '').lower()
            if len(tags) == 1:
                item.append((word, 0))
            else:
                item.append((word, i/(len(tags)-1)))
        count_index.append(item)

    count = {x: len(l) for x, l in count_index.items()}
    mean_value = {key: np.mean(value) for key, value in count_index.items()}
    results = {key: count[key] * (1 - mean_value[key]) for key in count.keys()}
    index = np.argsort(list(results.values()))[-1::-1]
    for i in range(20):
        print(list(results.keys())[index[i]], list(results.values())[index[i]])
    print()
