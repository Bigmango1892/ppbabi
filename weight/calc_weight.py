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


with open('../BIO_data/data_jd.txt', 'r') as f:
    jd_texts = f.read().strip('\n').split('\n')
with open('../BIO_data/data_bio.txt', 'r') as f:
    bio_texts = f.read().strip('\n').split('\n')

if __name__ == '__main__':
    count_index = {}
    for i in range(len(jd_texts)):
        jd = jd_texts[i]
        bio = bio_texts[i].split(' ')
        tags = find_tags(jd, bio)
        # weight, alpha, flag, beta = 1, 0.96, False, 1.3
        num_total = -1
        for word, tag in tags:
            if tag == 'AbilityTag':
                num_total = num_total + 1
        if num_total == 0:
            continue
        num = -1
        for word, tag in tags:
            # if tag == 'LevelTag':
            #     flag = True
            #     continue
            # if flag:
            #     count[word] = count.get(word, 0) + weight * beta
            #     flag = False
            # else:
            #     count[word] = count.get(word, 0) + weight
            # weight = weight * alpha

            # if tag == 'AbilityTag':
            #     count[word] = count.get(word, 0) + 1

            if tag == 'AbilityTag':
                num = num + 1
                word = word.replace('编程', '').replace('能力', '').upper()
                if word not in count_index:
                    count_index[word] = []
                count_index[word].append(num/num_total)
                # count[word] = count.get(word, 0) + weight
                # weight = weight * alpha

    count = {x: len(l) for x, l in count_index.items()}
    mean_value = {key: np.mean(value) for key, value in count_index.items()}
    results = {key: count[key] * (1 - mean_value[key]) for key in count.keys()}
    index = np.argsort(list(results.values()))[-1::-1]
    for i in range(20):
        print(list(results.keys())[index[i]], list(results.values())[index[i]])
    print()
