import pickle
import os

with open(os.path.dirname(os.path.realpath(__file__)) + '/polymerize_lower.pkl', 'rb') as f:
    poly = pickle.load(f)
poly_set = set([j for i in poly.values() for j in i])


def substitute_words(words):
    # words 形如 "xxx;yyy;zzz"
    if not isinstance(words, str):
        return words
    words_list = words.split(';')
    substitution = []
    for word in words_list:
        word_lower = word.lower()
        if word_lower not in poly_set:
            substitution.append(word)
            continue
        for key, value in poly.items():
            if word_lower in value:
                substitution.append(key)
                break
    return ';'.join(substitution)


if __name__ == '__main__':
    print(substitute_words('交流能力;C++编程基础;Python能力'))
