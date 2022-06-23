import os


def getFlist(path):
    for root, dirs, files in os.walk(path):
        return files


files = getFlist('./')
file_bio = open('data_bio.txt', 'w')
file_jd = open('data_jd.txt', 'w')
sep = '\n' + ''.join(['\n= O' for _ in range(44)]) + '\n\n'

for file in files:
    if file[-4:] == '.bio':
        with open(file, 'r') as f:
            text_list = f.read().split(sep=sep)[:-1]
        print(len(text_list))
        if len(text_list) != 50:
            del_word_col = []
            for text in text_list:
                if text[0] == '\n':
                    del_word_col.append(text_list.index(text))
            for col in del_word_col:
                text_list[col] = text_list[col][178:]
                print([text_list[col]])
                print('')

        for text in text_list:
            text_cuts = [[x[0], x[2:]] for x in text.split('\n')]
            describe = ''.join([x[0] for x in text_cuts])
            tags = [x[1] for x in text_cuts]
            if len(tags) != len(describe):
                print('{} ERROR'.format(file))
            tags = ' '.join(tags)

            print(describe, file=file_jd)
            print(tags, file=file_bio)

file_jd.close()
file_bio.close()
