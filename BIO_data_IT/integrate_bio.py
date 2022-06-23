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
