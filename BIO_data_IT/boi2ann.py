import os


def getFlist(path):
    for root, dirs, files in os.walk(path):
        return files


files = getFlist('./')
sep = '\n' + ''.join(['\n= O' for _ in range(44)]) + '\n\n'
for file in files:
    if file[-4:] == '.bio':
        with open(file, 'r') as f:
            text_list = f.read().split(sep=sep)[:-1]
        Flag = True
        write_text = []

        for text in text_list:
            text_cuts = [[x[0], x[2:]] for x in text.split('\n')]
            describe = ''.join([x[0] for x in text_cuts])
            tags = [x[1] for x in text_cuts]
            if len(tags) != len(describe):
                print('{} ERROR'.format(file))
            for i in range(len(tags)):
                if tags[i][0] == 'I':
                    if tags[i-1][0] == 'O':
                        print('ERROR:\n\tFile = {},\n\tIndex = {},\n\tTags = {},\nNO B TAG BEFORE THIS I TAG\n'
                              .format(file, text_list.index(text), i))
                        Flag = False
                    elif tags[i-1][1:] != tags[i][1:]:
                        print('ERROR:\n\tFile = {},\n\tIndex = {},\n\tTags = {},\nCOMPLEX TAGS\n'
                              .format(file, text_list.index(text), i))
                        Flag = False

                    if not Flag:
                        d = input('auto fix?(y/n)')
                        while d != 'y' and d != 'n':
                            d = input('error, retype: auto fix?(y/n)')
                        if d == 'y':
                            tags[i] = 'B' + tags[i][1:]
                        Flag = True

            if Flag:
                text_out = ''
                for i in range(len(describe)):
                    if tags[i] == 'O':
                        text_out = text_out + describe[i]
                    elif tags[i][0] == 'B':
                        text_out = text_out + '[@' + describe[i]
                        if i == len(describe) - 1 or tags[i+1][0] != 'I':
                            text_out = text_out + '#' + tags[i][2:] + '*]'
                    else:
                        text_out = text_out + describe[i]
                        if i == len(describe) - 1 or tags[i+1][0] != 'I':
                            text_out = text_out + '#' + tags[i][2:] + '*]'
                write_text.append(text_out)

        with open('./ann_files/' + file[:-4] + '.ann', 'w') as f:
            print('\n============================================\n'.join(write_text), file=f)
