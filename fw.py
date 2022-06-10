with open('./textBOI.txt', 'r') as f:
    result = f.read().strip('\n').split('\n')
f = open('BOI.txt', 'w')
for describe in result:
    index = ''
    flag = False
    first_flag = True
    for word in describe:
        if word == '@':
            flag = not flag
            if flag:
                first_flag = True
        elif flag:
            if first_flag:
                index = index + 'B'
                first_flag = False
            else:
                index = index + 'I'
        else:
            index = index + 'O'
    print(index, file=f)
f.close()


