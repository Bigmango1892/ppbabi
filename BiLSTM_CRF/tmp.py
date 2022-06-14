with open('/Users/dingyouqian/Desktop/JD12.txt.bio', 'r') as f:
    text_list = f.read().split('\n')

text_out = ''
for word_pos in range(len(text_list)):
    word = text_list[word_pos]
    if word == '':
        text_out = text_out + '\n'
        continue
    if word[2] == 'O':
        text_out = text_out + word[0]
    elif word[2] == 'B':
        text_out = text_out + '[@' + word[0]
        tag = word[4:]
        if text_list[word_pos + 1] == '' or text_list[word_pos + 1][2] != 'I':
            text_out = text_out + '#' + tag + '*]'
    elif word[2] == 'I':
        text_out = text_out + word[0]
        if text_list[word_pos + 1] == '' or text_list[word_pos + 1][2] != 'I':
            text_out = text_out + '#' + tag + '*]'

with open('/Users/dingyouqian/Desktop/JD12.txt.ann', 'w') as f:
    print(text_out, file=f)
