from tkinter import *
from pickle import load, dump

with open('words.data', 'rb') as f:
    WORDS = load(f)
with open('tags.data', 'rb') as f:
    TAGS = load(f)
HEAD = TAGS.index(0)

Color1 = 'SkyBlue'
Color2 = 'Pink'

root = Tk()
word_box = [StringVar() for _ in range(10)]
labels = [Label() for _ in range(len(word_box))]
buttons_y, buttons_n = [Button() for _ in range(len(word_box))], [Button() for _ in range(len(word_box))]


def page_up():
    global HEAD, word_box, labels
    HEAD = HEAD - len(word_box)
    if HEAD < 0:
        HEAD = 0
    for i, box in enumerate(word_box):
        box.set(WORDS[HEAD+i])
    for i in range(len(word_box)):
        if TAGS[HEAD + i] == 1:
            labels[i].configure(bg=Color2)
        elif TAGS[HEAD + i] == 2:
            labels[i].configure(bg=Color1)
        else:
            labels[i].configure(bg='white')
    with open('tags.data', 'wb') as f:
        dump(TAGS, f)


def page_down():
    global HEAD, word_box, labels
    HEAD = HEAD + len(word_box)
    if HEAD > len(WORDS) - len(word_box):
        HEAD = len(WORDS) - len(word_box)
    for i, box in enumerate(word_box):
        box.set(WORDS[HEAD + i])
    for i in range(len(word_box)):
        if TAGS[HEAD + i] == 1:
            labels[i].configure(bg=Color2)
        elif TAGS[HEAD + i] == 2:
            labels[i].configure(bg=Color1)
        else:
            labels[i].configure(bg='white')
    with open('tags.data', 'wb') as f:
        dump(TAGS, f)


def yes_button(pos: int):
    global labels, TAGS, HEAD
    labels[pos].configure(bg=Color2)
    TAGS[HEAD+pos] = 1


def no_button(pos: int):
    global labels, TAGS, HEAD
    labels[pos].configure(bg=Color1)
    TAGS[HEAD + pos] = 2


def exit_GUI():
    global root
    with open('tags.data', 'wb') as f:
        dump(TAGS, f)
    root.quit()


def create_GUI():
    global root, labels, buttons_n, buttons_y, word_box

    for i, box in enumerate(word_box):
        box.set(WORDS[HEAD+i])
    root.title('通用技能')

    # frame_head，其父窗口为root
    frame_head = Frame(root)
    frame_head.pack(side=TOP, fill=BOTH, expand=YES)

    # frame_data，父窗口为frame_data
    frame_data = Frame(frame_head)
    frame_data.grid(row=0, column=0)
    for i in range(len(word_box)):
        labels[i] = Label(frame_data, textvariable=word_box[HEAD + i], width=20)
        labels[i].grid(row=i, column=0, padx=10, pady=5)
        if TAGS[HEAD+i] == 1:
            labels[i].configure(bg=Color2)
        elif TAGS[HEAD+i] == 2:
            labels[i].configure(bg=Color1)
    for i in range(len(word_box)):
        buttons_y[i] = Button(frame_data, text='通用技能', width=15, command=lambda pos=i: yes_button(pos))
        buttons_n[i] = Button(frame_data, text='非通用技能', width=15, command=lambda pos=i: no_button(pos))
        buttons_y[i].grid(row=i, column=1, padx=10, pady=5)
        buttons_n[i].grid(row=i, column=2, padx=10, pady=5)

    # frame_page
    frame_pages = Frame(frame_head)
    frame_pages.grid(row=1, column=0)
    button_up = Button(frame_pages, text='上一页', width=15, command=page_up)
    button_up.grid(row=0, column=0, padx=50, pady=5)
    button_down = Button(frame_pages, text='下一页', width=15, command=page_down)
    button_down.grid(row=0, column=1, padx=50, pady=5)
    button_exit = Button(frame_pages, text='保存并退出', width=15, command=exit_GUI)
    button_exit.grid(row=0, column=2, padx=50, pady=5)
    root.mainloop()


if __name__ == '__main__':
    create_GUI()
