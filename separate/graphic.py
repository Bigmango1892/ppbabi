from tkinter import *
from pickle import load, dump

with open('words.data', 'rb') as f:
    WORDS = load(f)
with open('tags.data', 'rb') as f:
    TAGS = load(f)
HEAD = TAGS.index(0)


def page_up():
    pass


def page_down():
    pass


def yes_button(pos: int):
    # global labels, TAGS, HEAD
    # labels[pos].configure(bg='red')
    # TAGS[HEAD+pos] = 1
    print(pos)


def no_button(i: int):
    pass


root = Tk()
word_box = [StringVar() for _ in range(10)]
labels = [Label() for _ in range(len(word_box))]
buttons_y, buttons_n = [Button() for _ in range(len(word_box))], [Button() for _ in range(len(word_box))]

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
for i in range(len(word_box)):
    buttons_y[i] = Button(frame_data, text='通用技能', width=15, command=lambda: yes_button(i))
    buttons_n[i] = Button(frame_data, text='非通用技能', width=15, command=lambda: no_button(i))
    buttons_y[i].grid(row=i, column=1, padx=10, pady=5)
    buttons_n[i].grid(row=i, column=2, padx=10, pady=5)

# frame_page
frame_pages = Frame(frame_head)
frame_pages.grid(row=1, column=0)
button_up = Button(frame_pages, text='上一页', width=15, command=page_up)
button_up.grid(row=0, column=0, padx=50, pady=5)
button_down = Button(frame_pages, text='下一页', width=15, command=page_down)
button_down.grid(row=0, column=1, padx=50, pady=5)
root.mainloop()
