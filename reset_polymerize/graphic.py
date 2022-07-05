import pickle
from tkinter import *
import weight.calc_weight as reset_std
import JD_category.output_reset as reset_words


with open('../weight/polymerize0630.data', 'rb') as f:
    poly = pickle.load(f)

root = Tk()
entry = Entry()

def change_poly():
    global poly
    reset_words.poly = {key: {x.lower() for x in value} for key, value in poly.items()}
    reset_words.poly_words_all = [j for i in reset_words.poly.values() for j in i]
    reset_words.calc_words()
    reset_std.calc_std()


def createGUI():
    global root, entry

    root.title('重置同类词匹配')

    frame_head = Frame(root)
    frame_head.grid(row=0, column=0)

    frame_entry = Frame(frame_head)
    frame_entry.grid(row=0, column=0)
    entry = Entry(frame_entry, width=15)
    entry.grid(row=0, column=0)
    label1 = Label(frame_entry, width=15)

    root.mainloop()


if __name__ == "__main__":
    createGUI()
