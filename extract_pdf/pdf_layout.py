from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfparser import PDFParser
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import PDFPageAggregator
from pdfminer.layout import LAParams, LTTextBox, LTTextLine, LTFigure

from LAC import LAC
import re
from collections import OrderedDict

lac = LAC(mode='rank')
# 需要提取的栏目：个人信息（姓名、电话、邮箱、地址）、教育经历、获奖经历、项目经历、实习经历、求职意向、技能、自我评价、校园/社会实践经历
# 需要提取的栏目：个人信息（地址）、教育经历（及获奖经历）、项目经历、实习经历、求职意向、技能、自我评价、校园/社会实践经历
# 除了个人信息之外，其余的基本都会有一个标题；个人信息一般在开头

def parse_layout(layout):
    """Function to recursively parse the layout tree."""
    box_dict = {}
    for lt_obj in layout:
        print(lt_obj.__class__.__name__)
        print(lt_obj.bbox)
        if isinstance(lt_obj, LTTextBox) or isinstance(lt_obj, LTTextLine):
            print(lt_obj.get_text())
            if lt_obj.__class__.__name__ == "LTTextBoxHorizontal":
                box_dict[lt_obj.bbox] = lt_obj.get_text()
        elif isinstance(lt_obj, LTFigure):
            parse_layout(lt_obj)  # Recursive
    return box_dict

# 基本可以，特殊情况：resume5.pdf，这个要在其他函数里处理
def find_name(texts):
    for text in texts:
        rank_result = lac.run(text)
        if 'PER' in rank_result[1]:
            print(rank_result[0][rank_result[1].index('PER')])
            break

def find_mail(texts):
    for text in texts:
        t = re.findall(r"\w*@\w*.com", text)
        if len(t) > 0:
            print(t)

def find_number(texts):
    for text in texts:
        text = text.replace('-','')
        t = re.findall(r"1\d\d\d\d\d\d\d\d\d\d", text)
        if len(t) > 0:
            print(t)

def match_blocks(box_dict):
    targets = ['教育','项目','技能','评价','实习','实践']
    box_flag = {}
    for i in targets:
        for box, text in box_dict.items():
            t = re.findall("\w*"+i+"\w*", text)
            if len(t) > 0 and len(t[0]) < 10:
                box_flag[i] = box
                break
    return box_flag

    # if len(box_dict[box]) > len(t[0]):
    #     print(box_dict[box])


def big_box(box_dict, box_flag):
    targets,pos = box_flag.keys(), box_flag.values()
    all_box = box_dict.keys()
    all_box = sorted(all_box, key = lambda x:x[1], reverse = True)
    print(all_box)
    for i in ["教育",]:
        tmp = []
        init_box = box_flag[i]
        index = all_box.index(init_box)
        for box in all_box[index+1:]:
            if box in box_flag.values():
                print(1)
                break
            tmp.append(box)
    target_sentence = []
    for key, value in box_dict.items():
        if key in tmp: target_sentence.append(value)
    print(target_sentence)



if __name__ =="__main__":
    fp = open('test_data/resume1.pdf', 'rb')
    parser = PDFParser(fp)
    doc = PDFDocument(parser)

    rsrcmgr = PDFResourceManager()
    laparams = LAParams()
    device = PDFPageAggregator(rsrcmgr, laparams=laparams)
    interpreter = PDFPageInterpreter(rsrcmgr, device)
    for page in PDFPage.create_pages(doc):
        interpreter.process_page(page)
        layout = device.get_result()
        box_dict = parse_layout(layout)
    box_flag = match_blocks(box_dict)
    big_box(box_dict,box_flag)