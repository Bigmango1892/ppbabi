# 能力分词分析
用于实现文本中能力词的提取，同义词替换，岗位能力模板输出以及相似能力计算。其中：

能力词提取实现的必要组件为 `./BiLSTM-CRF/` 目录下全部内容；

同义词替换的必要组件为 `./similar_words/` 目录下的全部内容；

岗位能力模板的必要组件为 `./capability_template/` 目录下的全部内容；

相似能力计算的必要组件为 `./ability_similarity/` 目录下的全部内容。

## JD文本中提取能力词
通过 *BiLSTM-CRF* 模型实现分词，引入目录下论文中提到的一些文本特征值，使用标注数据进行训练，已训练好的模型参数保存于 `./BiLSTM_CRF/ner_trained_model.cpt`
下， 训练的数据集存于 `./BIO_data` 目录下。

### Quick Start
引入 `model` 后加载训练好的模型 `ner_trained_model.cpt` 即可进行模型计算，具体见`predict.py`样例。

### model.py
模型的 `class` 定义，`if __name__ == '__main__` 语句下为训练模型。

### input_layer.py
输入层，见论文，需要使用的参数存于 `data` 目录下。输出为类

```python
class TextFeature:
    # 类包含呢6个属性，其中text记录原文本，words记录分词结果，后四个为论文中指出的文本特征值
    def __init__(self, text, words=None):
        # 输入 text 为文本，请勿包含换行符'\n'以及空格' '
        # 若已进行分词，则携带 words 参数传入；否则对实例会调用jieba库进行分词
        self.text = text
        self.words = ...
        self.onehot = ...
        self.seg = ...
        self.con = ...
        self.pos = ...
```

## 同义词替换
`polymerize_lower.pkl` 下以字典形式存储了替换词表，其中字典的键值为替换后的词，字典的值为集合，包含所有对应的待替换的表述，如
```python
poly = {
    'TCP': {'tcpip', 'tcp', '\\u2005tcp', '网络tcp', 'tcp协议'},
    'CFA证书': {'cfa持证', 'cfa资格', 'cfa证书', 'cfa认证'},
    ...
 }
```
其中，所有待替换的表述均为小写后的表述，寻找对应词时请务必将待匹配的词也转化为小写进行匹配。具体可见 `synonym_substitution.py` 中样例。

此外，可见 `renew.py` 样例对 `polymerize_lower.pkl` 进行更新，`similiar_words.py` 为得到原始版本 `polymerize.pkl` 的算法。


## 岗位能力模板
用以计算岗位能力模板。通过统计同一类岗位中不同技能词出现的词频、位置信息以及其在整个岗位分类目录中出现词频的标准差三个量，计算每个词在同一类岗位中的特征值，进而排序输出岗位能力模板。

其中，`std_factor` 目录下包含计算标准差量的相关内容，`result` 目录下保留了全部测试结果。`output.py` 为读取 `.csv` 文件后直接输出岗位能力模板，`position_capability_template.py` 下为输出单个模板的相关函数。

### Quick Start
直接运行 `output.py` 即可读取 `按【要求】提取JD_能力词.csv` 中内容进行技能模板输出。其调用了 `position_capability_template.py` 中的 `calc_all_template()` 函数，也可直接参考
`position_capability_template.py` 中的样例。

### position_capability_template.py
`calc_one_template` 函数输入为一个分类下的全部JD能力词，输出该分类的技能模板。如：
```python
ability_words = [
    ['沟通能力'], # 这是一条JD的技能词
    ['表达能力', '沟通能力', '其他']  # 这是另一条JD的技能词
]

template = calc_one_template(ability_words, 'std_factor0721_岗系.data')
```
其中，第二个参数为预处理完的标准差数值文件，存于 `std_factor/data/` 目录下，详见 **std_factor** 模块。
`calc_one_template` 中分三大块计算内容，见代码注释。

`calc_all_template` 函数输入为包含 "岗位名称目录" 和 "能力关键词（同义替换）" 两列的 Dataframe，输出为按 "岗位名称目录"
分类后计算的能力模板。例如：

```python
data = [
    # 这是一条JD
    {'岗位名称目录': '分析师', '能力关键词（同义替换）': ['沟通能力', '表达能力']},
    # 这是另一条JD
    {'岗位名称目录': '分析师', '能力关键词（同义替换）': ['沟通', '表达']},
    # 这还是另一条JD
    {'岗位名称目录': '精算师', '能力关键词（同义替换）': ['沟通能力', '表达能力']}
]
# 分别聚合计算 '分析师'，'精算师' 的能力模板
template = calc_all_template(pd.DataFrame(data))  # import pandas as pd
```

### 计算方案
详见 `position_capability_template.py` 中的 `calc_one_template` 函数，其中对每条JD中的能力词，从0到1递增标注其位置信息；

### std_factor


## 相似能力计算

## pdf提取：extract_pdf
## pdf_layout.py
* 从pdf格式简历中提取信息，流程：
    * 对pdf简历进行分区(box)
    * 识别姓名（LAC识别）、电话（正则）、邮箱（正则）
    * 除去以上姓名、电话、邮箱的box后，按块合并剩余box（目前仅对“教育经历”进行测试）
    * 提取能力关键词（可以用提取JD能力词的模型，但效果有优化空间）
* [参考资料](https://tianchi.aliyun.com/notebook-ai/detail?spm=5176.12586969.1002.6.3d347fd7Cn1wa8&postId=112407)
