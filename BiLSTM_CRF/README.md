# JD文本中提取能力词

见`predict.py`样例

### model.py
模型的 `class` 定义，`if __name__ == '__main__` 语句下为训练模型

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

### ner_trained_model.cpt
保存训练完毕的模型参数