import numpy as np
from model import *

model_path = 'ner_trained_model.cpt'
trained_ner_model = torch.load(model_path)


def predict(text):
    if not isinstance(text, str):
        return np.nan
    with torch.no_grad():
        text_feature = input_layer.preprocess(text, is_string=True)
        test_text = text_feature.text
        test_pred = trained_ner_model(text_feature)
        test_pred_1 = [k for k, j in enumerate(test_pred) if j == 1]
        test_pred_0 = [k for k, j in enumerate(test_pred) if j == 0]
        words = [test_text[t] for t in sorted(test_pred_1 + test_pred_0)]
        pivot = [k for k in [sorted(test_pred_1 + test_pred_0).index(t) for t in test_pred_0]]
        if len(pivot) == 1:
            test_pred_words = [''.join(words[pivot[-1]:])]
        elif len(pivot) == 0:
            test_pred_words = []
        else:
            test_pred_words = [''.join(words[pivot[k]:pivot[k + 1]]) for k in range(len(pivot) - 1)]
            test_pred_words.append(''.join(words[pivot[-1]:]))
    return test_pred_words


if __name__ == '__main__':
    print(predict('【岗位要求】1. 熟悉Java编程语言; 2. 熟悉Python'))
