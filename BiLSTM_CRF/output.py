import numpy as np
import pandas as pd
from model import BiLSTM_CRF
import input_layer
import torch


def predict(text):
    if not isinstance(text, str):
        return np.nan
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


START_TAG = '<START>'
STOP_TAG = '<STOP>'
tag_to_ix_new = {"B-AbilityTag": 0, "I-AbilityTag": 1, "O": 2, START_TAG: 3, STOP_TAG: 4, 'B-LevelTag': 2, 'I-LevelTag': 2}
model_path = 'ner_trained_model_0623.cpt'
trained_ner_model = torch.load(model_path)
with torch.no_grad():
    test_data = pd.read_csv('../jd.csv')
    key_words = test_data['工作内容（总的）'].apply(predict)
    test_data['能力关键词'] = key_words.apply(lambda x: ';'.join(x) if isinstance(x, list) else x)
    test_data.to_csv('../jd_with_ability.csv', index=False)
