from model import BiLSTM_CRF
import input_layer
import torch
import pickle


START_TAG = '<START>'
STOP_TAG = '<STOP>'

test_data, _ = input_layer.preprocess('../JD_NoNA.csv')
model_path = 'ner_trained_model_0623.cpt'
trained_ner_model = torch.load(model_path)
tag_to_ix_new = {"B-AbilityTag": 0, "I-AbilityTag": 1, "O": 2, START_TAG: 3, STOP_TAG: 4, 'B-LevelTag': 2, 'I-LevelTag': 2}
with torch.no_grad():
    words_all = []
    for i in range(len(test_data)):
        if i % 50 == 0:
            print(i)
        precheck_sent = test_data[i]
        test_text = precheck_sent.text

        test_pred = trained_ner_model(precheck_sent)
        test_pred_1 = [k for k, j in enumerate(test_pred) if j == 1]
        test_pred_0 = [k for k, j in enumerate(test_pred) if j == 0]
        words = [test_text[t] for t in sorted(test_pred_1 + test_pred_0)]
        pivot = [k for k in [sorted(test_pred_1 + test_pred_0).index(t) for t in test_pred_0]]
        if len(pivot) == 1:
            test_pred_words= [''.join(words[pivot[-1]:])]
        elif len(pivot) == 0:
            test_pred_words = []
        else:
            test_pred_words = [''.join(words[pivot[k]:pivot[k+1]]) for k in range(len(pivot)-1)]
            test_pred_words.append(''.join(words[pivot[-1]:]))

        words_all.append(test_pred_words)

    with open('../testing_data/result/all.data', 'wb') as f:
        pickle.dump(words_all, f)

