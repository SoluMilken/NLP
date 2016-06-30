# -*- coding: utf-8 -*-
import json
import jieba
import ipdb
import numpy as np
#from nlp.utils import IterTimer


jieba.set_dictionary('dict.txt.big.txt')

topic = "是否滿意北市府的處理態度"
#"是否滿意北市府的處理態度" / "是否支持華航工會罷工"
topic_output_path = \
    "/home/en/Natural_Language_Processing/dataset/" + topic + ".json"

#with IterTimer('Tokens and Stop words'):
tokens = open('nlp/symbol_to_be_removed/tokens.txt', 'rb').read().decode("utf-8") 
stop_words = open('nlp/symbol_to_be_removed/stop_words_chinese.txt', 'rb').read().decode("utf-8") 
stop_words = " ".join(stop_words.split(","))
tokens_and_stopwords = (tokens + " " + stop_words)

#with IterTimer('Preprocessing...'):
corpus = []
with open(topic_output_path) as json_file:
    json_data = json.load(json_file)
    dataset = [None] * len(json_data['data'])
    for num, comment_dic in enumerate(json_data['data']):
        paragraph = ''
        comment_cut_list = []
        comment = comment_dic['content']
        label = comment_dic['answer']

        # jieba cutting
        comment_cut_generater = jieba.cut(comment, cut_all=False)

        # filtering
        for word in comment_cut_generater:
            if word not in tokens_and_stopwords:
                comment_cut_list.append(word)
                paragraph = paragraph + ',' + word
            dataset[num] = [comment_cut_list, label]
        corpus.append(paragraph)

#with IterTimer('TFIDF'):
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()
tfidf = vectorizer.fit_transform(corpus)
print(tfidf.shape)

words = vectorizer.get_feature_names()

# for i in range(len(corpus)):
#     print('----Document %d----' % (i))
#     for j in range(len(words)):
#         if tfidf[i, j] > 0.1:
#             print(words[j], tfidf[i, j])





## training
dataset = np.array(dataset)
label = dataset[:, 1].copy()
label_list = np.unique(label)
for num, label_cat in enumerate(label_list):
    label[label == label_cat] = num

label = label.astype(np.int32, copy=False)


def evaluator(predicted_label, true_label):
    diff = predicted_label - true_label
    accuracy = len(diff[diff == 0]) / float(len(true_label))
    bias = len(predicted_label[predicted_label == 3]) / float(len(predicted_label))
    return accuracy, bias

from sklearn.linear_model import LogisticRegression
from collections import Counter
weight = Counter(label)
for cat in list(weight.keys()):
    weight[cat] = 10.0 / weight[cat]
model = LogisticRegression(C=15, multi_class='ovr', n_jobs=1, class_weight=weight)

from sklearn.cross_validation import KFold
kf = KFold(len(label), n_folds=5, shuffle=True)

from sklearn.metrics import log_loss

for train_index, test_index in kf:
    model.fit(tfidf[train_index, :], label[train_index])
    predicted_train_label = model.predict(tfidf[train_index, :])
    accuracy, bias = evaluator(predicted_train_label, label[train_index])
    print("train accuracy = {:f}, biased value = {:f}".format(accuracy, bias))

    # log_loss = log_loss(label[train_index], predicted_train_label, eps=1e-15, normalize=True, sample_weight=None)
    # print("train logloss = %f" % log_loss)

    predicted_test_label = model.predict(tfidf[test_index, :])
    accuracy, bias = evaluator(predicted_test_label, label[test_index])
    print("test  accuracy = {:f}, biased value = {:f}".format(accuracy, bias))

    # log_loss = log_loss(label[test_index], predicted_test_label, eps=1e-15, normalize=True, sample_weight=None)
    # print("test logloss = %f" % log_loss)


ipdb.set_trace()
