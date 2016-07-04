# -*- coding: utf-8 -*-
import json
import jieba
import ipdb
import numpy as np
#import itertool
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

# filename = topic + 'comment_corpus'
# with open(filename, ):


#with IterTimer('TFIDF'):
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()
tfidf = vectorizer.fit_transform(corpus)
index = vectorizer.vocabulary_
idf = vectorizer.idf_
vocab = list(map(lambda k: (k, index[k], idf[index[k]]), index))
vocab.sort(key=lambda x: x[2])
feature_index = [v[1] for v in vocab[:200]]
tfidf = tfidf[:, feature_index]
#words = vectorizer.get_feature_names()

# for i in range(len(corpus)):
#     print('----Document %d----' % (i))
#     for j in range(len(words)):
#         if tfidf[i, j] > 0.2:
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
    #logloss = 
    return accuracy, bias

from sklearn.linear_model import LogisticRegression
from collections import Counter
weight = Counter(label)
for cat in list(weight.keys()):
    weight[cat] = int(1000.0 / weight[cat])

from sklearn.cross_validation import StratifiedKFold
kf = StratifiedKFold(label, n_folds=5, shuffle=True, random_state=1233)

from sklearn.dummy import DummyClassifier

from sklearn.metrics import log_loss

for train_index, test_index in kf:
    for index, state in zip([train_index, test_index], ['train', 'test']):
        dummy_model = DummyClassifier(strategy='most_frequent')
        dummy_model.fit(tfidf[index, :], label[index])
        dummy_predicted_label = dummy_model.predict_proba(tfidf[index, :])
        dummy_mean_accuracy = dummy_model.score(tfidf[index, :], label[index])
        #dummy_log_loss = log_loss(label[index], dummy_predicted_label, eps=1e-15, normalize=True)

        model = LogisticRegression(C=1e+5, multi_class='ovr', n_jobs=1, class_weight='balanced')
        model.fit(tfidf[index, :], label[index])
        predicted_label = model.predict_proba(tfidf[index, :])
        mean_accuracy = model.score(tfidf[index, :], label[index])

        print(state + " accuracy = {:f}, dummy accuracy = {:f}".format(mean_accuracy, dummy_mean_accuracy))

        #log_loss = log_loss(label[index], predicted_label, eps=1e-15, normalize=True)
        #print(state + " logloss = {:f}, dummy logloss = {:f}".format(log_loss, dummy_log_loss))
        