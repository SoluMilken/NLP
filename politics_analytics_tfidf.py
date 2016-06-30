# -*- coding: utf-8 -*-
import json
import jieba
import ipdb

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

for i in range(len(corpus)):
    print('----Document %d----' % (i))
    for j in range(len(words)):
        if tfidf[i, j] > 0.1:
            print(words[j], tfidf[i, j])

ipdb.set_trace()
