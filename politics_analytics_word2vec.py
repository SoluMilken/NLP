# -*- coding: utf-8 -*-
import jieba
import ipdb
import logging
import json

#from nlp.utils import IterTimer
from gensim.models import Word2Vec
from gensim import corpora, models, similarities

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


jieba.set_dictionary('dict.txt.big.txt')

topic = "是否滿意北市府的處理態度"
#"是否滿意北市府的處理態度" / "是否支持華航工會罷工"
topic_output_path = \
    "/home/en/Natural_Language_Processing/dataset/" + topic + ".json"

#with IterTimer('tokens'):
tokens = open('nlp/symbol_to_be_removed/tokens.txt', 'rb').read().decode("utf-8")
stop_words = open('nlp/symbol_to_be_removed/stop_words_chinese.txt', 'rb').read().decode("utf-8") 
stop_words = " ".join(stop_words.split(","))

tokens_and_stopwords = (tokens + " " + stop_words)

#with IterTimer('jieba cutting'):
corpus = []
with open(topic_output_path) as json_file:
    json_data = json.load(json_file)
    dataset = [None] * len(json_data['data'])
    for num, comment_dic in enumerate(json_data['data']):
        paragraph = []
        comment_cut_list = []
        comment = comment_dic['content']
        label = comment_dic['answer']

        # jieba cutting
        comment_cut_generater = jieba.cut(comment, cut_all=False)

        # filtering
        for word in comment_cut_generater:
            if word not in tokens_and_stopwords:
                comment_cut_list.append(word)
                paragraph.append(word)
            dataset[num] = [comment_cut_list, label]
        corpus.append(paragraph)

dictionary = corpora.Dictionary(corpus)
print(len(dictionary))


model = Word2Vec(corpus, size=100, window=5, min_count=5, workers=1)
#print(model[u"台灣"])
# y2 = model.most_similar(u"台灣", topn=5)
# for item in y2:
#     print(item[0], item[1])
# print("--------\n")

ipdb.set_trace()

# # find relationship
# print u"书-不错，质量-"
# y3 = model.most_similar([u'质量', u'不错'], [u'书'], topn=3)
# for item in y3:
#     print item[0], item[1]
# print "--------\n"


# # find words do not match
# y4 = model.doesnt_match(u"书 书籍 教材 很".split())
# print u"不合群的词：", y4
# print "--------\n"




#model.save(fname)
#model = Word2Vec.load(fname)




#print filterpunt(content)

