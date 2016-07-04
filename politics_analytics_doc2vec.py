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




import gensim
size = 500

# initiate DM and DBOW model
model_dm = \
	gensim.models.Doc2Vec(min_count=1, window=10, size=size, sample=1e-3, negative=5, workers=3)
model_dbow = \
	gensim.models.Doc2Vec(min_count=1, window=10, size=size, sample=1e-3, negative=5, dm=0, workers=3)

# build vocab
model_dm.build_vocab(corpus)
model_dbow.build_vocab(corpus)

# train
for epoch in range(10):
	perm = np.random.permutation(corpus)
	model_dm.train(corpus[perm])
