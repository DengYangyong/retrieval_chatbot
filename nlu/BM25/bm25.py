# -*- coding:utf-8 -*-
import math
import jieba
import numpy as np
import logging
import pandas as pd
from collections import Counter
jieba.setLogLevel(logging.INFO)


class BM25(object):
  def __init__(self,docs):
    self.docs = docs   # 传入的docs要求是已经分好词的list 
    self.doc_num = len(docs) # 文档数
    self.vocab = set([word for doc in self.docs for word in doc]) # 文档中所包含的所有词语
    self.avgdl = sum([len(doc) + 0.0 for doc in docs]) / self.doc_num # 所有文档的平均长度
    self.k1 = 1.5
    self.b = 0.75

  def idf(self,word):
    if word not in self.vocab:
      word_idf = 0
    else:
      qn = {}
      for doc in self.docs:
        if word in doc:
          if word in qn:
            qn[word] += 1
          else:
            qn[word] = 1
        else:
          continue
      word_idf = np.log((self.doc_num - qn[word] + 0.5) / (qn[word] + 0.5))
    return word_idf

  def score(self,word):
    score_list = []
    for index,doc in enumerate(self.docs):
      word_count = Counter(doc)
      if word in word_count.keys():
        f = (word_count[word]+0.0) / len(doc)
      else:
        f = 0.0
      r_score = (f*(self.k1+1)) / (f+self.k1*(1-self.b+self.b*len(doc)/self.avgdl))
      score_list.append(self.idf(word) * r_score)
    return score_list 

  def score_all(self,sequence):
    sum_score = []
    for word in sequence:
      sum_score.append(self.score(word))
    sim = np.sum(sum_score,axis=0) 
    return sim

def cut(sentence):
  return [word for word in jieba.lcut(sentence) if word not in stopwords]


def get_topn(data,inputs):
  docs = []
  for line in data.question:
    tokens = cut(line)
    docs.append(tokens)

  question = cut(inputs)

  bm = BM25(docs)
  score = bm.score_all(question)

  sim_list = [(index,sim) for index,sim in enumerate(score)]
  sim_list = sorted(sim_list,key=lambda x:x[1],reverse=True)
  sim_list = sim_list[:10]

  output = []
  for t in sim_list:
    output.append((data.question[t[0]],t[1]))

  return output

if __name__ == "__main__":
  data = pd.read_excel('./drive/My Drive/Colab/project3_dataset/qa_corpus_v.xlsx')
  get_topn(data,"我要办信用卡")