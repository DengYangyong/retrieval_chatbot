import pandas as pd
import numpy as np
import jieba
import jieba.analyse


class BoolSearch():
  def __init__(self,data):
    self.data = data
    self.doc_num = len(self.data.question)
    self.stopwords = open('./哈工大停用词表.txt').read().split('\n')
    self.index = self.get_inverted_index(self.data)
    self.vector = self.get_word_vector(self.index)

  # 获取倒排索引表
  def get_inverted_index(self,data):
    allwords = set([word for line in data.question for word in jieba.lcut(str(line)) if word not in self.stopwords])
    inverted_index = {}
    for word in allwords:
      doc = []
      for index,question in enumerate(data.question):
        if word in str(question):
          doc.append(index)
        else:
          continue
      inverted_index[word] = doc
    return inverted_index

  # 生成语料库中的词的vector
  def get_word_vector(self,index):
    vector = {}
    for word in index.keys():
      code = np.zeros(self.doc_num).astype(np.int16)
      for doc_index in index[word]:
        code[doc_index] = 1
      vector[word] = code
    return vector

  # 获取输入语句的关键词  
  def get_inputs_keywords(self,inputs):
    outputs = []
    key_words = jieba.analyse.extract_tags(inputs,withWeight=True)
    for word in key_words:
      if word[0] in self.index.keys():
        outputs.append(word[0])
    return outputs

  # 获取输入语句关键词的code
  def get_inputs_code(self,inputs):
    outputs = self.get_inputs_keywords(inputs)
    inputs_code = []
    for keyword in outputs:
      inputs_code.append(self.vector[keyword])
    return inputs_code

  # 获取同时包含关键词的文档
  def get_doc(self,inputs):
    inputs_code = self.get_inputs_code(inputs)
    if_false = np.zeros(self.doc_num).astype(np.int16)

    for i in range(3,0,-1):
      doc = np.where(np.array(inputs_code[:i]).sum(axis=0)==len(inputs_code[:i]), 1,0)
      if (doc == if_false).all():
        continue
      else:
        break

    doc_index = []
    for index,i in enumerate(doc):
      if i == 1:
        doc_index.append(index)
    
    return list(self.data.iloc[doc_index[:10]].question)

if __name__ == "__main__":
  business_data = pd.read_excel('./qa_corpus_v.xlsx')
  chatting_data = pd.read_excel('./chatting.xlsx')

  business_bs = BoolSearch(business_data)
  chatting_bs = BoolSearch(chatting_data)

  business_ = business_bs.get_doc("办信用卡需要准备哪些证件")
  chatting_ = chatting_bs.get_doc("今天的风儿甚是喧嚣")

  print("办信用卡需要准备哪些证件:{}".format(business_))
  print("\n")
  print("今天的风儿甚是喧嚣:{}".format(chatting_))