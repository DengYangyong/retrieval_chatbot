#coding:utf-8
import sys,pickle,time,re,jieba
import numpy as np
import pandas as pd
from cosine_config import CosConfig
from scipy.spatial.distance import cosine

config = CosConfig()

""" 加载pickle对象 """ 
def load_pickle(file_path):
    with open(file_path,'rb') as f:
        s = pickle.load(f)
    return s

""" 进行文本清洗 """
def clean_text(text):
    text = re.sub(
            "[\s+\-\|\!\/\[\]\{\}_,.$%^*(+\"\')]+|[:：+——()?【】《》“”！，。？、~@#￥%……&*（）]+", '',str(text).lower())
    if not text:
        return np.nan
    return text

def transfer_char(text):
    text = clean_text(text)
    if type(text) is not str:
        return []    
    return [char.strip() for char in text if char.strip()]

class CosRank(object):
    def __init__(self):
        '''
        提前准备好计算句向量必备的文件
        '''
        self.vocab = load_pickle(config.vocab_path)
        self.emb_matrix = np.load(config.embed_path)
        self.tokenizer = transfer_char
    
    def _sentence_embedding(self,words):

        ids = [(self.vocab.stoi.get(w) or 0) for w in words]
        embs = self.emb_matrix[ids]
        ques_emb = np.mean(embs,axis=0)
        
        return ques_emb
      
    def _calcu_score(self,embeddings):
      
        query_emb = embeddings[0]
        cosine_scores = [1 - cosine(query_emb,emb) if sum(emb)!=0 and sum(query_emb)!=0 else -1 for emb in embeddings[1:]]
        return cosine_scores
             
    def rank(self,query,topn_recall):
       
        sentences_all = [query] + topn_recall
        print(topn_recall)
        sentences_seg = [self.tokenizer(text) for text in sentences_all]
        
        embeddings = [self._sentence_embedding(sent)  for sent in sentences_seg]
        scores = self._calcu_score(embeddings)  
        print(scores)
        idx = np.argmax(scores)
        
        return topn_recall[idx], scores[idx]      

if __name__ == '__main__':

    ranker = CosRank()
    
    query = "取号需要什么证件"
    
    topn_recall = ['哪些证件属于存款实名制证件', '取号需要什么证件', '开户需要什么证件', '出国留学准备', '办存单', '外卡柜面取现需要携带证件么', '我要办业务', '智慧柜员机办存折', '智慧柜员机办存单', '党费管家业务办理准备工作']
    
    print(ranker.rank(query, topn_recall))
    
