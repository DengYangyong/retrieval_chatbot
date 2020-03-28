#coding:utf-8
import sys
from SIF_embedding import sif_embedding
from data_utils import load_json,load_pickle,calcu_sents_id,calcu_sents_weight
import numpy as np
import pandas as pd
from SIF_config import SIFConfig
from SIF_prepare import clean_seg
from sklearn.metrics.pairwise import cosine_similarity
import time

config = SIFConfig()

class SIFRank(object):
    def __init__(self):
        '''
        提前准备好计算句向量必备的文件：word2id，id2embed，id2weight
        '''
        self.word_id = load_json(config.word_id_path)
        self.id_emb = load_pickle(config.id_emb_path)
        self.id_weight = load_pickle(config.id_weight_path)
        self.tokenizer = clean_seg
    
    def _sentence_embedding(self,sentences):
        '''
        :param sentences: 分好词的n条句子
        :return: n条句子的sif句向量
        '''
        sents_id_array = calcu_sents_id(sentences, self.word_id) 
        sents_weight_array = calcu_sents_weight(sents_id_array, self.id_weight) 

        sents_sif_embeds = sif_embedding(self.id_emb, sents_id_array, sents_weight_array) 
        return sents_sif_embeds    
      
    def _calcu_score(self,embeddings):
      
        cosine_scores = []
        query_emb = embeddings[0]
        for emb in embeddings[1:]:
            score = cosine_similarity(emb.reshape(1,-1), query_emb.reshape(1,-1))
            cosine_scores.append(score.item())
        return cosine_scores

             
    def rank(self,query,topn_recall):
       
        sentences_all = [query] + topn_recall
        sentences_seg = [self.tokenizer(text) for text in sentences_all]
        
        embeddings = self._sentence_embedding(sentences_seg)     
        scores = self._calcu_score(embeddings)       
        idx = np.argmax(scores)
        max_score = scores[idx]
        
        return topn_recall[idx], max_score       

if __name__ == '__main__':

    ranker = SIFRank()
    
    query = "办信用卡需要准备哪些证件"
    
    topn_recall = ['哪些证件属于存款实名制证件', '取号需要什么证件', '开户需要什么证件', '出国留学准备', '办存单', '外卡柜面取现需要携带证件么', '我要办业务', '智慧柜员机办存折', '智慧柜员机办存单', '党费管家业务办理准备工作']
    
    print(ranker.rank(query, topn_recall))
    