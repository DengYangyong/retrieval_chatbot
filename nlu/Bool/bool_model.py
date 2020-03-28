# -*- coding:utf-8 -*-
import re,os,time
from collections import defaultdict
from jieba.analyse import extract_tags
import numpy as np
from itertools import combinations


class BoolSearch:
    def __init__(self, documents,tokenizer=None):
        
        self.documents = documents
        self.tokenizer = tokenizer
        self.dic_word_doc = defaultdict(list)
        self.doc_map = {}
        self.dic_word_id = defaultdict(int)
        self.matrix = self._build_matrix()
                
                
    """ 得到倒排表 """
    def _build_rev_dic(self):
        
        for doc_id, doc in enumerate(self.documents):
            self.doc_map[doc_id] = doc
            
            words = self.tokenizer(doc)
            for word in words:
                self.dic_word_doc[word].append(doc_id)
      
                
    """ 得到布尔矩阵 """           
    def _build_matrix(self):
        
        self._build_rev_dic()
        
        """ 构造布尔矩阵 """
        word_num = len(self.dic_word_doc)
        doc_num = len(self.documents)
        matrix = np.zeros((word_num,doc_num)).astype(np.int16)
               
        for word_id,(word,doc_ids) in enumerate(self.dic_word_doc.items()):
            
            for doc_id in self.doc_map:
                if doc_id in doc_ids:
                    matrix[word_id,doc_id] = 1 
            
            """ 构建词表 """ 
            self.dic_word_id[word] = word_id
            
        return matrix
    
    """ 取出所有关键词的布尔向量 """
    def _get_vector_inter(self,word_ids):
        
        """ 取取布尔向量 """
        vectors = self.matrix[word_ids]
        
        """ 求交集 """
        vector_inter = np.where(vectors.sum(axis=0) == len(word_ids),1,0)
        
        return vector_inter
            
    """ 取出包含文档的布尔向量 """
    def _get_vector(self,word_ids,topn=3):
        
        """ 返回 [] """
        if topn == 0:
            return []
        
        """ 如果关键词数量小于3，那么把topn设为关键词数量 """
        topn = len(word_ids[:topn])
        
        """ 对关键词做组合 """
        comb_ids = list(combinations(range(len(word_ids)),topn))
        for ids in comb_ids:
            word_ids_f = [word_ids[idx] for idx in ids]
            vector_inter = self._get_vector_inter(word_ids_f)
            
            """ 取到文档，则返回结果 """
            if max(vector_inter) == 1:
                return vector_inter
        
        return self._get_vector(word_ids, topn-1)
                
                
    """ 取出前topn条问题 """
    def get_topn(self,query,n=10):
        
        """ 得到关键词 """
        words = self.tokenizer(str(query))
        
        """ 过滤不在词表中的词 """
        word_ids = [self.dic_word_id[word] for word in words if word in self.dic_word_id] 
        
        if not word_ids:
            return []
        
        """ 取出布尔向量 """
        vector = self._get_vector(word_ids)
        
        if len(vector) == 0:
            return []
        
        docs = [self.doc_map[i] for i,idx in enumerate(vector) if idx==1]
        return docs[:n]