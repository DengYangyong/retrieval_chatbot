# -*- coding:utf-8 -*-
import re,os,time
from collections import defaultdict


dictionary = {}
matrix = {}
# This is the map of docId to input file name
docIdMap = {}


class BoolSearch:
    def __init__(self, documents,tokenizer=None):
        
        self.documents = documents
        self.tokenizer = tokenizer
        self.dic_word = defaultdict(list)
        self.doc_map = {}
        self.dic_matrix = {}
        
        if tokenizer:
            self.documents = self._tokenize_doc(self.documents)
            
        
    """ 分词 """   
    def _tokenize_doc(self,documents):
        tokenized_docments = [self.tokenizer(doc) for doc in documents]
        return tokenized_docments        
    
    """ 得到布尔矩阵 """
    def _build_matrix(self):
        
        """ 得到倒排表 """
        for doc_id, doc in enumerate(self.documents):
            count = 0
            self.doc_map[doc_id] = doc

            for word in doc:
                self.dic_word[word].append[doc_id]
                
        """ 得到布尔矩阵"""
        for word,doc_ids in self.dic_word:
            
            vector = []
            for doc_id in self.doc_map:
                if doc_id in doc_ids:
                    vector.append(1)
                else:
                    vector.append(0)
                    
            self.dic_matrix[word] = vector



def main():
    pass

if __name__ == '__main__':
    main()
