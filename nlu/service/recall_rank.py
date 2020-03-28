#coding:utf-8
import sys
sys.path.extend(["../textcnn","../BM25","../Bool","../kmeans","../SIF","../syntax","../cosine"])
from textcnn_predict import TextcnnPredict
from data_loader import config,load_pickle
from km_recall import KmRecall
from bm25_recall import Bm25Recall
from bool_recall import BoolRecall
from SIF_rank import SIFRank
from cosine_rank import CosRank
from finalAnswer import suggest_answer
import time
from itertools import chain



class Rank(object):
    
    def __init__(self):
        self.textcnn = TextcnnPredict()
        self.km_recall = KmRecall().recall
        self.bm_recall = Bm25Recall().recall
        self.bool_recall = BoolRecall().recall
        self.sif_rank = SIFRank().rank
        self.cosine_rank = CosRank().rank
        self.answer_map = load_pickle(config.answer_map_path)
        
    
    def _recall_topn(self,query,topn=10):
        
        ques_type = self.textcnn.predict(query)
           
        topn_km = self.km_recall(query,ques_type)
        topn_bm = self.bm_recall(query,ques_type)
        topn_bool = self.bool_recall(query,ques_type)
        print(topn_km)
        print(topn_bm)
        print(topn_bool)
       
        topn_recall = list(set(topn_km + topn_bm + topn_bool))
        
        return topn_recall, ques_type
    
    def _get_top_one(self,query,topn=10):
        
        topn_recall, ques_type = self._recall_topn(query)
    
        top_one, score = self.cosine_rank(query,topn_recall)
    
        return top_one, ques_type, score
    
    def get_answer(self, query, topn=10, threshold=0.2):
        
        topn_one,ques_type,score = self._get_top_one(query)
        print(topn_one)
        print(score)
        
        if ques_type == "busi" and score < threshold:
            return suggest_answer(), ques_type

        if ques_type == "chat" and score < threshold:
            return "邓小帅竟无言以对！聊点别的呗[坏笑]",ques_type
        
        return self.answer_map[ques_type][topn_one], ques_type
            
    
if __name__ == "__main__":
    
    questions = ["办信用卡需要准备哪些证件","社保业务","存单业务","查询外汇汇率","你家乡在哪里","你好啊","你有男朋友吗","你觉得自己长得怎么样"]

    ranker = Rank()
    
    for query in questions:
        
        match_answer = ranker.get_answer(query)
        
        print("\nThe question matched is %s \n" % str(match_answer))
        
    
    




    
