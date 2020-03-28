#coding:utf-8
from bm25_model import BM25kapi
from bm25_config import BmConfig
import pandas as pd
import numpy as np
import jieba,os,re

np.random.seed(10)
config = BmConfig()


""" 一:清洗文本并划分数据集 """
def load_corpus(config):
    
    """ 读取数据 """
    print("\nLoading the dataset ... \n")
    corpus_xls = pd.ExcelFile(config.corpus_path)
    business_df = corpus_xls.parse("business_question")
    chatting_df = corpus_xls.parse("chatting_question")
    
    """ 进行文本清洗 """
    business_df["question_seg"] = business_df["question"].apply(clean_text)
    chatting_df["question_seg"] = chatting_df["question"].apply(clean_text)    
    
    business_df.dropna(inplace=True)
    chatting_df.dropna(inplace=True)    
    
    """ to list """
    busi_ques = business_df["question"].tolist()
    chat_ques = chatting_df["question"].tolist()

    return busi_ques, chat_ques


""" 进行文本清洗 """
def clean_text(text):
    text = re.sub(
            "[\s+\-\|\!\/\[\]\{\}_,.$%^*(+\"\')]+|[:：+——()?【】《》“”！，。？、~@#￥%……&*（）]+", '',str(text).lower())
    
    if not text:
        return np.nan
    return text

""" 用jieba分词 """
def clean_seg(text):
    text = clean_text(text)
    if type(text) is not str:
        return []
    else:
        return jieba.lcut(text)


""" 二：bm25取前5最相似句子 """
class Bm25Recall(object):
    def __init__(self):
        self.busi_ques, self.chat_ques = load_corpus(config)
        self.tokenizer = clean_seg
        self.bm25_busi = BM25kapi(self.busi_ques, self.tokenizer)
        self.bm25_chat = BM25kapi(self.chat_ques, self.tokenizer)
        
    def recall(self, query, ques_type, topn=10):
        
        if ques_type == "busi":
            return self.bm25_busi.get_top_n(query, self.busi_ques, n=topn)
        
        elif ques_type == "chat":
            return self.bm25_chat.get_top_n(query, self.chat_ques, n=topn)
        

if __name__ == "__main__":
    
    model = Bm25Recall()
    
    questions = ["办信用卡需要准备哪些证件","社保业务","存单业务","查询外汇汇率","你家乡在哪里","我好饿啊","你有男朋友吗","你觉得自己长得怎么样"]
    question_types = ["busi"] * 4 + ["chat"] * 4
    
    for question, type_ in zip(questions, question_types):
        print(model.recall(question,type_))    