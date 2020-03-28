#coding:utf-8
import numpy as np
import pandas as pd
from data_utils import calcu_word_freq,calcu_word_weight,calcu_id_weight,load_json,load_pickle,save_pickle, dump_json
from SIF_config import SIFConfig
import re,jieba
from itertools import chain
from tqdm import tqdm

config = SIFConfig()

""" 一:清洗文本并划分数据集 """
def load_corpus():
    
    """ 读取数据 """
    print("\nLoading the dataset ... \n")
    corpus_xls = pd.ExcelFile(config.corpus_path)
    business_df = corpus_xls.parse("business_question")
    chatting_df = corpus_xls.parse("chatting_question")
    
    """ 进行分词 """
    business_df["question_seg"] = business_df["question"].apply(clean_seg)
    chatting_df["question_seg"] = chatting_df["question"].apply(clean_seg)
    business_df["answer_seg"] = business_df["answer"].apply(clean_seg)
    chatting_df["answer_seg"] = chatting_df["answer"].apply(clean_seg)    
    
    business_df.dropna(inplace=True)
    chatting_df.dropna(inplace=True) 
    
    busi_ques_seg = business_df["question_seg"].tolist()
    busi_ans_seg = business_df["answer_seg"].tolist()
    busi_corpus = [ques + ans for ques,ans in zip(busi_ques_seg, busi_ans_seg)]
    
    chat_ques_seg = chatting_df["question_seg"].tolist()
    chat_ans_seg = chatting_df["answer_seg"].tolist()
    chat_corpus = [ques + ans for ques,ans in zip(chat_ques_seg, chat_ans_seg)]
    
    corpus = list(chain.from_iterable(busi_corpus + chat_corpus))
    vocab = {w:i for i,w in enumerate(set(corpus))}
    
    return corpus, vocab


""" 进行文本清洗，并用jieba分词 """
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


def prepare_sif():
    
    corpus,vocab = load_corpus()
    
    word_freq = calcu_word_freq(corpus, min_freq=1)
    word_weight = calcu_word_weight(word_freq)
    
    word2id, id2emb = load_emb_matrix(vocab)
    id2weight = calcu_id_weight(word2id, word_weight)
    
    dump_json(word2id, config.word_id_path)
    save_pickle(id2emb, config.id_emb_path)
    save_pickle(id2weight, config.id_weight_path)
    

""" 二: 加载预训练词向量，并与词表相对应 """ 
def load_emb_matrix(vocab):
    
    """ 1: 加载百度百科词向量 """ 
    print("\nLoading baidu baike word2vec ...\n")
    emb_dic = load_w2v(config.w2v_path)
    
    """ 2: 词向量矩阵与词表相对应 """ 
    emb_dic_new = {}
    vocab_dic_new = {}
    
    for word,index in vocab.items():
        emb = emb_dic.get(word)
        if emb is not None:
            emb_dic_new[index] = emb
            vocab_dic_new[word] = index
            
    return vocab_dic_new,emb_dic_new
        
""" 加载百度百科词向量 """
def load_w2v(path):
    
    file = open(path,encoding="utf-8")
    
    emb_dic = {}
    for i,line in tqdm(enumerate(file)):
        if i == 0:
            continue
        value = line.split()
        word = value[0]
        emb = np.asarray(value[1:], dtype="float32")
        emb_dic[word] = emb
        
    return emb_dic
    
if __name__ == "__main__":
    
    prepare_sif()
