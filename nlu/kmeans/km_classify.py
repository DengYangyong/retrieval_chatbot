#coding:utf-8
import pandas as pd
import numpy as np
from gensim.models import word2vec
import jieba,os,re
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import pickle
from km_config import KmConfig
from collections import defaultdict
from sklearn.cluster import KMeans
from numpy import math
from itertools import chain

np.random.seed(10)
config = KmConfig()

""" 保存为pickle对象 """ 
def save_pickle(s,file_path):
    with open(file_path,'wb') as f:
        pickle.dump(s,f, protocol=2)

""" 一:清洗文本并划分数据集 """
def load_corpus(config):
    
    """ 读取数据 """
    print("\nLoading the dataset ... \n")
    corpus_xls = pd.ExcelFile(config.corpus_path)
    business_df = corpus_xls.parse("business_question")
    chatting_df = corpus_xls.parse("chatting_question")
    
    """ 进行分词 """
    business_df["question_seg"] = business_df["question"].apply(clean_seg)
    chatting_df["question_seg"] = chatting_df["question"].apply(clean_seg)   
    
    business_df.dropna(inplace=True)
    chatting_df.dropna(inplace=True)    
    
    """ load wor2vec """
    vocab,emb_matrix = load_vocab_emb()
    
    busi_ques_seg = business_df["question_seg"].tolist()
    busi_ques = business_df["question"].tolist()
    busi_ques_emb = calcu_weighted_emb(vocab, emb_matrix, busi_ques_seg)

    chat_ques_seg = chatting_df["question_seg"].tolist()
    chat_ques = chatting_df["question"].tolist()
    chat_ques_emb = calcu_weighted_emb(vocab, emb_matrix, chat_ques_seg)
    
    return busi_ques_emb, busi_ques, chat_ques_emb, chat_ques


""" 进行文本清洗，并用jieba分词 """
def clean_seg(line):
    line = re.sub(
            "[\s+\-\|\!\/\[\]\{\}_,.$%^*(+\"\')]+|[:：+——()?【】《》“”！，。？、~@#￥%……&*（）]+", '',str(line).lower())
    words = jieba.lcut(line, cut_all=False)
    if not words:
        return np.nan
    return words

""" 对文本进行pad，并转化为id """
def sent_to_emb(words,vocab,emb_matrix):
    words_sub = [word if word in vocab.stoi else '<unk>'for word in words]
    ids = [vocab.stoi[w] for w in words_sub]
    embs = emb_matrix[ids]
    return embs
    

def calcu_weighted_emb(vocab, emb_matrix, queses):
        
    sentence_embs = [sent_to_emb(ques, vocab,emb_matrix) for ques in queses]
    sentence_embs = [np.average(emb, axis=0) for emb in sentence_embs]
        
    return sentence_embs


def load_vocab_emb():
    
    vocab = load_pickle(config.vocab_path)
    emb_matrix = np.load(config.embed_path)
    
    return vocab,emb_matrix


def train_kmeans(ques_embs,config, type_):
    
    """ train kmeans"""
    silhouette_all = []
    for n in range(10,200,5):
        km = KMeans(n_clusters=n,max_iter=500)
        km.fit(ques_embs)
        ques_class = km.predict(ques_embs)
    
        """ 计算聚类的轮廓系数 """
        sil = silhouette_score(ques_embs, ques_class, metric='euclidean')
        print("{}类数量为{}时的轮廓系数为: {}\n".format(type_,n,sil)) 

        silhouette_all.append(sil)
        
    class_ = list(range(10,200,5))
    sil_max = max(silhouette_all)
    index = silhouette_all.index(sil_max)
    n_cluster = class_[index]
    print("%s 类轮廓系数最高的个数为 %d,对应的轮廓系数为：%.4f\n" % (type_, n_cluster,sil_max))  
    
    visualize_sil(silhouette_all, class_,config,type_)
    
    return n_cluster
    
    
def visualize_sil(sil_all,class_,config,type_):
    
    print("对个数和轮廓系数的关系进行可视化 ...\n")
    plt.plot(class_,sil_all,'r',label=type_)
    plt.title("Silhouette values and cluster numbers")
    plt.xlabel('cluster numbers')
    plt.ylabel('silhouette value') 
    plt.legend()
    save_path = os.path.join(config.km_dir,'silhouette_{}.png'.format(type_))
    plt.savefig(save_path)
    plt.clf()
            
    
def retrain_kmeans(n,ques_embs,queses,config, type_):
    
    assert len(ques_embs) == len(queses)
    
    """ train kmeans"""
    km = KMeans(n_clusters=n,max_iter=500)
    km.fit(ques_embs)
    
    ques_class = km.predict(ques_embs)
    
    ques_dic = defaultdict(list)
    for class_, emb, ques in zip(ques_class,ques_embs,queses):
        ques_dic[class_].append((emb,ques))
    
    km_path = os.path.join(config.km_dir,'km_{}.model'.format(type_))
    result_path = os.path.join(config.km_dir,'km_cls_{}.pickle'.format(type_))
    
    save_pickle(km, km_path)
    save_pickle(ques_dic,result_path)
 
""" 加载pickle对象 """ 
def load_pickle(file_path):
    with open(file_path,'rb') as f:
        s = pickle.load(f)
    return s   

def main(config):
    
    busi_ques_emb, busi_ques, chat_ques_emb, chat_ques = load_corpus(config)
    
    #busi_n_cluster = train_kmeans(busi_ques_emb, config, "busi")
    #chat_n_cluster = train_kmeans(chat_ques_emb, config, "chat")
    
    busi_n_cluster = 20
    chat_n_cluster = 50
     
    retrain_kmeans(busi_n_cluster, busi_ques_emb, busi_ques, config, "busi")
    retrain_kmeans(chat_n_cluster, chat_ques_emb, chat_ques, config, "chat")
    
    
if __name__ == "__main__":
    
    main(config)
    
    