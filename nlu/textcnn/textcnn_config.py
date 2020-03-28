#coding:utf-8
import numpy as np
import os,pathlib
import torch

root = pathlib.Path(os.path.abspath(__file__)).parent.parent.parent


class TextcnnConfig(object):
    def __init__(self):
        self.corpus_path = os.path.join(root,"data","dataset","origin","qa_corpus.xlsx")
        self.answer_map_path = os.path.join(root,"data","dataset","origin","answer_map.pkl")
        self.proc_dir = os.path.join(root,"data","dataset","proc")
        self.class_map = {"chat":0, "busi": 1,}
        self.class_map_reverse = {0:"chat",1:"busi"}
        self.stopwords_path = os.path.join(root,"data","stopwords","哈工大停用词表.txt")
        self.w2v_dir = os.path.join(root,"data","w2v")
        self.save_path = os.path.join(root,"data","textcnn_results","question_cls.h5")
        self.vocab_path = os.path.join(root,"data","w2v","vocab.pickle")
        self.embed_path = os.path.join(root,"data","w2v","embed_matrix.npy")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.vocab_size = 11368
        self.batch_size = 64
        self.max_lengths = 15
        self.pad_idx = 1
        self.embed_dim = 300
        self.filter_sizes = [2,3,4]
        self.num_filters = 64
        self.dense_units = 32       
        self.num_classes = 2
        self.dropout = 0.5
        self.learning_rate = 1e-3
        self.num_epochs = 50
        self.max_grad_norm = 2.0
        self.gamma = 0.9
        self.require_improve = 2000
        
