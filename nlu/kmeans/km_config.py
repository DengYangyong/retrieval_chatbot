#coding:utf-8
import numpy as np
import os,pathlib

root = pathlib.Path(os.path.abspath(__file__)).parent.parent.parent


class KmConfig(object):
    def __init__(self):
        self.corpus_path = os.path.join(root,"data","dataset","origin","qa_corpus.xlsx")
        self.proc_dir = os.path.join(root,"data","dataset","proc")
        self.class_map = {"business_question":1, "chatting_question":0}     
        
        self.w2v_dir = os.path.join(root,"data","w2v")
        self.km_dir = os.path.join(root,"data","kmeans")
        self.vocab_path = os.path.join(root,"data","w2v","vocab.pickle")
        self.embed_path = os.path.join(root,"data","w2v","embed_matrix.npy")         
        
