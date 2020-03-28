#coding:utf-8
import numpy as np
import os,pathlib

root = pathlib.Path(os.path.abspath(__file__)).parent.parent.parent


class SIFConfig(object):
    def __init__(self):
        self.corpus_path = os.path.join(root,"data","dataset","origin","qa_corpus.xlsx")
        self.sif_dir = os.path.join(root,"data","SIF_data")
        self.word_id_path = os.path.join(self.sif_dir,"word_id.json")
        self.id_emb_path = os.path.join(self.sif_dir,"id_embed.pickle")
        self.id_weight_path = os.path.join(self.sif_dir,"id_weight.pickle")
        self.w2v_path = os.path.join(root,"data","w2v","w2v_bk.txt")
        
        self.proc_dir = os.path.join(self.sif_dir,"proc")
        
        
