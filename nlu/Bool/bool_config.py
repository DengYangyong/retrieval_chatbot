#coding:utf-8
import numpy as np
import os,pathlib

root = pathlib.Path(os.path.abspath(__file__)).parent.parent.parent


class BoolConfig(object):
    def __init__(self):
        self.corpus_path = os.path.join(root,"data","dataset","origin","qa_corpus.xlsx")
        
