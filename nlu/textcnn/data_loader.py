#coding:utf-8
import torch
from torchtext import data
from torchtext.vocab import Vectors
import os,re,jieba
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from textcnn_config import TextcnnConfig
import matplotlib.pyplot as plt 
import pickle

config = TextcnnConfig()
np.random.seed(10)

""" 一:清洗文本并划分数据集 """
def build_dataset(config):
    
    """ 读取数据 """
    print("\nLoading the dataset ... \n")
    corpus_xls = pd.ExcelFile(config.corpus_path)
    busi_df = corpus_xls.parse("business_question")
    chat_df = corpus_xls.parse("chatting_question")
    busi_df.dropna(inplace=True)
    chat_df.dropna(inplace=True)

    """ 贴标签 """ 
    print("\nLabeling the dataset ... \n")
    busi_df['label'] = 'busi'
    chat_df['label'] = 'chat'
    
    """ 数据清洗和分词 """ 
    print("\nCleaning text and segmenting ... \n")
    busi_df['question_seg'] = busi_df['question'].apply(clean_seg)
    chat_df['question_seg'] = chat_df['question'].apply(clean_seg)
    busi_df.dropna(inplace=True)
    chat_df.dropna(inplace=True) 
    
    """ 问题到答案的映射 """
    answer_map = {}
    answer_map["busi"] = dict(zip(busi_df["question"],busi_df["answer"]))
    answer_map["chat"] = dict(zip(chat_df["question"],chat_df["answer"]))
    save_pickle(answer_map,config.answer_map_path)
    
    """ 划分数据集并保存 """ 
    print("\nMerging and spliting dataset ... \n")
    corpus_df = pd.concat([busi_df,chat_df],axis=0,sort=True)
    corpus_df.dropna(inplace=True)
    
    config.max_lengths = calcu_max_len(corpus_df)
    print(f"\nThe shape of the qa corpus : {corpus_df.shape}\n")
    
    train_data, test_data = train_test_split(corpus_df[['question','label']],test_size=0.15)
    train_data, valid_data = train_test_split(train_data, test_size=0.15)
    
    train_data.to_csv(os.path.join(config.proc_dir,'train.csv'),header=True, index=False)
    valid_data.to_csv(os.path.join(config.proc_dir,'valid.csv'),header=True, index=False)
    test_data.to_csv(os.path.join(config.proc_dir,'test.csv'),header=True, index=False)

""" 进行文本清洗，并用jieba分词 """
def clean_seg(line):
    line = re.sub(
            "[\s+\-\|\!\/\[\]\{\}_,.$%^*(+\"\')]+|[:：+——()?【】《》“”！，。？、~@#￥%……&*（）]+", '',str(line).lower())
    words = jieba.lcut(line, cut_all=False)
    if not words:
        return np.nan
    return ' '.join(words)


""" 计算输入长度，使其能涵盖98%样本 """ 
def calcu_max_len(df):
    
    df["lengths"] = df["question_seg"].apply(lambda x:x.count(' ')+1)
    max_lengths = max(df["lengths"])
    for len_ in range(5,max_lengths,5):
        bool_ = df["lengths"] < len_
        cover_rate = sum(bool_.apply(int)) / len(bool_)
        if cover_rate >= 0.98:
            return len_

    
""" 二: 生成 batch 迭代器 """   
def batch_generator(config):
    
    """ 定义Field对象 """ 
    print("\nDefining the Field ... \n")
    tokenizer = lambda x: x.split(" ")
    TEXT = data.Field(sequential=True, 
                      tokenize=tokenizer,
                      batch_first=True,
                      fix_length=config.max_lengths,
                      include_lengths=False)
    
    LABEL = data.LabelField(sequential=False,
                            dtype=torch.int64)
    fields = [('question_seg', TEXT),('label', LABEL)]
    
    """ 加载CSV数据，建立词汇表 """ 
    print("\nBuilding the vocabulary ... \n")
    train_data, valid_data, test_data = data.TabularDataset.splits(
                                            path = config.proc_dir,
                                            train = 'train.csv',
                                            validation = 'valid.csv',
                                            test = 'test.csv',
                                            format = 'csv',
                                            fields = fields,
                                            skip_header = True) 
    
    """ 千万注意，创建成功后 ，vocab size 为 max features 加 2
     而embedding层的input size 为 vocab size。"""
    vectors_bk = Vectors(name="w2v_bk.txt", cache=config.w2v_dir)
    
    TEXT.build_vocab(train_data,
                     min_freq=1,
                     vectors=vectors_bk,
                     unk_init=torch.Tensor.normal_)
    LABEL.build_vocab(train_data)
    
    config.vocab_size = len(TEXT.vocab)
    config.embed_matrix = TEXT.vocab.vectors
    
    print(f"\nUnique tokens in TEXT vocabulary: {len(TEXT.vocab)}\n")
    print(f"\nLABEL vocabulary is: {LABEL.vocab.stoi}\n") 
    
    """ 把 vocab 文件保存为pickle对象，预测时备用。 """
    print("\nSaving the vocab file\n")
    save_pickle(TEXT.vocab, config.vocab_path)
    
    """ 把对应的词向量矩阵保存备用 """
    np.save(config.embed_path, TEXT.vocab.vectors)
    
    print(f"\nInput size or vocab size is {len(TEXT.vocab)}\n")
    config.vocab_size = len(TEXT.vocab)
    print(f"\nPad index is {TEXT.vocab.stoi[TEXT.pad_token]}")
    config.pad_idx = TEXT.pad_token
    
    """ 生成batch """ 
    print("\nCreating the batch ... \n")
    train_iter, valid_iter, test_iter = data.Iterator.splits(
                                          (train_data, valid_data, test_data), 
                                          batch_sizes = (config.batch_size,) * 3,
                                          # 这里注意加一行，而且 item 是之前在Field里面定义好的
                                          sort_key = lambda x: len(x.question_seg),
                                          sort_within_batch=False,
                                          device=config.device)
    
    return train_iter, valid_iter, test_iter


""" 三：计算类别权重，缓解类别不平衡问题 """    
def calcu_class_weights(config):
    
    """ 读取标签数据并转化为数字（非one-hot） """ 
    train_data = pd.read_csv(os.path.join(config.proc_dir,"train.csv"))
    labels = train_data["label"].map(config.class_map)
    labels = np.array(labels.tolist(),dtype=np.int32)
    
    """ 计算class weights """ 
    freqs = np.bincount(labels)
    
    """ 作图观察类别不平衡情况 """ 
    visualize_freqs(freqs)
    
    p_class = freqs / len(labels)
    class_weights = 1 / np.log(1.02 + p_class)
    
    class_weights = torch.FloatTensor(class_weights).to(config.device)
    return class_weights
 
""" 观察是否存在类别不平衡问题 """
def visualize_freqs(freq):
    plt.bar(range(2),freq,width=0.25,color=['r','b'],label="question")
    plt.xticks(range(2),["chatting_question","business_question"])
    plt.title("The frequencies of Two classes")
    plt.legend()
    plt.savefig(os.path.join(config.proc_dir,"class_weights.png"))
    plt.clf()    

""" 保存为pickle对象 """ 
def save_pickle(s,file_path):
    with open(file_path,'wb') as f:
        pickle.dump(s,f)
        
""" 加载pickle对象 """ 
def load_pickle(file_path):
    with open(file_path,'rb') as f:
        s = pickle.load(f)
    return s


if __name__ == "__main__":
    
    
    """ 划分数据集并保存 """ 
    build_dataset(config)
    
    """ 计算不平衡样本的class weights """ 
   # class_weights = calcu_class_weights(config)
   # config.class_weights = class_weights
    
    """ 生成batch """ 
    #train_iter, valid_iter, test_iter = batch_generator(config)
    #for x_batch, y_batch in train_iter:
     #   print(f"The shape of item batch is {x_batch.shape}")
     #   print(f"The shape of label batch is {y_batch.shape}")
    
    """
    The format of shape is [sequence lengths, batch size], which is different from tensorflow.
    
    The shape of item batch is torch.Size([573, 64])
    The shape of label batch is torch.Size([64])
    """

