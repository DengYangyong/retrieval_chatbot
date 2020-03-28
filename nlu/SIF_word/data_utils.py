#coding:utf-8
import numpy as np
import pickle,json,jieba,re
from collections import Counter
from itertools import chain

'''
中文维基文件里的title字段，准备成了自定义字典，可以加载到jieba里，提高分词准确率。
每次起服务时都导入自定义字典，导致速度有点慢。
'''
# jieba.load_userdict(user_dict_path)

# 加载json文件
def load_json(filename):
    return json.load(open(filename,'r',encoding='utf-8'))
# 保存为json文件
def dump_json(s,filename):
    return json.dump(s,open(filename,'w',encoding='utf-8'),indent=2,ensure_ascii=False)

# 保存为pickle对象
def save_pickle(subj,file_path):
    with open(file_path,'wb') as f:
        pickle.dump(subj,f)

# 加载pickle对象
def load_pickle(file_path):
    with open(file_path,'rb') as f:
        subj = pickle.load(f)
    return subj


def calcu_word_freq(corpus,min_freq):
    '''
    :param corpus: 未去重的词库
    :param min_freq: 过滤词频低于阈值的词
    :return: 词与词频的字典
    '''
    counter = Counter(corpus)
    counter = sorted(counter.items(),key=lambda x:x[1],reverse=True)
    word_freq = {word:freq for word,freq in counter if freq >= min_freq}
    return word_freq

def calcu_word_weight(word_freq,param=1e-3):
    '''
    :param word_freq: 词频字典
    :param param: 计算权重的一个参数
    :return: 词与权重的字典
    '''
    if param <= 0: 
        param = 1.0
    freq_sum = sum(word_freq.values())
    word_weight = {}
    for word, freq in word_freq.items():
        word_weight[word] = param / (param + freq / freq_sum)
    return word_weight

def calcu_word_id_embed(vocab,embeddings):
    '''
    :param vocab: 去重的词表
    :param embeddings: 词对应的词向量
    :return: 词与id的字典、id与词向量对应的字典
    '''
    word_id = dict(zip(vocab,range(len(vocab))))
    id_embed = dict(zip(range(len(vocab)),embeddings))
    return word_id,id_embed

def calcu_id_weight(word_id, word_weight):
    '''
    :param word_id:  词与id的字典
    :param word_weight: 词与权重的字典
    :return: id与权重的字典
    '''
    id_weight = {}
    for word,ind in word_id.items():
        if word in word_weight:
            id_weight[ind] = word_weight[word]
        else:
            id_weight[ind] = 1.0
    return id_weight

def calcu_sents_id(sentences,word_id):
    '''
    :param sentences: n条句子组成的列表
    :param word_id: 词与id的字典
    :return: n条句子中词的id
    '''
    sents_id = []
    for sent in sentences:
        sent_id = [word_id[word] for word in sent if word in word_id]
        if sent_id:
            sents_id.append(sent_id)
    sents_id_array = array_pad(sents_id)
    return sents_id_array

def array_pad(id_lists):
    '''
    在获得n条句子中词的id时，将不同长度的句子统一长度，长度为其中最长句的长度。
    由于0本身就是词的id，所以用-1来填充其他句子。
    :param id_lists:n条句子的词的id
    :return: pad后统一长度的id的array数组
    '''
    lengths = [len(ids) for ids in id_lists]
    n_samples = len(id_lists)
    maxlen = np.max(lengths)
    id_pad_array = np.zeros((n_samples, maxlen),dtype=np.int32) - 1
    for ind, ids in enumerate(id_lists):
        id_pad_array[ind,:lengths[ind]] = ids
    return id_pad_array

def calcu_sents_weight(sents_id, id_weight):
    '''
    根据句子中词的id的array数组，得到句子中词的权重的array数组。
    同样要统一长度，用0来填充。
    :param sents_id: 句子中词的id的array数组
    :param id_weight: 提前准备好的id和权重对应的字典
    :return: 句子中词的权重的array数组
    '''
    sents_weight_array = np.zeros(sents_id.shape)
    for i in range(sents_id.shape[0]):
        for j in range(sents_id.shape[1]):
            if sents_id[i,j] >= 0:
                sents_weight_array[i,j] = id_weight[sents_id[i,j]]
    return sents_weight_array


def topk_by_score(sentences,scores,topk):
    '''
    根据平滑后的句子分数，取分数最高的前几个作为摘要，同时保持句子原来的顺序。
    :param sentences: 划分好的n条句子
    :param scores: n条句子的平滑分数
    :param topk: 要取多少条摘要
    :return: 新闻的摘要
    '''
    topk_score = sorted(scores,reverse=True)[topk-1]
    pairs = zip(sentences,scores)
    sentences_topk = [sent for sent,score in pairs if score >= topk_score]
    return sentences_topk

if __name__ == '__main__':
    # 测试提取双引号句子的正则。
    title = '使这个给分点更具操作性'
    news = '细化评分细则。更具操作性。“每个人”细化评分细则。“做一遍。再结合标准答案 ”。使这个更具操作性。“做一遍”使这个更具操作性。“做一遍之后再结合标准答案 。” 细化评分细则？'
    prepare_sentences(title, news)