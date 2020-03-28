#coding:utf-8
import numpy as np
from sklearn.decomposition import TruncatedSVD

def calcu_weighted_average(embeds, ids, weights):
    '''
    根据n条句子的词id和词权重，对句子中的词的向量进行加权，计算的n条句子的句向量。
    :param embeds: id和词向量对应的字典，根据词id取词向量
    :param ids: n条句子中词id的array数组
    :param weights: n条句子中词权重的array数组
    :return:
    '''
    n_samples = ids.shape[0]
    weighted_embeds = np.zeros((n_samples, embeds[0].shape[0]))
    for i in range(n_samples):
        embeds_i = np.array([embeds[ind] for ind in ids[i,:] if ind >= 0])
        weights_i = weights[i,:embeds_i.shape[0]]
        weighted_embeds[i,:] = weights_i.dot(embeds_i) / embeds_i.shape[0]
    return weighted_embeds

def compute_pc(X,npc=1):
    '''
    计算矩阵的主成分
    :param X: 句子的向量矩阵
    :param npc: 主成分的个数
    :return: 主成分
    '''
    # svd是计算主成分的一种比较好的方法，所以选择svd来计算
    svd = TruncatedSVD(n_components=npc, n_iter=7, random_state=0)
    svd.fit(X)
    return svd.components_

def remove_pc(X, rmpc=1):
    '''
    句向量矩阵减去一个主成分，得到最终的sif句向量
    :param X: 句向量矩阵
    :param rmpc: 主成分个数
    :return:  sif句向量矩阵
    '''
    pc = compute_pc(X,rmpc)
    if rmpc == 1:
        X_new = X - X.dot(pc.transpose()) * pc
    else:
        X_new = X - X.dot(pc.transpose()).dot(pc)
    return X_new

def sif_embedding(id_embed, sents_id, sents_weight):
    '''
    计算句子的加权向量，并且减去一个主成分，得到最终的sif句向量
    :param id_embed:  词与id对应的字典
    :param sents_id:  sents_id[i, :]是第i句的所有词的id
    :param sents_weight: sents_weight[i, :]是第i句的所有词的权重
    :return: sif句向量
    '''
    embeds_weighted = calcu_weighted_average(id_embed, sents_id, sents_weight)
    embeds_sif = remove_pc(embeds_weighted)
    return embeds_sif
