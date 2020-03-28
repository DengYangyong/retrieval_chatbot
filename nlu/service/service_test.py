# coding=utf8
import os,sys
import json
import socket
import time
import urllib.request
from datetime import timedelta
sys.path.append("../textcnn")
from textcnn_config import TextcnnConfig
from recall_rank import Rank
import pandas as pd

config = TextcnnConfig()


""" 记录花费的时间 """
def get_time_dif(start_time):
    end_time = time.time()
    time_dif = end_time - start_time 
    return time_dif          

""" 测试服务的响应时间 """
def test_service(content, port):
    
    url = 'http://0.0.0.0:{}/QA'.format(port)
    app_data = {"request_id": "QAServer", "query": content}
    
    """ 转化为json格式 """
    app_data=json.dumps(app_data).encode("utf-8")
    
    start_time = time.time()
    req = urllib.request.Request(url, app_data)
    try:
        """ 调用服务，得到结果 """
        response = urllib.request.urlopen(req)
        response = response.read().decode("utf-8")
        
        """ 从json格式中解析出来 """
        response = json.loads(response)
    except Exception as e:
        print(e)
        response = None
        
    """ 打印耗时 """
    time_usage = get_time_dif(start_time)
    print("Time usage: {}".format(time_usage))
    print(response)
    return time_usage
    
def calcu_accu():
    
    """ 读取数据 """
    print("\nLoading the dataset ... \n")
    corpus_xls = pd.ExcelFile(config.corpus_path)
    busi_df = corpus_xls.parse("business_question")
    chat_df = corpus_xls.parse("chatting_question")
    busi_df.dropna(inplace=True)
    chat_df.dropna(inplace=True)
    
    ranker = Rank()
    
    busi_ques = busi_df["question"].tolist()
    busi_pred = [ranker._get_top_one(ques)[0] for ques in busi_ques]
    
    busi_accu = sum([int(q1 == q2) for q1,q2 in zip(busi_ques,busi_pred)]) / len(busi_ques)
    print("\n业务类问题的准确率为:{:.6f}\n".format(busi_accu))

    chat_ques = chat_df["question"].tolist()
    chat_pred = [ranker._get_top_one(ques)[0] for ques in chat_ques]    
    chat_accu = sum([int(q1 == q2) for q1,q2 in zip(chat_ques,chat_pred)]) / len(chat_ques)
    print("\n闲聊类问题的准确率为:{:.6f}\n".format(chat_accu))
    

if __name__=='__main__':
    
    """ 测试1000次，得到平均响应时间 """
    time_usage = 0
    for i in range(100):
        content = "你有男朋友吗"
        time_usage += test_service(content, 6060)
    print("Time usage average is {}".format( time_usage / 100))
    #calcu_accu()



