#coding:utf-8
import torch
from torchtext import data
import jieba,re
from textcnn_config import TextcnnConfig
import pickle

config = TextcnnConfig()

""" 加载pickle对象 """ 
def load_pickle(file_path):
    with open(file_path,'rb') as f:
        s = pickle.load(f)
    return s

""" 进行文本清洗，并用jieba分词 """
def clean_seg(text):
    line = re.sub(
            "[\s+\-\|\!\/\[\]\{\}_,.$%^*(+\"\')]+|[:：+——()?【】《》“”！，。？、~@#￥%……&*（）]+", '',str(text))
    words = jieba.lcut(line, cut_all=False)
    return words


""" 对文本进行pad，并转化为id """
def pad_proc(words,max_len,vocab):
    words = words[:max_len]
    words_sub = [word if word in vocab.stoi else '<unk>'for word in words]
    words_pad = words_sub + ['<pad>'] * (max_len - len(words))
    return words_pad

    
class TextcnnPredict:
    def __init__(self):
        self.vocab = load_pickle(config.vocab_path)
        self.model = torch.load(config.save_path,map_location="cpu")
        self.model.eval()

    """把评论转化为id，并进行pad"""
    def content_to_ids(self,words):
        words_pad = pad_proc(words,config.max_lengths, self.vocab)
        ids = [self.vocab.stoi[w] for w in words_pad]
        return ids  
    
    """进行预测"""
    def predict(self,content):
        
        words = clean_seg(content)
        if not words:
            return "chat"        
        
        with torch.no_grad():
            content_ids = self.content_to_ids(words)
            ids_tensor = torch.LongTensor([content_ids])
            outputs = self.model(ids_tensor)
            class_ = torch.argmax(outputs).item()  
            question_type = config.class_map_reverse[int(class_)]
        return question_type
    
if __name__ == "__main__":
    
    model = TextcnnPredict()
    
    for question in ["其他保险","社保业务","存单业务","查询外汇汇率","你家乡在哪里","你能讲话吗","你有男朋友吗","你觉得自己长得怎么样"]:
        question_type = model.predict(question)
        print("Question type: {}\n".format(question_type))