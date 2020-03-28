# coding:utf-8
from recall_rank import Rank

""" 服务被调用时的情况 """
message_dic={'200':'正常',
             '300':'请求格式错误',
             '400':'模型预测失败'}

class Server:
    def __init__(self):
        """ 
        把模型的预测函数初始化,
        设置使用CPU还是GPU启动服务.
        """
        self.predict = Rank().get_answer
    
    """ 把字典格式的请求数据，解析出来 """
    def parse(self, app_data):
        request_id = app_data["request_id"]
        text = app_data["query"]
        return request_id, text
    
    """ 得到服务的调用结果，包括模型结果和服务的情况 """
    def get_result(self,data):
        code = '200'
        try:
            request_id, text = self.parse(data) 
        except Exception as e:
            print('error info : {}'.format(e))
            code='300'
            request_id = "None"
        try:
            if code == '200':
                answer,ques_type = self.predict(text)
            elif code == '300':
                answer = '亲,对不起,邓小帅目前还理解不了你的问题'
                ques_type = "None"
        except Exception as e:
            print('error info : {}'.format(e))
            answer = '亲,对不起,邓小帅目前还理解不了你的问题'
            ques_type = 'None'
            code='400'
    
        result = {'answer': answer, "question_type":ques_type,'code':code,'message':message_dic[code],'request_id':request_id}  
        return result

if __name__ == "__main__":
    
    server = Server()
    data = {"request_id": "ExamServer", 
            "query" :"你家乡在哪里"}
    print("\n The result is {}".format(server.get_result(data)))
