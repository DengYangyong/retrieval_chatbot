# !/usr/bin/env python
# -*- coding:utf-8 -*-

import random

suggestRule = """
sentence_grammer = hello sorry reason suggest
hello = 亲, | 您好, | 你好, | emm... | 这个... | 我想想啊...
sorry = 对不起, | 实在抱歉, | 不好意思, | 抱歉,
reason = name 还在学习中,暂时理解不了你说的问题。 | name 还只是一个0岁的孩子，不太明白您的意思。 | name 目前还理解不了你的问题。
name = 邓小帅 | 我
suggest = name 可以为您提供这些服务： service | name 可以给您提供这些帮助哟： service
service = 存款/取款/转账/理财/挂失银行卡/补办银行卡/办理银行卡
"""

def generation_answer_by_rule(grammer_str:str,target='sentence_grammer',stmt_split='=',expr_split='|'):
    """
	按照suggestRule随机生成最后的回答
    """
    rules = dict()
    for line in grammer_str.split('\n'):
        if not line:continue
        stmt,expr = line.split(stmt_split)
        rules[stmt.strip()] = expr.split(expr_split)
        
    answer = generation(rules,target=target)
        
    return answer

def generation(rules,target):
    if target in rules:
        candidates = rules[target]
        candidate = random.choice(candidates)
        return ''.join(generation(rules,target=part.strip()) for part in candidate.split()) 
    else:
        return target
    
def suggest_answer():
    
    return generation_answer_by_rule(suggestRule)



if __name__ == "__main__":
    
    print(suggest_answer())

