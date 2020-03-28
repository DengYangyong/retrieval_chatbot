#coding:utf-8
from wxpy import *
from server import chat_service
import random,re

bot = Bot(cache_path=True)
#my_friend = bot.friends().search("Katherine 石萌多吃会变丑")[0]
#print(my_friend)

#@bot.register(my_friend)
@bot.register(chats=[Friend])
def auto_response(msg):
    print("[接收]:"+str(msg))
    print("[格式]:"+str(msg.type))

    if msg.type == "Picture":
        return "邓小帅不会斗图也不会发表情包哦！[奸笑]"

    if msg.type == "Recording":
        return "邓小帅没带耳机呢[皱眉]打字行吗？"

    if msg.type == "Video":
        return "你发邓紫棋的MV我才会看哦！[吃瓜]"

    query = str(msg).split(':')[1]
    query = re.sub("(Text)","",query).strip()
    print("[接收]:"+query)
    if query in ["[捂脸]","[皱眉]","[奸笑]","[旺柴]","[微笑]","[撇嘴]","[发呆]","[流泪]","[尴尬]","[偷笑]","[奋斗]","[抠鼻]","[坏笑]","[吃瓜]","[呲牙]","[耶]","[Emm]","[社会社会]","[嘿哈]"]:
        return random.choice(["[捂脸]","[皱眉]","[奸笑]","[旺柴]","[微笑]","[偷笑]","[坏笑]","[吃瓜]","[社会社会]"])
    
    if query in ["小帅","邓邓","邓邓最棒","邓杨勇","杨勇","邓阳勇","杨杨","杨","阳勇","勇哥","小邓","阿勇","勇仔"]:
        return "邓小帅在呢，您说吧。"
    
    if query in ["。。。","...","......","。。。。。。","？","？？？"]:
        return "无语了是吧？我比你更无语呢。"
        
    ret = chat_service(query,6060)
    print("[发送]:"+str(ret))
    return ret

embed()
