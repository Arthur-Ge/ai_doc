# 服务框架使用Flask
# 导入必备的工具
from flask import Flask
from flask import request
app = Flask(__name__)

# 导入发送http请求的requests工具
import requests

# 导入操作redis数据库的工具
import redis

# 导入加载json文件的工具
import json

# 导入已写好的Unit API调用文件
from unit import unit_chat

# 导入操作neo4j数据库的工具
from neo4j import GraphDatabase

# 从配置文件中导入需要的配置
# NEO4J的连接配置
from config import NEO4J_CONFIG
# REDIS的连接配置
from config import REDIS_CONFIG
# 句子相关模型服务的请求地址
from config import model_serve_url
# 句子相关模型服务的超时时间
from config import TIMEOUT
# 规则对话模版的加载路径
from config import reply_path
# 用户对话信息保存的过期时间
from config import ex_time

# 建立REDIS连接池
pool = redis.ConnectionPool(**REDIS_CONFIG)

# 初始化NEO4J驱动对象
_driver = GraphDatabase.driver(**NEO4J_CONFIG)

def query_neo4j(text):
    """
    description: 根据用户对话文本中的可能存在的症状查询图数据库。
    :param text: 用户的输入文本。症状名称
    :return: 用户描述的症状对应的疾病列表。
    """
    # 开启一个session操作图数据库
    with _driver.session() as session:
         # cypher语句，匹配句子中存在的所有症状节点，
         # 保存这些节点并逐一通过关系dis_to_sym进行对应病症的查找，返回找到的疾病名字列表。
        cypher = "MATCH(a:Symptom) WHERE(%r contains a.name) WITH \
                  a MATCH(a)-[r:dis_to_sym]-(b:Disease) RETURN b.name LIMIT 5" %text
        # 运行这条cypher语句
        record = session.run(cypher)
        # 从record对象中获得结果列表
        result = list(map(lambda x: x[0], record))
    return result
class Handler(object):
    """主要逻辑服务的处理类"""
    def __init__(self, uid, text, r, reply):
        """
        :param uid: 用户唯一标示uid
        :param text: 该用户本次输入的文本
        :param r: redis数据库的连接对象
        :param reply: 规则对话模版加载到内存的对象(字典)
        """
        self.uid = uid
        self.text = text
        self.r = r
        self.reply = reply

    def non_first_sentence(self, previous):
        """
        description: 非首句处理函数
        :param previous: 该用户当前句(输入文本)的上一句文本
        :return: 根据逻辑图返回非首句情况下的输出语句
        """
        # 尝试请求模型服务，若失败则打印错误结果
        try:
            data = {"text1": previous, "text2": self.text}
            result = requests.post(model_serve_url, data=data, timeout=TIMEOUT)
            if not result.text: return unit_chat(self.text)
        except Exception as e:
            print("模型服务异常：", e)
            return unit_chat(self.text)
        # 继续查询图数据库，并获得结果
        s = query_neo4j(self.text)
        # 判断结果为空列表，则直接使用UnitAPI返回
        if not s: return unit_chat(self.text)
        # 若结果不为空，获取上一次已回复的疾病old_disease
        old_disease = self.r.hget(str(self.uid), "previous_d")
        if old_disease:
            # new_disease是本次需要存储的疾病，是已经存储的疾病与本次查询到疾病的并集
            new_disease = list(set(s) | set(eval(old_disease)))
            # res是需要返回的疾病，是本次查询到的疾病与已经存储的疾病的差集
            res = list(set(s) - set(eval(old_disease)))
        else:
            # 如果old_disease为空，则它们相同都是本次查询结果s
            res = new_disease = list(set(s))

        # 存储new_disease覆盖之前的old_disease
        self.r.hset(str(self.uid), "previous_d", str(new_disease))
        # 设置过期时间
        self.r.expire(str(self.uid), ex_time)
        # 将列表转化为字符串，添加到规则对话模版中返回
        if not res:
            return self.reply["4"]
        else:
            res = ",".join(res)
            return self.reply["2"] %res


    def first_sentence(self):
        """首句处理函数"""
        # 直接查询图数据库，并获得结果
        s = query_neo4j(self.text)
        # 判断结果为空列表，则直接使用UnitAPI返回
        if not s: return unit_chat(self.text)
        # 将s存储为"上一次返回的疾病"
        self.r.hset(str(self.uid), "previous_d", str(s))
        # 设置过期时间
        self.r.expire(str(self.uid), ex_time)
        # 将列表转化为字符串，添加到规则对话模版中返回
        res = ",".join(s)
        return self.reply["2"] %res

# 设定主要逻辑服务的路由和请求方法
@app.route('/v1/main_serve/', methods=["POST"])
def main_serve():
    # 接收来自werobot服务的字段
    uid = request.form['uid']
    text = request.form['text']
    # 从redis连接池中获得一个活跃连接
    r = redis.StrictRedis(connection_pool=pool)
    # 根据该uid获取他的上一句话(可能不存在)
    previous = r.hget(str(uid), "previous")
    # 将当前输入的文本设置成上一句
    r.hset(str(uid), "previous", text)
    # 读取规则对话模版内容到内存
    reply = json.load(open(reply_path, "r"))
    # 实例化主要逻辑处理对象
    handler = Handler(uid, text, r, reply)
    # 如果previous存在，说明不是第一句话
    if previous:
        # 调用non_first_sentence方法
        return handler.non_first_sentence(previous)
    else:
        # 否则调用first_sentence()方法
        return handler.first_sentence()
