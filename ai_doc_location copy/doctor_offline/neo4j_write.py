# 导包
import os
import fileinput
from neo4j import GraphDatabase

# 设置neo4j图数据库的配置信息
NEO4J_CONFIG = {
	"uri": "bolt://localhost:7687",
	"auth": ("neo4j", "12345678"),
	"encrypted": False
}

driver = GraphDatabase.driver(**NEO4J_CONFIG) # 两个星号对字典进行解包，key就是参数名

# 1 读取数据
def _load_data(path):
    """
    先获取文件名称，也就是疾病名称
    获取每个文件的内容，也就是症状
    构造成字典：{疾病1: [症状1, 症状2, ...], 疾病2: [症状1, 症状2, ...]}
    :param path: 文件存储的路径
    :return: 构造的字典
    """
    # 获取疾病csv列表
    disesase_csv_list = os.listdir(path)
    # print('disesase_csv_list:', disesase_csv_list)
    # 去掉.csv后缀，得到疾病列表 xxx.csv
    disesase_list = list(map(lambda x: x.split('.')[0], disesase_csv_list))
    # print(disesase_list)

    # 初始化一个列表 存放每种疾病对应的症状的列表
    symptom_list = []

    # 遍历每个csv
    for disesase_csv in disesase_csv_list:
        symptom = list(map(lambda x:x.strip(), fileinput.FileInput(os.path.join(path, disesase_csv))))
        symptom = list(filter(lambda x: 0 < len(x) < 100, symptom))
        symptom_list.append(symptom)

    # 转成字典，返回
    print(dict(zip(disesase_list, symptom_list)))
    return dict(zip(disesase_list, symptom_list))


# 2 把数据写入到neo4j中
def neo4j_write(path):
    """
    把数据写入到neo4j中
    :param path: 存放的是通过审核模型审核完成的疾病的名称和对应的症状文件
    :return:
    """
    # 保存上一步返回的字典
    disease_symptom_dict = _load_data(path)

    # 开启会话，写入数据到neo4j中
    with driver.session() as session:
        for key, values in disease_symptom_dict.items():
            # key 就是疾病名称
            # values 是一个列表，key这个疾病对应的多种症状
            # 创建疾病节点
            cypher = "MERGE (a:Disease{name:%r}) RETURN a" % key
            session.run(cypher)

            for value in values:
                # 创建症状节点
                cypher = "MERGE (b:Symptom{name:%r}) RETURN b" % value
                session.run(cypher)

                # 创建疾病和症状的关系
                cypher = "MATCH (a:Disease{name:%r}) MATCH(b:Symptom{name:%r}) WITH a, b MERGE(a)-[r:dis_to_sym]-(b)" % (key, value)
                session.run(cypher)

        cypher = "CREATE INDEX FOR (a:Disease) ON (a.name)"
        session.run(cypher)
        cypher = "CREATE INDEX FOR (b:Symptom) ON (b.name)"
        session.run(cypher)


if __name__ == '__main__':
    neo4j_write('./structured_unstructured_data/structured/reviewed/') # 使用相对路径