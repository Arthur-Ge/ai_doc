import torch
from datasets import load_from_disk
from transformers import BertTokenizer
from bilstm_crf import NER

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def evaluate(model=None, tokenizer=None, data=None):
    if data is None:
        # 读取测试数据
        data = load_from_disk('./ner_data/bilstm_crf_data_aidoc')['valid']

    # 1. 计算各个不同类别总实体数量

    # 计算测试集实体数量
    total_entities = {'DIS': [], 'SYM': []}

    # indicators
    indicators = []

    def calculate_handler(data_inputs, data_labels):
        # 将 data_inputs 转换为没有空格隔开的句子
        data_inputs = ''.join(data_inputs.split())

        # 提取句子中的实体
        extract_entities = extract_decode(data_labels, data_inputs)
        # 统计每种实体的数量
        nonlocal total_entities
        for key, value in extract_entities.items():
            total_entities[key].extend(value)

    # 统计不同实体的数量
    data.map(calculate_handler, input_columns=['data_inputs', 'data_labels'])
    # print(total_entities)

    # 2. 计算模型预测的各个类别实体数量
    if model is None:
        model_param = torch.load('./ner_data/model/BiLSTM-CRF-700.bin')
        model = NER(**model_param['init']).cuda(device)
        model.load_state_dict(model_param['state'])

    # 构建分词器
    if tokenizer is None:
        tokenizer = BertTokenizer(vocab_file='./ner_data/bilstm_crf_vocab_aidoc.txt')

    model_entities = {'DIS': [], 'SYM': [], }

    def start_evaluate(data_inputs):

        # 对输入文本进行分词
        model_inputs = tokenizer.encode(data_inputs, add_special_tokens=False, return_tensors='pt')[0]
        model_inputs = model_inputs.to(device)
        # 文本送入模型进行计算
        with torch.no_grad():
            label_list = model.predict(model_inputs)

        # 统计预测的实体数量
        text = ''.join(data_inputs.split())

        # 从预测结果提取实体名字
        extract_entities = extract_decode(label_list, text)
        nonlocal model_entities

        for key, value in extract_entities.items():
            model_entities[key].extend(value)

    # 统计预测不同实体的数量
    data.map(start_evaluate, input_columns=['data_inputs'], batched=False)
    # print(model_entities)

    # 3. 统计每个类别的召回率
    total_pred_correct = 0
    total_true_correct = 0
    for key in total_entities.keys():

        # 获得当前 key 类别真实和模型预测实体列表
        true_entities = total_entities[key]
        true_entities_num = len(true_entities)
        pred_entities = model_entities[key]
        pred_entities_num = len(pred_entities)

        # 分解预测实体中，pred_correct 表示预测正确，pred_incorrect 表示预测错误
        pred_correct, pred_incorrect = 0, 0
        for pred_entity in pred_entities:
            if pred_entity in true_entities:
                pred_correct += 1
                continue
            pred_incorrect += 1

        # 计算共预测正确多少个实体
        total_pred_correct += pred_correct
        # 计算共有多少个真实的实体
        total_true_correct += true_entities_num

        # 计算精度
        # 精确率：预测结果为正例样本中真实为正例的比例
        # 召回率：真实为正例的样本中预测结果为正例的比例
        recall = pred_correct / true_entities_num
        precision = pred_correct / pred_entities_num
        f1 = 0
        if recall != 0 or precision != 0:
            f1 = 2 * precision * recall / (precision + recall)

        print(key, '查全率：%.3f' % recall)
        print(key, '查准率：%.3f' % precision)
        print(key, 'f1: %.3f' % f1)
        print('-' * 50)
        indicators.extend([recall, precision, f1])

    print('准确率：%.3f' % (total_pred_correct / total_true_correct))
    indicators.append(total_pred_correct / total_true_correct)

    return indicators


def extract_decode(label_list, text):
    """
    :param label_list: 模型输出的包含标签序列的一维列表
    :param text: 模型输入的句子
    :return: 提取到的实体名字
    """

    label_to_index = {"O": 0, "B-dis": 1, "I-dis": 2, "B-sym": 3, "I-sym": 4}
    B_DIS, I_DIS = label_to_index['B-dis'], label_to_index['I-dis']
    B_SYM, I_SYM = label_to_index['B-sym'], label_to_index['I-sym']

    # 提取连续的标签代表的实体
    def extract_word(start_index, next_label):

        # index 表示最后索引的位置
        index, entity = start_index + 1, [text[start_index]]
        for index in range(start_index + 1, len(label_list)):
            if label_list[index] != next_label:
                break
            entity.append(text[index])

        return index, ''.join(entity)

    # 存储提取的命名实体
    extract_entites, index = {'DIS': [], 'SYM': []}, 0
    # 映射下一个持续的标签
    next_label = {B_DIS: I_DIS, B_SYM: I_SYM}
    # 映射词的所属类别
    word_class = {B_DIS: 'DIS', B_SYM: 'SYM'}

    while index < len(label_list):
        # 获得当前位置的标签
        label = label_list[index]
        if label in next_label.keys():
            # 将当前位置和对应的下一个持续标签传递到 extract_word 函数
            index, word = extract_word(index, next_label[label])
            extract_entites[word_class[label]].append(word)
            continue
        index += 1

    return extract_entites


if __name__ == '__main__':
    evaluate()
