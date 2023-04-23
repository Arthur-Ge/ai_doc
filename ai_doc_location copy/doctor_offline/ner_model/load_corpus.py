import pandas as pd
import json

def load_corpus():
    # 定义训练数据集和验证数据集的路径
    train_data_file_path = './ner_data/train.txt'
    validate_file_path = './ner_data/validate.txt'

    data_inputs, data_labels = [], []

    # 因为每行都是一个样本，所以按行遍历即可
    for line in open(train_data_file_path, mode='r', encoding='utf8'):
        # 每行样本数据都是json字符串，可以直接进行loads,  然后追加进结果列表中
        data = json.loads(line)
        # print(type(data))
        data_inputs.append(' '.join(data['text']))
        data_labels.append(' '.join(data['label']))
    train_data_df = pd.DataFrame()
    train_data_df['data_inputs'] = data_inputs
    train_data_df['data_labels'] = data_labels
    train_data_df.to_csv('./ner_data/01-训练集_aidoc.csv')
    print('训练集数据量：', len(train_data_df))

    data_inputs, data_labels = [], []
    for line in open(validate_file_path, mode='r', encoding='utf8'):
        data = json.loads(line)
        data_inputs.append(' '.join(data['text']))
        data_labels.append(' '.join(data['label']))
    # 存储测试集数据
    test_data_df = pd.DataFrame()
    test_data_df['data_inputs'] = data_inputs
    test_data_df['data_labels'] = data_labels
    test_data_df.to_csv('./ner_data/02-测试集_aidoc.csv')
    print('测试集数据量：', len(test_data_df))

if __name__ == '__main__':
    load_corpus()
