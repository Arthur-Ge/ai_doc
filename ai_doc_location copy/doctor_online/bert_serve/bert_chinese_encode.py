import torch
import torch.nn as nn

#print(torch.hub.get_dir()) #查看模型的默认存储路径
# root用户从本地加载
source = '/Users/geqirui/.cache/torch/hub/huggingface_pytorch-transformers_main'
# 直接使用预训练的bert中文模型
model_name = 'bert-base-chinese'
# 通过torch.hub获得已经训练好的bert-base-chinese模型
model =  torch.hub.load(source, 'model', model_name, source='local')
# 获得对应的字符映射器，它将把中文的每个字映射成一个数字
tokenizer = torch.hub.load(source, 'tokenizer', model_name, source='local')
# 指定本地路径
#model_name = 'bert-base-chinese'
#source = 'huggingface/pytorch-transformers'
#model =  torch.hub.load(source, 'model', model_name, source='github')#转成向量
#tokenizer = torch.hub.load(source, 'tokenizer', model_name, source='github')#转成数字


def get_bert_encode(text_1, text_2, mark=102, max_len=10):
    """
    description: 使用bert中文模型对输入的文本对进行编码
    :param text_1: 代表输入的第一句话
    :param text_2: 代表输入的第二句话
    :param mark: 分隔标记，是预训练模型tokenizer本身的标记符号，当输入是两个文本时，
                 得到的index_tokens会以102进行分隔
    :param max_len: 文本的允许最大长度，也是文本的规范长度即大于该长度要被截断，小于该长度要进行0补齐
    :return 输入文本的bert编码
    """
    # 使用tokenizer的encode方法对输入的两句文本进行字映射。
    indexed_tokens = tokenizer.encode(text_1, text_2)
    # 准备对映射后的文本进行规范长度处理即大于该长度要被截断，小于该长度要进行0补齐
    # 所以需要先找到分隔标记的索引位置
    k = indexed_tokens.index(mark)
    # 首先对第一句话进行长度规范因此将indexed_tokens截取到[:k]判断
    if len(indexed_tokens[:k]) >= max_len:
        # 如果大于max_len, 则进行截断
        indexed_tokens_1 = indexed_tokens[:max_len]
    else:
        # 否则使用[0]进行补齐，补齐的0的个数就是max_len-len(indexed_tokens[:k])
        indexed_tokens_1 = indexed_tokens[:k] + (max_len-len(indexed_tokens[:k]))*[0]

    # 同理下面是对第二句话进行规范长度处理，因此截取[k:]
    if len(indexed_tokens[k:]) >= max_len:
        # 如果大于max_len, 则进行截断
        indexed_tokens_2 = indexed_tokens[k:k+max_len]
    else:
         # 否则使用[0]进行补齐，补齐的0的个数就是max_len-len(indexed_tokens[:k])
        indexed_tokens_2 = indexed_tokens[k:] + (max_len-len(indexed_tokens[k:]))*[0]

    # 最后将处理后的indexed_tokens_1和indexed_tokens_2再进行相加
    indexed_tokens = indexed_tokens_1 + indexed_tokens_2
    # 为了让模型在编码时能够更好的区分这两句话，我们可以使用分隔ids,
    # 它是一个与indexed_tokens等长的向量，0元素的位置代表是第一句话
    # 1元素的位置代表是第二句话，长度都是max_len
    segments_ids = [0]*max_len + [1]*max_len
    # 将segments_ids和indexed_tokens转换成模型需要的张量形式
    segments_tensor = torch.tensor([segments_ids])
    tokens_tensor = torch.tensor([indexed_tokens])
    # 模型不自动求解梯度
    with torch.no_grad():
        # 使用bert model进行编码，传入参数tokens_tensor和segments_tensor得到encoded_layers
        encoded_layers = model(tokens_tensor, token_type_ids=segments_tensor)
    return encoded_layers[0]
