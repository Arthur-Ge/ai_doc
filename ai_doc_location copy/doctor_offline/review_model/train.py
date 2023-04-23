import random
import torch
import time
import math
import torch.nn as nn
import pandas as pd
from collections import Counter
from  bert_chinese_encode import get_bert_encode_for_single
from  RNN_MODEL import RNN

#构建随机选取数据函数
def randomTrainingExample(train_data):
    """
    随机选取数据函数
    :param train_data:训练集列表形式数据
    :return:category, line, category_tensor, line_tensor
    """
    #从train_data里面随机选取一条数据
    category,line = random.choice(train_data)

    #将里面的文字通过bert与训练模型进行编码，获取编码后的tensor类型数据
    line_tensor = get_bert_encode_for_single(line)

    #将类别标签封装成tensor
    category_tensor = torch.tensor([int(category)])

    #返回四个结果
    return category,line,category_tensor,line_tensor

#构建模型训练函数
# 选取损失函数为NLLLoss()
# CrossEntropyLoss是Softmax–Log–NLLLoss,这里直接计算NLLLose
criterion = nn.NLLLoss()
# 学习率为0.005
learning_rate = 0.005
def train(category_tensor,line_tensor):
    """
    模型训练函数
    :param category_tensor: tensor类型的类别
    :param line_tensor: tensor类型的line
    :return: 返回结果和损失值
    """
    #初始化隐藏层
    hidden = rnn.initHidden()
    #模型梯度归0
    rnn.zero_grad()
    #遍历line_tensor中的每一个字的张量表示
    for i in range(line_tensor.size()[1]):
        # 然后将其输入到rnn模型中，因为模型要求是输入必须是二维张量，因此需要拓展一个维度，循环调用rnn直到最后一个字
        output, hidden = rnn(line_tensor[0][i].unsqueeze(0), hidden)
        # 根据损失函数计算损失，输入分别是rnn的输出结果和真正的类别标签
    loss = criterion(output, category_tensor)
    # 将误差进行反向传播
    loss.backward()

    # 更新模型中所有的参数
    for p in rnn.parameters():
        # 将参数的张量表示与参数的梯度乘以学习率的结果相加以此来更新参数
        p.data.add_(-learning_rate, p.grad.data)

    # 返回结果和损失的值
    return output, loss.item()

#构建模型验证函数
def valid(category_tensor, line_tensor):
    """模型验证函数，category_tensor代表类别张量，line_tensor代表编码后的文本张量"""
    # 初始化隐层
    hidden = rnn.initHidden()
    # 验证模型不自动求解梯度
    with torch.no_grad():
        # 遍历line_tensor中的每一个字的张量表示
        for i in range(line_tensor.size()[1]):
            # 然后将其输入到rnn模型中，因为模型要求是输入必须是二维张量，因此需要拓展一个维度，循环调用rnn直到最后一个字
            output, hidden = rnn(line_tensor[0][i].unsqueeze(0), hidden)
        # 获得损失
        loss = criterion(output, category_tensor)
     # 返回结果和损失的值
    return output, loss.item()

# 构建时间计算函数
def timeSince(since):
    "获得每次打印的训练耗时，since是训练开始时间"
    # 获得当前时间
    now = time.time()
    # 获得时间差，就是训练耗时
    s = now - since
    # 将秒转化为分钟，并取整
    m = math.floor(s / 60)
    # 计算剩下不够凑成1分钟的秒数
    s -= m * 60
    # 返回指定格式的耗时
    return '%dm %ds' % (m, s)




if __name__ == '__main__':
    # 读取数据
    train_data_path = './train_data.csv'
    train_data = pd.read_csv(train_data_path, header=None, sep='\t')

    # 打印数据正负标签比例
    #print(dict(Counter(train_data[0].values)))

    # 转换数据到列表形式
    train_data = train_data.values.tolist()
    #print(train_data[:10])

    # 实例化RNN的
    input_size = 768
    hidden_size = 128
    n_categories = 2
    rnn = RNN(input_size,hidden_size,n_categories)

    # 设置迭代次数为50000步
    n_iters = 50000

    # 打印间隔为1000步
    plot_every = 1000

    # 初始化打印间隔中训练和验证的损失和准确率
    train_current_loss = 0
    train_current_acc = 0
    valid_current_loss = 0
    valid_current_acc = 0

    # 初始化存储每次打印间隔的平均损失和准确率的列表
    all_train_losses = []
    all_train_acc = []
    all_valid_losses = []
    all_valid_acc = []

    # 获取开始时间戳
    start = time.time()

    # 循环遍历n_iters次
    for iter in range(1, n_iters + 1):
        # 调用两次随机函数分别生成一条训练和验证数据
        category, line, category_tensor, line_tensor = randomTrainingExample(train_data)
        category_, line_, category_tensor_, line_tensor_ = randomTrainingExample(train_data)
        # 分别调用训练和验证函数，获得输出和损失
        train_output, train_loss = train(category_tensor, line_tensor)
        valid_output, valid_loss = valid(category_tensor_, line_tensor_)
        # 进行训练损失，验证损失，训练准确率和验证准确率分别累加
        train_current_loss += train_loss
        train_current_acc += (train_output.argmax(1) == category_tensor).sum().item()
        valid_current_loss += valid_loss
        valid_current_acc += (valid_output.argmax(1) == category_tensor_).sum().item()
        # 当迭代次数是指定打印间隔的整数倍时
        if iter % plot_every == 0:
            # 用刚刚累加的损失和准确率除以间隔步数得到平均值
            train_average_loss = train_current_loss / plot_every
            train_average_acc = train_current_acc / plot_every
            valid_average_loss = valid_current_loss / plot_every
            valid_average_acc = valid_current_acc / plot_every
            # 打印迭代步，耗时，训练损失和准确率，验证损失和准确率
            print("Iter:", iter, "|", "TimeSince:", timeSince(start))
            print("Train Loss:", train_average_loss, "|", "Train Acc:", train_average_acc)
            print("Valid Loss:", valid_average_loss, "|", "Valid Acc:", valid_average_acc)
            # 将结果存入对应的列表中，方便后续制图
            all_train_losses.append(train_average_loss)
            all_train_acc.append(train_average_acc)
            all_valid_losses.append(valid_average_loss)
            all_valid_acc.append(valid_average_acc)
            # 将该间隔的训练和验证损失及其准确率归0
            train_current_loss = 0
            train_current_acc = 0
            valid_current_loss = 0
            valid_current_acc = 0
    # 绘制训练和验证的损失和准确率对照曲线
    import matplotlib.pyplot as plt

    plt.figure(0)
    plt.plot(all_train_losses, label="Train Loss")
    plt.plot(all_valid_losses, color="red", label="Valid Loss")
    plt.legend(loc='upper left')
    plt.savefig("./loss.png")

    plt.figure(1)
    plt.plot(all_train_acc, label="Train Acc")
    plt.plot(all_valid_acc, color="red", label="Valid Acc")
    plt.legend(loc='upper left')
    plt.savefig("./acc.png")

    # 保存路径
    MODEL_PATH = './BERT_RNN.pth'
    # 保存模型参数
    torch.save(rnn.state_dict(), MODEL_PATH)