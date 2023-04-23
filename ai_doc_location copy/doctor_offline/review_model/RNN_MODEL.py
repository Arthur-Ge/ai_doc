import torch
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        '''
        :param input_size: 输入张量最后一个维度的大小
        :param hidden_size: 隐藏层张量最后一个维度的大小
        :param output_size: 输出张量最后一个维度的大小
        '''
        super(RNN, self).__init__()

        # 将隐藏层的大小写成类的内部变量
        self.hidden_size = hidden_size

        # 构建第一个线性层，输入尺寸是input_size+hidden_size, 因为真正进入全连接层的张量是X(t)+h(t-1)
        # 输出尺寸是hidden_size
        self.i2h= nn.Linear(input_size+hidden_size, hidden_size)

        # tanh
        self.tanh = nn.Tanh()

        # 构建第二个线性层，输入尺寸是hidden_size
        # 输出尺寸是output_size
        self.i2o = nn.Linear(hidden_size, output_size)

        # 定义最终输出的softmax处理层
        self.softmax = nn.LogSoftmax(dim=-1)#输出对应每个标签的概率

    def forward(self, input1, hidden1):
        '''
        :param input1: 相当与x(t)
        :param hidden1: 相当于h(t-1)
        :return:
        '''
        # 首先要进行输入张量的拼接，将x(t)和h(t-1)拼接在一起
        combined = torch.cat((input1, hidden1), 1)

        # 让输入经过隐藏层的获得hidden
        hidden = self.i2h(combined)

        # tanh层
        hidden = self.tanh(hidden)
        # print('hidden.shape:', hidden.shape)
        # 让输入经过输出层获得output
        output = self.i2o(hidden)

        # 让output经过softmax层
        output = self.softmax(output)
        # 返回两个张量output, hidden
        return output, hidden

    def initHidden(self):
        # 将隐藏层初始化为一个[1, hidden_size]的全零张量
        return torch.zeros(1, self.hidden_size)

if __name__ == '__main__':
    input_size = 768
    hidden_size = 128
    n_categories = 2 #因为是二分类

    input = torch.rand(1, input_size)
    hidden = torch.rand(1, hidden_size)

    rnn = RNN(input_size,hidden_size,n_categories)

    outputs, hidden = rnn(input, hidden)
    print("outputs:", outputs)
    print("hidden:", hidden)