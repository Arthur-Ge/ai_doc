# 本段代码构建类BiLSTM, 完成初始化和网络结构的搭建
# 总共3层：词嵌入层，双向LSTM层，全连接线性层
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# 定义计算设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class BiLSTM(nn.Module):

    def __init__(self, vocab_size, label_num):
        super(BiLSTM, self).__init__()
        # 用于将输入转换为词向量
        self.embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=256)
        # 用于提取输入的双向语义表示向量
        self.blstm = nn.LSTM(input_size=256,
                             hidden_size=512,
                             bidirectional=True,
                             num_layers=1)
        # 用于将 self.blstm 的输出向量映射为标签 logits
        self.liner = nn.Linear(in_features=1024, out_features=label_num)
    # 参数：句子长度
    # SENTENCE_LENGTH = 20

    def forward(self, inputs, length):

        # 将输入的 token 索引转换为词向量
        outputs_embed = self.embed(inputs)
        # 由于填充了很多0，此处将0进行压缩（因为补0）
        outputs_packd = pack_padded_sequence(outputs_embed, length) #length是真实长度
        # BiLSTM 用于提取双向语义，提取每个句子中的 token 表示
        outputs_blstm, (hn, cn) = self.blstm(outputs_packd)
        # outputs_paded 表示填充后的 BiLSTM 对每个 token 的输出
        # outputs_length 表示每个句子实际的长度
        outputs_paded, output_lengths = pad_packed_sequence(outputs_blstm)
        outputs_paded = outputs_paded.transpose(0, 1)
        # 线性层计算，计算出发射矩阵，形状：(16, 57, 7)
        output_logits = self.liner(outputs_paded)

        outputs = []
        for output_logit, outputs_length in zip(output_logits, output_lengths):
            outputs.append(output_logit[:outputs_length])

        return outputs
    def predict(self, inputs):

        # 将输入的 token 索引转换为词向量
        outputs_embed = self.embed(inputs)
        # 增加一个 batch 维度在 1 位置
        outputs_embed = outputs_embed.unsqueeze(1)
        # 对每个 Token 进行语义表示
        outputs_blstm, (hn, cn) = self.blstm(outputs_embed)
        # 把 1 位置的 batch 值去掉
        outputs_blstm = outputs_blstm.squeeze(1)

        # 计算每个 Token 的发射分数
        output_liner = self.liner(outputs_blstm)

        return output_liner

class CRF(nn.Module):

    def __init__(self, label_num):
        super(CRF, self).__init__()

        # 转移矩阵的标签数量
        self.label_num = label_num
        # [TAG1, TAG2, TAG3...STAR, END]初始化转移矩阵
        params = torch.randn(self.label_num + 2, self.label_num + 2) #加2是在原序列中假如start和end
        self.transition_scores = nn.Parameter(params)#指定为模型需要学习的参数
        # 开始和结束标签
        START_TAG, ENG_TAG = self.label_num, self.label_num + 1
        self.transition_scores.data[:, START_TAG] = -1000 #表示任意标签不可以转到start
        self.transition_scores.data[ENG_TAG, :] = -1000  #表示end不可以转到任意标签上去
        # 定义一个较小值用于扩展发射和转移矩阵时填充
        self.fill_value = -1000.0

        #计算真实路径的分数（单条路径计算）
    def _get_real_path_score(self, emission_score, sequence_label):

        # 计算标签的数量
        seq_length = len(sequence_label)
        # 计算真实路径发射分数
        real_emission_score = torch.sum(emission_score[list(range(seq_length)), sequence_label])
        # 在真实标签序列前后增加一个 start 和 end
        b_id = torch.tensor([self.label_num], dtype=torch.int32, device=device)
        e_id = torch.tensor([self.label_num + 1], dtype=torch.int32, device=device)
        sequence_label_expand = torch.cat([b_id, sequence_label, e_id])
        # 计算真实路径转移分数
        pre_tag = sequence_label_expand[list(range(seq_length + 1))]
        now_tag = sequence_label_expand[list(range(1, seq_length + 2))]
        real_transition_score = torch.sum(self.transition_scores[pre_tag, now_tag])
        # 计算真实路径分数
        real_path_score = real_emission_score + real_transition_score

        return real_path_score


    def _log_sum_exp(self, score):
        # 计算 e 的指数时，每个元素都减去最大值，避免数值溢出
        max_score, _ = torch.max(score, dim=0)
        max_score_expand = max_score.expand(score.shape)
        return max_score + torch.log(torch.sum(torch.exp(score - max_score_expand), dim=0))

    def _expand_emission_matrix(self, emission_score):

        # 计算句子的长度
        sequence_length = emission_score.shape[0]
        # 扩展时会增加 START 和 END 标签，定义该标签的值
        b_s = torch.tensor([[self.fill_value] * self.label_num + [0, self.fill_value]], device=device)
        e_s = torch.tensor([[self.fill_value] * self.label_num + [self.fill_value, 0]], device=device)
        # 扩展发射矩阵为 (sequence_length + 2, label_num + 2)
        expand_matrix = self.fill_value * torch.ones([sequence_length, 2], dtype=torch.float32, device=device)
        emission_score_expand = torch.cat([emission_score, expand_matrix], dim=1)
        emission_score_expand = torch.cat([b_s, emission_score_expand, e_s], dim=0)

        return emission_score_expand

    def _get_total_path_score(self, emission_score):

        # 扩展发射分数矩阵
        emission_score_expand = self._expand_emission_matrix(emission_score)
        # 计算所有路径分数
        pre = emission_score_expand[0]  #表示分别以各个标签结束的每个路径的分数
        for obs in emission_score_expand[1:]:
            # 扩展 pre 维度
            pre_expand = pre.reshape(-1, 1).expand([self.label_num + 2, self.label_num + 2])
            # 扩展 obs 维度
            obs_expand = obs.expand([self.label_num + 2, self.label_num + 2])
            # 扩展之后 obs pre 和 self.transition_scores 维度相同
            score = obs_expand + pre_expand + self.transition_scores
            # 计算对数分数
            pre = self._log_sum_exp(score)

        return self._log_sum_exp(pre)
    def forward(self, emission_scores, sequence_labels):

        total_loss = 0.0
        for emission_score, sequence_label in zip(emission_scores, sequence_labels):
            # 计算真实路径得分
            real_path_score = self._get_real_path_score(emission_score, sequence_label)
            # 计算所有路径分数
            total_path_score = self._get_total_path_score(emission_score)
            # 最终损失
            finish_loss = total_path_score - real_path_score
            # 累加不同句子的损失
            total_loss += finish_loss

        return total_loss
    def predict(self, emission_score):
        """使用维特比算法，结合发射矩阵+转移矩阵计算最优路径"""

        # 扩展发射分数矩阵
        emission_score_expand = self._expand_emission_matrix(emission_score)

        # 计算分数
        ids = torch.zeros(1, self.label_num + 2, dtype=torch.long, device=device)
        val = torch.zeros(1, self.label_num + 2, device=device)

        pre = emission_score_expand[0]

        for obs in emission_score_expand[1:]:

            # 扩展 pre 维度
            pre_expand = pre.reshape(-1, 1).expand([self.label_num + 2, self.label_num + 2])
            # 扩展 obs 维度
            obs_expand = obs.expand([self.label_num + 2, self.label_num + 2])
            # 扩展之后 obs pre 和 self.transition_scores 维度相同
            score = obs_expand + pre_expand + self.transition_scores

            # 获得当前多分支中最大值的分支索引
            value, index = score.max(dim=0)
            # 拼接每一个时间步的结果
            ids = torch.cat([ids, index.unsqueeze(0)], dim=0)
            val = torch.cat([val, value.unsqueeze(0)], dim=0)
            # 计算分数
            pre = value

        # 先取出最后一个的最大值
        index = torch.argmax(val[-1])
        best_path = [index]

        # 再回溯前一个最大值
        # 由于为了方便拼接，我们在第一个位置默认填充了0
        for i in reversed(ids[1:]):
            # 获得分数最大的索引
            # index = torch.argmax(v)
            # 获得索引对应的标签ID
            index = i[index].item()
            best_path.append(index)

        best_path = best_path[::-1][1:-1]

        return best_path

class NER(nn.Module):

    def __init__(self, vocab_size, label_num):
        super(NER, self).__init__()

        self.vocab_size = vocab_size
        self.label_num = label_num

        # 双向长短记忆网络
        self.bilstm = BiLSTM(vocab_size=self.vocab_size, label_num=self.label_num)
        # 条件随机场网络层
        self.crf = CRF(label_num=self.label_num)

    def forward(self, inputs, labels, length):

        # 计算输入批次样本的每个 Token 的分数，即：每个句子的发射矩阵
        emission_scores = self.bilstm(inputs, length)
        # 计算批次样本的总损失
        batch_loss = self.crf(emission_scores, labels)

        # 返回总损失
        return batch_loss

    def save_model(self, save_apth):
        save_info = {
            'init': {'vocab_size': self.vocab_size, 'label_num': self.label_num},
            'state': self.state_dict()
        }
        torch.save(save_info, save_apth)

    def predict(self, inputs):

        # 计算输入批次样本的每个 Token 的分数，即：每个句子的发射矩阵
        emission_scores = self.bilstm.predict(inputs)
        # viterbi_decode 函数接收的发射矩阵为二维的 (seq_len, scores)
        logits = self.crf.predict(emission_scores)

        return logits
