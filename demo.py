import math
import sacrebleu
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence  
from matplotlib import pyplot as plt


# 自定义学习率调整策略
def lr_lambda_fn(step, wramup):
    """
    根据当前步骤数和预热步骤数调整学习率。

    如果当前步骤数小于等于预热步骤数，则学习率线性增加；否则，学习率线性减少。
    这种策略有助于模型在训练初期更快地进入状态，并在训练后期保持稳定的学习率。

    参数:
    step (int): 当前步骤数
    wramup (int): 预热步骤数

    返回:
    float: 调整后的学习率
    """
    # 初始化学习率为0，随后根据条件计算具体值
    lr = 0
    # 当前步骤数小于等于预热步骤数时，学习率线性增加
    if step <= wramup:
        lr = step / wramup * 10
    # 当前步骤数大于预热步骤数时，学习率线性减少
    else:
        lr = wramup / step * 10
    # 确保学习率不低于0.1，以防止学习率过早过低
    return max(lr, 0.1)

rates = []
total_epoch = 1000
steps = range(total_epoch)
for step in steps:
    r = lr_lambda_fn(step, total_epoch/4)
    rates.append(r)
plt.plot(steps, rates)
plt.show()
exit()


# 论文中的学习率调整策略
from matplotlib import pyplot as plt
def lr_lambda_fn(step, model_size, factor, warmup):
    if step == 0:
        step = 1
    return factor * (
        model_size ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5))
    )
d_model = 512
factor = 0.1
warmup = 4000
rates = []
steps = range(0, 20000)
for step in steps:
    r = lr_lambda_fn(step, d_model, factor, warmup)
    rates.append(r)
plt.plot(steps, rates)
plt.show()
# exit()


# 参考句子
refs = [['我喜欢吃苹果。', '我喜欢吃水果。'],
        ['这本书很有意思。', '这本书很好玩。'],
        ['他是一个出色的演员。', '他是一名杰出的演员。']]
# 候选句子
hyp = ['我爱吃苹果。', '这本书非常有趣。', '他是一位优秀的演员。']
bleu = sacrebleu.corpus_bleu(hyp, refs, tokenize='zh')
print(bleu.score)
# exit()

# pad_sequence
# 要进行批量运行，需要把句子填充成等长，之前项目里面一直用的是循环，换个方法。
batch_src = [
    [1, 2, 3],
    [4, 5]
]
# pad_sequence 函数说明
# 该函数用于将不同长度的序列填充到相同长度，以便进行批量处理。
# 参数:
# - sequences: 一个包含多个序列的列表，每个序列可以是Tensor。
# - batch_first: 如果为True，输出的Tensor将以batch为第一维度；如果为False，输出的Tensor将以sequence为第一维度。
# - padding_value: 用于填充序列的值。
# 返回值:
# 返回一个Tensor，其中每个序列都被填充或截断至相同长度。

# 使用pad_sequence函数对源句子进行填充，batch_first=True表示输出Tensor的第一个维度是batch大小，padding_value=0表示用0进行填充
src_pad = pad_sequence([torch.LongTensor(src) for src in batch_src], True, 0)
print(src_pad)

# 在每个源句子前后添加特殊标记后进行填充
# 200表示句子的开头，300表示句子的结尾，然后进行填充
src_pad = pad_sequence([torch.LongTensor([200]+src+[300]) for src in batch_src], True, 0)
print(src_pad)
exit()


# 数据对齐 zip 技巧
# 把数字和字母分开
batch = [
    [[1,2,3], ['a','b','c']],
    [[4,5,6], ['d','e','f']],
]
# nums = []
# abcs = []
# for num, abc in batch:
#     nums.append(num)
#     abcs.append(abc)

nums, abcs = zip(*batch)
print(nums)
print(abcs)
exit()



# 生成三角矩阵
#遮掩的位置可以是0，也可以用1，但要和 pad 规则一致，方便叠加。同时，需要注意主对角线上不能遮掩。
# tril = torch.tril(torch.ones((3, 3))) #torch.tril 函数返回一个下三角矩阵，其中对角线上的元素为0，其余元素为1。
# print(1-tril)
# exit()
# scores = torch.randn(2, 3, 3)
# print(scores)
# inputs = torch.tensor([
#     [1, 2, 3],
#     [4, 5, 0]
# ])
# mask = (inputs == 0).unsqueeze(1).bool()
# print(mask)
# scores = scores.masked_fill(mask, -1e9)
# print(scores)
# exit()


# 封装 Embedding 类（文本嵌入层）
class Embeddings(nn.Module):
    def __init__(self, vocab_size, d_model): #vocab_size词表大小，d_model编码维度
        
        super().__init__()
        # Embedding层
        self.lut = nn.Embedding(vocab_size, d_model)
        # Embedding维数
        self.d_model = d_model

    def forward(self, x):
        # 返回x对应的embedding矩阵，需要乘以math.sqrt(d_model)
        return self.lut(x) * math.sqrt(self.d_model)
    
# emb = Embeddings(10, 8)
# inputs = torch.tensor([   #输入的x
#     [1, 2, 3],
#     [4, 5, 0],
# ])
# output1 = emb(inputs)
# print(output1)
# exit()

# print(output1.shape)


# 封装位置编码层
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        # 位置和除数
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -math.log(10000) / d_model)
        # 修改pe矩阵的值
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # 扩展 batch 维度
        pe = pe.unsqueeze(0)
        # 存储为不需要计算梯度的参数
        self.register_buffer('pe', pe)
    def forward(self, x):
        # 直接使用 self.pe 而不是 Variable
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)
    
    # def forward(self, x):
    #     x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
    #     return self.dropout(x)
    

# class PositionalEncoding(nn.Module):
#     def __init__(self, d_model, dropout=0.1, max_len=5000):
#         super().__init__()
#         self.dropout = nn.Dropout(dropout)

#         # 计算位置编码值
#         self.pe = torch.zeros(max_len, d_model)
#         for pos in range(max_len):
#             for j in range(d_model):
#                 angle = pos / math.pow(10000, (j//2)*2 / d_model)
#                 if j % 2 == 0:
#                     self.pe[pos][j] = math.sin(angle)
#                 else:
#                     self.pe[pos][j] = math.cos(angle)

#         # norm 实例化，对输入的x进行归一化
#         self.norm = nn.LayerNorm(d_model)

    # def forward(self, x):
    #     return self.dropout(x + self.pe[:x.size(1)]) #1表示（2，3，8）中的3，3个词
    
# pe = PositionalEncoding(8)
# output2 = pe(output1)
# print(output2)
# exit()


# Attention处理Key_Padding_Mask
def get_padding_mask(x, padding_idx):
    # 扩展Q维度
    return (x == padding_idx).unsqueeze(1).bool()


# Attention函数实现
def attention(query, key, value, mask=None, dropout=None):
    # 将query矩阵的最后一个维度值作为d_k
    d_k = query.size(-1)  #(2,3,8)的8

    # 将key的最后两个维度互换(转置)，与query矩阵相乘，除以d_k开根号
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        mask = mask.bool()
        scores = scores.masked_fill(mask, -1e9)

    p_attn = torch.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

# 调用示例
# query = key = value = torch.randn(2, 3, 8)
# dropout = nn.Dropout(0.1)
# padding_mask = get_padding_mask(inputs, 0)
# print(padding_mask)
# exit()

# result = attention(query, key, value,padding_mask, dropout=dropout)

# result[0] 是经过注意力机制计算后的输出张量，形状为 (2, 3, 8)。
# result[1] 是注意力权重矩阵 p_attn，形状为 (2, 3, 3)。 mask后应该为0
# print(result[1]) # .shape 看形状 (2, 3, 8)

# 多头注意力
class MultiHeadedAttention(nn.Module):
    def __init__(self, d_model, n_head, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        # 头的数量要能整除词向量维度
        assert d_model % n_head == 0
        self.d_k = d_model // n_head
        self.n_head = n_head
        
        # 三个线性变换，一个多头拼接之后的线性变换
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.linear = nn.Linear(d_model, d_model, bias=False)

        # norm
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, query, key, value, mask=None):
        residual = query    #保存的是​多头注意力子层的输入​（即进入 W_Q、W_K、W_V 前的原始输入）。
        # 分头
        batch_size = query.size(0)
        query = self.W_Q(query).view(batch_size, -1, self.n_head, self.d_k).transpose(1, 2)
       
        # 打印query 的形状
        # print(query.shape)
        key = self.W_K(key).view(batch_size, -1, self.n_head, self.d_k).transpose(1, 2)
        value = self.W_V(value).view(batch_size, -1, self.n_head, self.d_k).transpose(1, 2)

        # 计算注意力
        # mask 升维 
        if mask is not None:
             mask = mask.unsqueeze(1) #mask 升维, (2,2,3,4),unsqueeze(1)指2个头的维度
        context, attn = attention(query, key, value, mask, self.dropout)
        # print(context.shape)

        # 拼接
        concat = context.transpose(1, 2).reshape(batch_size, -1, self.n_head * self.d_k)
        output = self.linear(concat) #子层输出 output 是经过多头注意力计算后的结果
        # print(output.shape) #(2,3,8)
        # print(residual + output)
        return self.norm(output + residual) #标准化。返回多头注意力子层的输出，即经过多头注意力计算后的结果。  

# 测试多头注意力
# mha = MultiHeadedAttention(8, 2)
# # result = mha(query, key, value, None)
# output3 = mha(query, key, value, padding_mask)
# print(output3.shape)

# 前馈神经网络层
# 残差连接，线性变换和残差返回
class FeedForwardNet(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        """
        初始化FeedForward神经网络层。
        参数:
        - d_model: 输入和输出的特征维度。
        - d_ff: 前馈神经网络内部的隐藏层维度。
        - dropout: 在网络中应用的Dropout比例，默认为0.1。
        该构造函数初始化了网络的所有必要组件，包括两个线性变换、ReLU激活函数、LayerNorm层和Dropout层。
        """
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()
        self.norm = nn.LayerNorm(d_model)
    def forward(self, x):
        """
        实现前馈神经网络的前向传播过程。
        参数:
        - x: 输入特征。
        返回:
        - 输出经过前馈神经网络处理后的特征。
        这个函数首先保存输入的残差，然后对输入进行第一次线性变换，接着通过ReLU激活函数，
        再进行第二次线性变换，并应用Dropout。最后，将变换后的输出与最初的残差相加，并通过LayerNorm层，
        以确保输出具有稳定的均值和方差。
        """
        residual = x
        x = self.relu(self.w1(x))
        x = self.dropout(self.w2(x))
        return self.norm(x + residual)
# 调用测试
# ffn = FeedForwardNet(8, 2048) #从8维到2048维，再转换回去
# output4 = ffn(output3) #多头的输出
# print(output4.shape)


# 编码器子层
class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_head, d_ff, dropout=0.1):
        super().__init__()
        self.mha = MultiHeadedAttention(d_model, n_head, dropout)
        self.ffn = FeedForwardNet(d_model, d_ff, dropout)
    def forward(self, x, mask=None):
        output = self.mha(x, x, x, mask)
        return self.ffn(output)

# 编码器层
class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_head, d_ff, N=6, dropout=0.1):
        super().__init__()
        self.emb = Embeddings(vocab_size, d_model)
        self.pos = PositionalEncoding(d_model, dropout)
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, n_head, d_ff, dropout)
            for _ in range(N)
        ])
    def forward(self, x, mask):
        x = self.emb(x)
        x = self.pos(x)
        for layer in self.layers:
            x = layer(x, mask)
        return x
# # 调用测试
# inputs = torch.tensor([
#     [1, 2, 3],
#     [4, 5, 0],
# ])
# mask = get_padding_mask(inputs, 0) #padding 的索引是0
# enc = Encoder(10, 8, 2, 32)
# output = enc(inputs, mask)
# print(output.shape)
# exit()

#封装函数
# 参数 size 为句子长度 （2，3，3）2个句子，3个词
def get_subsequent_mask(size):
    # 1为batch维度
    mask_shape = (1, size, size)
    # 使用torch.tril函数生成下三角矩阵，然后取反，以得到后续位置为0，当前位置及其前序位置为1的掩码
    # 最后返回这个掩码，用于后续的注意力机制中，以确保模型在预测第i个词时，只能看到第i个词之前的词
    return 1-torch.tril(torch.ones(mask_shape)).byte()

# 和padding mask 叠加
# inputs = torch.tensor([ # 2个句子，3个词，2×3的张量
#     [1, 2, 3],
#     [4, 5, 0],
# ])
# pad_mask = get_padding_mask(inputs, 0)
# print(pad_mask)  
# sub_mask = get_subsequent_mask(3) 
# print(sub_mask)
# mask = sub_mask | pad_mask # pad_mask和sub_mask叠加，取 或，1表示mask了
# print(mask)
# exit()


# 解码器子层
# 第二个注意力层中，query 来自上一层输出，key 和 value 来自编码层的输出，称为 memory。
class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_head, d_ff, dropout=0.1):
        super().__init__()
        self.self_mha = MultiHeadedAttention(d_model, n_head, dropout)
        self.mha = MultiHeadedAttention(d_model, n_head, dropout)
        self.ffn = FeedForwardNet(d_model, d_ff, dropout)
    def forward(self, x, mask, memory, src_mask): #src_mask 是编码器的mask,memory 是编码器的输出
        x = self.self_mha(x, x, x, mask)
        x = self.mha(x, memory, memory, src_mask) #(x, memory, memory)表示 Q K V 
        return self.ffn(x)


# 解码器层
class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_head, d_ff, N=6, dropout=0.1):
        super().__init__()
        self.emb = Embeddings(vocab_size, d_model)
        self.pos = PositionalEncoding(d_model, dropout)
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, n_head, d_ff, dropout)
            for _ in range(N)
        ])
    def forward(self, x, mask, memory, src_mask):
        x = self.emb(x)
        x = self.pos(x)
        for layer in self.layers:
            x = layer(x, mask, memory, src_mask)
        return x

# 调用测试
# 编码部分
# 2个batch, 3个词，8维向量
src_inputs = torch.tensor([
    [1, 2, 3],
    [4, 5, 0],
])
src_mask = get_padding_mask(src_inputs, 0)
encoder = Encoder(10, 8, 2, 32) # 编码成10个词，8维向量，2个头，feed forward 32维
memory = encoder(src_inputs, src_mask)
# 解码部分
# 2个batch, 4个词，8维向量
tgt_inputs = torch.tensor([
    [1, 2, 3, 4],
    [4, 5, 0, 0],
])
decoder = Decoder(20, 8, 2, 32)
# 处理mask
tgt_pad_mask = get_padding_mask(tgt_inputs, 0)
subsequent_mask = get_subsequent_mask(4)
tgt_mask = tgt_pad_mask | subsequent_mask
output = decoder(tgt_inputs, tgt_mask, memory, src_mask)
print(output.shape)



# 生成器层
# 模型的输出值，就是目标序列词表中，每个字的概率值。
class Generator(nn.Module):
    def __init__(self, d_model, vocab_size):
        """
        初始化函数，用于设置输入维度和词汇表大小。

        参数:
        d_model (int): 输入维度，表示模型中向量的维度。
        vocab_size (int): 词汇表大小，表示可能的输出类别数。
        """
        super().__init__()
         # 初始化一个线性变换层，用于将输入向量映射到词汇表大小的维度
        self.linear = nn.Linear(d_model, vocab_size)
    def forward(self, x):
        """
        实现前向传播过程，应用softmax激活函数处理线性变换结果。

        参数:
        x (Tensor): 输入数据，通常是从前一层网络传入的特征表示。

        返回:
        Tensor: 经过softmax激活后的输出，常用于分类任务中表示各类别的概率分布。
        """
        # 应用softmax激活函数，将线性变换后的结果转换为概率分布
        return torch.softmax(self.linear(x), dim=-1) #dim=-1表示对最后一个维度进行softmax操作，即对每个样本的每个类别进行softmax。

# 调用测试
# 通过softmax之后，最大的概率值对应的索引位置，就是生成的词的索引。
generator = Generator(8, 20) # 实例化生成器层
predict = generator(output)
print(predict.shape)
print(torch.argmax(predict, dim=-1)) #在最后一维上 argmax 函数返回的是最大值对应的索引位置，即生成的词的索引。





























# # 实例化类
# emb = nn.Embedding(10, 8, padding_idx=0)#10个词，8维向量

# # lookup table
# print(emb.weight) #weight是个tensor
# # print(emb.weight.shape)

# # 单个词向量
# print(emb(torch.tensor([1]))) #"抛"字
# # 两个句子
# print(emb(torch.tensor([
#     [1, 2, 3],# 抛弃放
#     [1, 5, 0],# 抛言P（填充的pad值，0是特殊字符时，占位），只做张量计算，不参与损失计算
# ])))
# # 指定填充id
# # emb = nn.Embedding(10, 8, padding_idx=0) #传入 padding对应的索引0，放到前面的代码生效






