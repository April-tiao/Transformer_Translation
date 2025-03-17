import math
import torch
from torch import nn

# #生成下三角掩码
# def get_subsequent_mask(size):
#     # 1为batch维度
#     mask_shape = (1, size, size)
#     # 使用torch.tril函数生成下三角矩阵，然后取反，以得到后续位置为0，当前位置及其前序位置为1的掩码
#     # 最后返回这个掩码，用于后续的注意力机制中，以确保模型在预测第i个词时，只能看到第i个词之前的词
#     return 1-torch.tril(torch.ones(mask_shape)).byte()

# # Attention处理Key_Padding_Mask
# def get_padding_mask(x, padding_idx):
#     # 扩展Q维度
#     return (x == padding_idx).unsqueeze(1).bool()

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
        key = self.W_K(key).view(batch_size, -1, self.n_head, self.d_k).transpose(1, 2)
        value = self.W_V(value).view(batch_size, -1, self.n_head, self.d_k).transpose(1, 2)

        # 计算注意力
        # mask 升维 
        if mask is not None:
             mask = mask.unsqueeze(1) #mask 升维, (2,2,3,4),unsqueeze(1)指2个头的维度
        context, attn = attention(query, key, value, mask, self.dropout)

        # 拼接
        concat = context.transpose(1, 2).reshape(batch_size, -1, self.n_head * self.d_k)
        output = self.linear(concat) #子层输出 output 是经过多头注意力计算后的结果
        return self.norm(output + residual) #标准化。返回多头注意力子层的输出，即经过多头注意力计算后的结果。  

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
        # return torch.softmax(self.linear(x), dim=-1) #dim=-1表示对最后一个维度进行softmax操作，即对每个样本的每个类别进行softmax。
        return self.linear(x) #不用softmax，解决训练慢的问题

# 封装完整模型结构
class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, n_head, d_ff, N=6, dropout=0.1):
        super().__init__()
        self.encoder = Encoder(src_vocab_size, d_model, n_head, d_ff, N, dropout)# src_vocab_size翻译前词表大小
        self.decoder = Decoder(tgt_vocab_size, d_model, n_head, d_ff, N, dropout)# tgt_vocab_size翻译后词表大小
        self.generator = Generator(d_model, tgt_vocab_size)
    def forward(self, src_x, src_mask, tgt_x, tgt_mask):

        memory = self.encoder(src_x, src_mask)
        output = self.decoder(tgt_x, tgt_mask, memory, src_mask)
        return self.generator(output)

# 使用 xavier 初始化模型参数
def make_model(src_vocab_size, tgt_vocab_size, d_model, n_head, d_ff, N=6, dropout=0.1):
    model = Transformer(src_vocab_size, tgt_vocab_size, d_model, n_head, d_ff, N, dropout)
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model


# 调用测试
if __name__ == '__main__':
    src_vocab_size = 1000
    tgt_vocab_size = 2000
    d_model = 512
    n_head = 8
    d_ff = 2048
    N = 6
    dropout = 0.1
    model = make_model(src_vocab_size, tgt_vocab_size, d_model, n_head, d_ff, N, dropout)
    # print(model)

    # # 输入数据
    # src_inputs = torch.tensor([
    #     [1, 2, 3],
    #     [4, 5, 0],
    # ])
    # src_mask = get_padding_mask(src_inputs, 0)
    # tgt_inputs = torch.tensor([
    #     [1, 2, 3, 4],
    #     [4, 5, 0, 0],
    # ])
    # # 处理mask
    # tgt_pad_mask = get_padding_mask(tgt_inputs, 0)
    # subsequent_mask = get_subsequent_mask(4)
    # tgt_mask = tgt_pad_mask | subsequent_mask
    # predict = model(src_inputs, src_mask, tgt_inputs, tgt_mask)
    # print(predict.shape) #[2, 4, 2000] 2个句子，4个词，2000个词表，2000个词表是指定的




