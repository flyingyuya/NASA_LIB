import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from math import sqrt
import random
import os
import math
import numpy as np
import copy

device = "cuda" if torch.cuda.is_available() else "cpu"

# 定义克隆函数
def clones(module, N):
    """ 
        用于生成相同网络层的克隆函数
    Args:
        module: 需要克隆的目标网络层
        N: 要克隆的数量
    Return:
        返回一个具有相同网络层的 ModuleList 类型的列表
     """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

# 生成自回归掩码张量
def gen_subsequent_mask(size, n_head):
    """ 
        生成向后遮掩的掩码张量, 1代表不被遮掩, 0代表被遮掩
    Args:
        size: 掩码张量最后两个维度的大小
    Return:
        生成的掩码张量(下三角为1, 上三角为0)
     """
    # 定义张量大小
    mask_shape = (n_head, size, size)
    # 生成上三角矩阵并转为 uint8 以节省空间
    subsequent_mask = np.triu(np.ones(mask_shape), k=1).astype('uint8')
    # 反转上三角矩阵并转为 tensor
    return torch.from_numpy(1 - subsequent_mask).bool()

# 计算注意力张量
def attention(query, key, value, mask=None, dropout=None):
    """ 
        计算注意力张量，当 query=key=value 时，则为自注意力机制
    Args:
        query: 输入的 query 张量
        key: 输入的 key 张量
        value: 输入的 value 张量
        mask: 输入的掩码张量
        dropout: 输入的 dropout 实例化对象
    Return:
        计算得到的 query 注意力表示和对应的注意力张量 p_attn
     """
    # 先取 query 最后一维的大小，一般为词嵌入维度
    d_k = query.size(-1)
    # 按照公式， query 与 key 的转置(这里使用 transpose 交换最后两个维度)相乘，在除以缩放系数
    # scores = scores - scores.max(dim=-1, keepdim=True)[0]  # 数值稳定性
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    # 判断是否使用掩码张量
    if mask is not None:
        # 使用 masked_fill 方法，将掩码张量和 scores 张量的每个位置进行比较
        # 如果对应掩码张量处为False，则使用一个很小的值(-1e9)进行替换
        scores = scores.masked_fill(mask==False, -1e9)

    # 对 scores 的最后一维进行 softmax 操作，得到最终的注意力张量
    p_attn = F.softmax(scores, dim=-1)

    # 判断是否进行随机'丢弃'
    if dropout is not None:
        p_attn = dropout(p_attn)

    # 最后，按照公式将注意力张量 p_attn 与 value 张量相乘得到最终的 query 注意力表示
    return torch.matmul(p_attn, value), p_attn

# 定义输入层类
class InputLayer(nn.Module):
    def __init__(self, features, d_model):
        """ 
            类初始化函数
        Args:
            features: 输入的特征维度
            d_model: 输出的词嵌入维度
        Return:
            None.
         """
        super(InputLayer, self).__init__()
        # 定义线性层，将维度映射到 d_model
        self.linear = nn.Linear(features, d_model)

    def forward(self, x):
        """ 
            前向传播函数
            Args:
                x: 输入张量表示
            Return:
                计算后的张量
         """
        return self.linear(x)

# 定义位置编码类
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=500):
        """ 
            类初始化函数
        Args:
            d_model: 词嵌入维度
            dropout: 置0比率
            max_len: 每个样本的最大长度
        Return:
            None.
         """
        super(PositionalEncoding, self).__init__()

        self.dropout = nn.Dropout(p=dropout)
        # 初始化一个位置编码矩阵 pe
        pe = torch.zeros(max_len, d_model)
        # 初始化一个绝对位置矩阵 position , 并拓展为 (max_len * 1) 的大小
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # 初始化一个大小为 (d_model * 1) 的变换矩阵, 产生一个递减频率
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        # 分别对位置编码矩阵 pe 的偶数列和奇数列进行编码
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # 扩展维度以适应上一层的输出
        pe = pe.unsqueeze(0)
        # 注册位置编码矩阵 pe 为 buffer , 它不随模型的优化而改变
        self.register_buffer('pe', pe)

    def forward(self, x):
        """ 
            由于默认的 max_len 太大，使用 x.size(1) 以适配输入张量的大小；
            使用 Variable 对 pe 进行封装，使其与 x 的样式相同，但是不需要进行梯度求解
            Args:
                x: 序列张量表示
            Return:
                编码后的张量
         """
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)

# 定义多头注意力机制类
class MutiHeadAttention(nn.Module):
    def __init__(self, n_head, embedding_dim, dropout=0.1):
        """ 
            类初始化函数
        Args:
            n_head: 多头注意力的头数
            embedding_dim: 词嵌入维度
            dropout: 置0比率
        Return:
            None.
         """
        super(MutiHeadAttention, self).__init__()
        # 判断模型维度 embedding_dim 能否被头数 n_head 整除
        assert embedding_dim % n_head == 0

        # 计算每个头的分割词向量维度
        self.d_k = embedding_dim // n_head
        
        self.n_head = n_head
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        # 克隆4个线性层的实例化对象，分别用于(Q, K, V) 以及 Concat
        self.linears = clones(nn.Linear(embedding_dim, embedding_dim), 4)
        
    def forward(self, query, key, value, mask=None):
        """ 
            前向传播函数
            Args:
                query: query 张量表示
                key: key 张量表示
                value: value 张量表示
                mask: 掩码张量表示, 默认为None
            Return:
                编码后的张量
         """
        # 判断是否存在掩码张量
        if mask is not None:
            # 扩展维度为4维
            mask = mask.unsqueeze(0)
        # 获取样本大小
        batch_size = query.size(0)
        # 使用 zip 方法对每个张量进行线性变换，然后使用 view 方法为每个头分割输入
        # 最后使用 transpose 方法将头的维度与序列长度的维度进行转置，使序列长度的维度与词向量维度相邻
        query, key, value = \
            [model(x).view(batch_size, -1, self.n_head, self.d_k).transpose(1, 2) 
            for model, x in zip(self.linears, (query, key, value))]
        
        # 计算每个头的注意力张量
        x, self.attn = attention(query, key, value, mask, dropout=self.dropout)

        # 使用 transpose 方法将多头计算结果组成的4维张量进行逆转置
        # 然后使用 contiguous 方法和 view 方法重塑维输入形状
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.n_head * self.d_k)

        # 使用模型列表中的最后一个线性层对 x 进行线性变换得到最后的多头注意力输出
        return self.linears[-1](x)

# 位置编码的前馈全连接层类的定义
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        """ 
            类初始化函数
        Args:
            d_model: 词嵌入维度
            d_ff: 前馈全连接层的映射维度
            dropout: 置0比率
        Return:
            None.
         """
        super(PositionwiseFeedForward, self).__init__()
        # 定义两个线性层的实例对象
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        # 定义 dropout 的实例化对象
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        """ 
            前向传播函数
            Args:
                x: 输入张量表示
            Return:
                计算结果的张量
         """
        return self.w2(self.dropout(F.relu(self.w1(x))))
    
class NormLayer(nn.Module):
    def __init__(self, features, eps=1e-6):
        """ 
            类初始化函数
        Args:
            features: 词嵌入维度
            eps: 非常小的一个参数，用于防止计算时出现分母为零的情况
        Return:
            None.
         """
        super(NormLayer, self).__init__()

        # 定义两个分别为全1和全0的参数张量，对输入进行规范化操作
        # 但是为了不改变目标的原有表征，使用 Parameter 进行封装，使其成为模型参数的一部分
        self.parm1 = nn.Parameter(torch.ones(features))
        self.parm2 = nn.Parameter(torch.zeros(features))

        self.eps = eps

    def forward(self, x):
        """ 
            前向传播函数
            Args:
                x: 输入张量表示
            Return:
                计算结果的张量
         """
        # 在最后一个维度上对 x 求均值和标准差，并保持计算张量的维度，再根据规范化公式进行计算
        # 其中 parm1 为缩放参数， parm2 为偏置参数，*号和+号会自动对张量采用广播机制进行计算
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.parm1 * (x - mean) / (std + self.eps) + self.parm2
    
# 定义子层连接类
class SubLayerConnection(nn.Module):
    def __init__(self, size, dropout=0.1):
        """ 
            类初始化函数
        Args:
            size: 词嵌入维度
            dropout: 置0比率
        Return:
            None.
         """
        super(SubLayerConnection, self).__init__()
        # 定义规范化层的实例对象
        self.norm = NormLayer(size)
        # 定义 dropout 层的实例对象
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, sublayer):
        """ 
            前向传播函数
            Args:
                x: 输入张量表示
                sublayer: 子层连接中的子层函数
            Return:
                计算结果的张量
         """
        # 残差连接，使用输入 x 加上子层计算结果作为最终结果
        return x + self.dropout(self.norm(sublayer(x)))
    
# 定义编码器层类
class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout=0.1):
        """ 
            类初始化函数
        Args:
            size: 词嵌入维度
            self_attn: 多头自注意力子层的实例化对象，自注意力即 Q=K=V
            feed_forward: 前馈全连接子层的实例化对象
            dropout: 置0比率
        Return:
            None.
         """
        super(EncoderLayer, self).__init__()

        self.size = size
        # 传入多头自注意力子层的实例化对象和前馈全连接子层的实例化对象
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        # 克隆2个子层连接对象
        self.sublayers = clones(SubLayerConnection(size, dropout=dropout), 2)

    def forward(self, x, mask=None):
        """ 
            前向传播函数
            Args:
                x: 输入张量表示
                mask: 输入的掩码张量
            Return:
                计算结果的张量
         """
        # 先通过第一个子层连接结构，其中包含多头自注意力子层
        # 再通过第二个子层连接结构，其中包含前馈连接子层
        x = self.sublayers[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayers[1](x, self.feed_forward)

# 定义编码器类
class Encoder(nn.Module):
    def __init__(self, layer, N):
        """ 
            类初始化函数
        Args:
            layer: 编码器层的实例化对象
            N: 编码器层的个数
        Return:
            None.
         """
        super(Encoder, self).__init__()
        # 克隆 N 个编码器层
        self.layers = clones(layer, N)
        # 定义一个规范化层
        self.norm = NormLayer(layer.size)

    def forward(self, x, mask=None):
        """ 
            前向传播函数
            Args:
                x: 输入张量表示
                mask: 输入的掩码张量
            Return:
                计算结果的张量
         """
        # 循环使用每个编码器层对输入张量 x 进行处理
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
    
# 定义解码器层类
class DecoderLayer(nn.Module):
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout=0.1):
        """ 
            类初始化函数
        Args:
            size: 词嵌入维度
            self_attn: 多头自注意力子层的实例化对象，自注意力，即 Q=K=V
            src_attn: 多头注意力子层的实例化对象，注意力，即 Q!=K=V
            feed_forward: 前馈全连接子层的实例化对象
            dropout: 置0比率
        Return:
            None.
         """
        super(DecoderLayer, self).__init__()
        # 传入参数
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        # 克隆3个子层连接对象
        self.sublayers = clones(SubLayerConnection(size, dropout=dropout), 3)

    def forward(self, x, memory, source_mask=None, target_mask=None):
        """ 
            前向传播函数
            Args:
                x: 输入张量表示
                memory: 来自编码器的语义存储张量
                source_mask: 源数据的掩码张量
                target_mask: 目标数据的掩码张量
            Return:
                计算结果的张量
         """
        m = memory
        # 第一个子层对目标数据进行遮掩，以使模型无法利用之后的信息，并计算目标数据的多头自注意力
        x = self.sublayers[0](x, lambda x: self.self_attn(x, x, x, target_mask))
        # 第二个子层对源数据进行遮掩，遮蔽掉对结果没有意义的信息，并计算源数据的多头注意力
        x = self.sublayers[1](x, lambda x: self.src_attn(x, m, m, source_mask))
        # 第三层为前馈全连接子层
        return self.sublayers[2](x, self.feed_forward)
    
# 定义解码器类
class Decoder(nn.Module):
    def __init__(self, layer, N):
        """ 
            类初始化函数
        Args:
            layer: 解码器层的实例化对象
            N: 解码器层的个数
        Return:
            None.
         """
        super(Decoder, self).__init__()
        # 克隆 N 个解码器层
        self.layers = clones(layer, N)
        # 定义一个规范化层
        self.norm = NormLayer(layer.size)

    def forward(self, x, memory, source_mask=None, target_mask=None):
        """ 
            前向传播函数
            Args:
                x: 输入张量表示
                memory: 来自编码器的语义存储张量
                source_mask: 源数据的掩码张量
                target_mask: 目标数据的掩码张量
            Return:
                计算结果的张量
         """
        # 循环使用每个解码器层对输入张量 x 进行处理
        for layer in self.layers:
            x = layer(x, memory, source_mask, target_mask)
        return self.norm(x)

# 定义CNN层类
class CNNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        """ 
            类初始化函数
        Args:
            in_dim: 输入的通道数
            out_dim: 输出的通道数
        Return:
            None.
         """
        super(CNNLayer, self).__init__()
        # 第一次卷积，在最后的维度上卷积，保持卷积后的输出形状不变
        # L_out = floor((L_in + 2*padding - dilation*(kernel_size-1)-1)/stride + 1)
        self.conv1 = nn.Conv1d(in_channels=in_dim, 
                               out_channels=64, 
                               kernel_size=3, 
                               padding=1)
        # 第二次卷积，保持卷积后的输出形状不变
        self.conv2 = nn.Conv1d(in_channels=64, 
                               out_channels=out_dim, 
                               kernel_size=3, 
                               padding=1)
    
    def forward(self, x):
        """ 
            前向传播函数
            Args:
                x: 输入张量表示
            Return:
                计算后的张量
         """
        # 交换最后两个维度，以便在序列维度上卷积
        x = F.relu(self.conv1(x.transpose(1, 2))) # x.shape([batch_size, 256, in_dim])
        x = F.relu(self.conv2(x)) # x.shape([batch_size, out_dim, in_dim])
        # 恢复之前的形状
        return x.transpose(1, 2)
    
class OutputLayer(nn.Module):
    def __init__(self, in_cnn, out_cnn, output_dim=1):
        """ 
            类初始化函数
        Args:
            in_cnn: 卷积输入的通道数
            out_dim: 卷积输出的通道数
            output_dim: 最终输出的特征维度，默认为 1
        Return:
            None.
         """
        super(OutputLayer, self).__init__()
        # 定义CNN层
        self.cnn = CNNLayer(in_cnn, out_cnn)
        # 定义线性输出层
        self.output = nn.Linear(out_cnn, output_dim)

    def forward(self, x):
        """ 
            前向传播函数
            Args:
                x: 输入张量表示
            Return:
                计算后的张量
         """
        x = self.cnn(x)
        # 取序列最后一个时间步的输出作为预测
        return self.output(x)[:, -1, :]

class TransformerCNN(nn.Module):
    def __init__(self, features, d_model, out_cnn, head=4, d_ff=2048,
                 output_dim=1, dropout=0.1, N=4):
        """ 
            类初始化函数
        Args:
            features: 数据的特征数
            d_model: 词嵌入的维度
            out_cnn: 卷积输出的通道数
            head: 多头注意力的头数
            d_ff: 前馈全连接层的映射维度
            output_dim: 最终输出的特征维度，默认为 1
            dropout: 置0比率
            N: 编码器和解码器层的个数
        Return:
            None.
         """
        super(TransformerCNN, self).__init__()
        c = copy.deepcopy
        # 定义注意力子层和前馈全连接子层对象
        self_attn = src_attn = MutiHeadAttention(head, d_model, dropout)
        ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        # 定义模型的实例化对象
        self.input = InputLayer(features, d_model)
        self.pe = PositionalEncoding(d_model, dropout)
        self.encoder = Encoder(EncoderLayer(d_model, c(self_attn), c(ff), dropout), N)
        self.decoder = Decoder(DecoderLayer(d_model, c(self_attn), 
                                            c(src_attn), c(ff), dropout), N)
        self.output = OutputLayer(d_model, out_cnn, output_dim)

    def forward(self, source, target, source_mask, target_mask):
        """
            前向传播函数
            Args:
                source: 源数据的输入张量表示
                target: 目标数据的输入张量表示
                source_mask: 源数据的掩码张量
                target_mask: 目标数据的掩码张量
            Return:
                计算结果的张量
         """
        # 首先对源数据和目标数据进行嵌入和位置编码
        source = self.pe(self.input(source))
        target = self.pe(self.input(target))
        # 然后将源数据进入编码器后输出
        memory = self.encoder(source, source_mask)
        # 接着将源数据和目标数据送入解码器后输出
        decode = self.decoder(target, memory, source_mask, target_mask)
        # 最后经过输出层输出
        return self.output(decode)

# 定义数据集类
class TimeSeriesDataset(Dataset):
    def __init__(self, data, window_size, pred_len=1):
        self.window_size = window_size
        self.pred_len = pred_len
        self.data = torch.tensor(data, dtype=torch.float32).to(device)
        # 计算最大索引
        self.max_index = self.data.shape[0] - self.window_size - pred_len

    def __len__(self):
        return self.max_index + 1 #  返回有效数据长度

    def __getitem__(self, index):
        if index > self.max_index:
            raise IndexError(f"Index {index} is out of bounds."
                            f"Max index is {self.max_index}")
        source = self.data[index:index + self.window_size]
        target = self.data[index + 1:index + self.window_size + 1] # 向后偏移一个时间步
        return source.unsqueeze(1), target.unsqueeze(1) # 添加一个特征维度

# 获取模型
def get_model(features, d_model, out_cnn, head=4, d_ff=2048, 
              output_dim=1, dropout=0.1, N=4):
    # 实例化模型
    model = TransformerCNN(features, d_model, out_cnn, head, d_ff, 
                           output_dim, dropout, N)
    # 初始化模型参数
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model

def main():
    d_model = 128
    seq_len = 4
    features = 1
    dropout = 0.2

    x = Variable(torch.FloatTensor([[100, 2, 421, 508], [491, 998, 1, 221]]))
    x = x.unsqueeze(2)

    # emb = InputLayer(features, d_model)
    # emb_r = emb(x)
    # print("embr:", embr)
    # print(embr.shape)

    # pe = PositionalEncoding(d_model, dropout=dropout, max_len=seq_len)
    # pe_r = pe(emb_r)
    # print("per:", per)
    # print(per.shape)

    # query = key = value = pe_r

    # attn, p_attn = attention(query, key, value)
    # print("attn:", attn)
    # print(attn.shape)
    # print("p_attn:", p_attn)
    # print(p_attn.shape)

    # mask = Variable(torch.zeros(2, 4, 4))
    # attn, p_attn = attention(query, key, value, mask=mask)
    # print("attn:", attn)
    # print(attn.shape)
    # print("p_attn:", p_attn)
    # print(p_attn.shape)

    head = 8
    mask = Variable(torch.zeros(head, seq_len, seq_len))
    # print(gen_subsequent_mask(seq_len, head))
    # self_attn = MutiHeadAttention(head, d_model, dropout=dropout)
    # self_attn_r = self_attn(query, key, value, mask)
    # print("mha_r", mha_r)
    # print(mha_r.shape)

    d_ff = 256
    # c = copy.deepcopy
    # ff = PositionwiseFeedForward(d_model, d_ff, dropout=dropout)
    # ff_r = ff(self_attn_r)
    # print("ff_r", ff_r)
    # print(ff_r.shape)

    # ln = NormLayer(d_model)
    # ln_r = ln(ff_r)
    # print("ln_r", ln_r)
    # print(ln_r.shape)

    # x = pe_r
    # sublayer = lambda x: self_attn(x, x, x, mask)
    # slc = SubLayerConnection(d_model, dropout=dropout)
    # slc_r = slc(pe_r, sublayer)
    # print("slc_r:", slc_r)
    # print(slc_r.shape)

    # el = EncoderLayer(d_model, c(self_attn), c(ff), dropout=dropout)
    # el_r = el(pe_r, mask)
    # print("el_r:", el_r)
    # print(el_r.shape)

    N = 4
    # en = Encoder(el, N)
    # en_r = en(pe_r, mask)
    # print("en_r:", en_r)
    # print(en_r.shape)

    # src_attn = self_attn
    source_mask = target_mask = mask
    # dl = DecoderLayer(d_model, c(self_attn), c(src_attn), c(ff), dropout=dropout)
    # dl_r = dl(pe_r, en_r, source_mask, target_mask)
    # print("dl_r:", dl_r)
    # print(dl_r.shape)

    # de = Decoder(dl, N)
    # de_r = de(pe_r, en_r, source_mask, target_mask)
    # print("de_r:", de_r)
    # print(de_r.shape)

    # cnn = CNNLayer(d_model, 64)
    # cnn_r = cnn(de_r)
    # print("cnn_r:", cnn_r)
    # print(cnn_r.shape)

    # ol = OutputLayer(d_model, 64)
    # ol_r = ol(de_r)
    # print("ol_r:", ol_r)
    # print(ol_r.shape)

    out_cnn = 32
    output_dim = 1
    source = target = x
    model = get_model(features, d_model, out_cnn, head, d_ff, output_dim, output_dim, N)
    print(model)
    # model_r = model(source, target, source_mask, target_mask)
    # print("model_r:", model_r)
    # print(model_r.shape)

if __name__ == "__main__":
    main()

""" 
最终模型架构：
TransformerCNN(
  (input): InputLayer(
    (linear): Linear(in_features=1, out_features=128, bias=True)
  )
  (pe): PositionalEncoding(
    (dropout): Dropout(p=1, inplace=False)
  )
  (encoder): Encoder(
    (layers): ModuleList(
      (0-3): 4 x EncoderLayer(
        (self_attn): MutiHeadAttention(
          (dropout): Dropout(p=1, inplace=False)
          (linears): ModuleList(
            (0-3): 4 x Linear(in_features=128, out_features=128, bias=True)
          )
        )
        (feed_forward): PositionwiseFeedForward(
          (w1): Linear(in_features=128, out_features=256, bias=True)
          (w2): Linear(in_features=256, out_features=128, bias=True)
          (dropout): Dropout(p=1, inplace=False)
        )
        (sublayers): ModuleList(
          (0-1): 2 x SubLayerConnection(
            (norm): NormLayer()
            (dropout): Dropout(p=1, inplace=False)
          )
        )
      )
    )
    (norm): NormLayer()
  )
  (decoder): Decoder(
    (layers): ModuleList(
      (0-3): 4 x DecoderLayer(
        (self_attn): MutiHeadAttention(
          (dropout): Dropout(p=1, inplace=False)
          (linears): ModuleList(
            (0-3): 4 x Linear(in_features=128, out_features=128, bias=True)
          )
        )
        (src_attn): MutiHeadAttention(
          (dropout): Dropout(p=1, inplace=False)
          (linears): ModuleList(
            (0-3): 4 x Linear(in_features=128, out_features=128, bias=True)
          )
        )
        (feed_forward): PositionwiseFeedForward(
          (w1): Linear(in_features=128, out_features=256, bias=True)
          (w2): Linear(in_features=256, out_features=128, bias=True)
          (dropout): Dropout(p=1, inplace=False)
        )
        (sublayers): ModuleList(
          (0-2): 3 x SubLayerConnection(
            (norm): NormLayer()
            (dropout): Dropout(p=1, inplace=False)
          )
        )
      )
    )
    (norm): NormLayer()
  )
  (output): OutputLayer(
    (cnn): CNNLayer(
      (conv1): Conv1d(128, 64, kernel_size=(3,), stride=(1,), padding=(1,))
      (conv2): Conv1d(64, 32, kernel_size=(3,), stride=(1,), padding=(1,))
    )
    (output): Linear(in_features=32, out_features=1, bias=True)
  )
)
 """
