import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset
# from sklearn.metrics import mean_squared_error
# from sklearn.metrics import r2_score
# from sklearn.metrics import mean_absolute_error
from math import sqrt
import random
import os
import math
import numpy as np
import copy
from thop import profile


class CNN(nn.Module):
    def __init__(self, in_cnn, out_cnn, output_dim=1):
        """ 
            类初始化函数
        Args:
            in_dim: 输入的通道数
            out_dim: 输出的通道数
        Return:
            None.
         """
        super(CNN, self).__init__()
        # 第一次卷积，在最后的维度上卷积，保持卷积后的输出形状不变
        # L_out = floor((L_in + 2*padding - dilation*(kernel_size-1)-1)/stride + 1)
        self.conv1 = nn.Conv1d(in_channels=in_cnn, 
                               out_channels=64, 
                               kernel_size=3, 
                               padding=1)
        # 第二次卷积，保持卷积后的输出形状不变
        self.conv2 = nn.Conv1d(in_channels=64, 
                               out_channels=out_cnn, 
                               kernel_size=3, 
                               padding=1)
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
        # 交换最后两个维度，以便在序列维度上卷积
        x = F.relu(self.conv1(x.transpose(1, 2))) # x.shape([batch_size, feature_dim, in_dim])
        x = F.relu(self.conv2(x)) # x.shape([batch_size, out_dim, in_dim])
        # 恢复之前的形状
        return self.output(x.transpose(1, 2))[:, -1, :]

# 定义CNN层
class CNNLayer(nn.Module):
    def __init__(self, num_channels, out_dim, kernel_size=1):
        super(CNNLayer, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=num_channels, 
                               out_channels=out_dim, 
                               kernel_size=kernel_size)
    
    def forward(self, x):
        x = F.relu(self.conv1(x)) # x.shape([batch_size, out_dim, 1])
        return x

# 定义LSTM层
class LSTMLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, bidirectional):
        super(LSTMLayer, self).__init__()
        self.hidden_size = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, 
                            num_layers=num_layers, batch_first=True, 
                            bidirectional=bidirectional)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers*2 if self.bidirectional else 
                         self.num_layers, x.size(0), 
                         self.hidden_size) # 初始化隐藏状态h0
        c0 = torch.zeros(self.num_layers*2 if self.bidirectional else 
                         self.num_layers, x.size(0), 
                         self.hidden_size)  # 初始化记忆状态c0
        output, (hidden, cell) = self.lstm(x, (h0, c0))
        # output.shape([batch_size, 1, hidden_dim*2 if bidirectional else hidden_dim])
        return output

# 定义Attention层
class AttentionLayer(nn.Module):
    def __init__(self, feature_dim, step_dim):
        super(AttentionLayer, self).__init__()
        self.attention = nn.Linear(feature_dim, step_dim)
        self.context_vector = nn.Linear(step_dim, 1, bias=False)

    def forward(self, x):
        # 将输出值限制在 -1 到 1 之间，shape：[batch_size, feature_dim, step_dim]
        attention_weights = torch.tanh(self.attention(x))
        # 为每个时间步计算一个未归一化的注意力权重，shape：[batch_size, 1]
        attention_weights = self.context_vector(attention_weights).squeeze(2)
        attention_weights = F.softmax(attention_weights, dim=1) # 对权重归一化
        # 将注意力权重与输入 x 相乘，shape：[batch_size, feature_dim]
        context_vector = torch.bmm(attention_weights.unsqueeze(1), x).squeeze(1)
        return context_vector, attention_weights


class CLAM(nn.Module):
    def __init__(self, num_channels, out_dim, kernel_size, hidden_dim,
                  num_layers, bidirectional, step_dim, output_dim):
        super(CLAM, self).__init__()
        self.cnn = CNNLayer(num_channels, out_dim, kernel_size)
        self.lstm = LSTMLayer(out_dim, hidden_dim, 
                              num_layers, bidirectional)
        self.attention = AttentionLayer(hidden_dim * 2 if bidirectional else 
                                        hidden_dim, step_dim)
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
    
    def forward(self, x):
        x = x.transpose(1, 2) # 交换维度
        x = self.cnn(x)
        x = x.transpose(1, 2)
        x = self.lstm(x)
        x, _ = self.attention(x)
        x = self.fc(x)
        # x = self.fc(x[:, -1, :]) # 取序列最后一个时间步的输出作为预测
        return x

class CLM(nn.Module):
    def __init__(self, num_channels, out_dim, kernel_size, hidden_dim, 
                 num_layers, bidirectional, output_dim):
        super(CLM, self).__init__()
        self.cnn = CNNLayer(num_channels, out_dim, kernel_size)
        self.lstm = LSTMLayer(out_dim, hidden_dim, 
                              num_layers, bidirectional)
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
    
    def forward(self, x):
        x = x.transpose(1, 2) # 交换维度
        x = self.cnn(x)
        x = x.transpose(1, 2)
        x = self.lstm(x)
        # x = self.fc(x)
        x = self.fc(x[:, -1, :]) # 取序列最后一个时间步的输出作为预测
        return x

class LM(nn.Module):
    def __init__(self, input_dim, hidden_dim, 
                 num_layers, bidirectional, output_dim):
        super(LM, self).__init__()
        self.lstm = LSTMLayer(input_dim, hidden_dim, 
                              num_layers, bidirectional)
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
    
    def forward(self, x):
        x = self.lstm(x)
        # x = self.fc(x)
        x = self.fc(x[:, -1, :]) # 取序列最后一个时间步的输出作为预测
        return x

def main():
    seq_len = 16
    features = 1
    dropout = 0.1
    kernel_size = 2
    hidden_dim = 32
    num_layers = 1
    bidirectional = False
    step_dim = 32
    out_cnn = 16
    output_dim = 1
    x = torch.randn(32, seq_len)
    x = x.unsqueeze(2)

    model = LM(features, hidden_dim, num_layers, bidirectional, output_dim)
    # print(model)
    print("\n--- 模型参数 ---")
    total_params = 0
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            print(f"层: {name}, 大小: {parameter.size()}, 参数数量: {parameter.numel()}")
            total_params += parameter.numel()
    print(f"总可训练参数数量: {total_params}")

    print("\n--- 模型 FLOPs 和参数量 ---")
    macs, params = profile(model, inputs=(x,))
    print(f"FLOPs: {macs *2}") # FLOPs
    print(f"参数量 (thop): {params}") # parameters
    
if __name__ == "__main__":
    main()




