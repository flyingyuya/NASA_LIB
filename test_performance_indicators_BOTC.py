import time
import torch
import numpy as np
import pandas as pd
import scipy.io
from datetime import datetime
import matplotlib.pyplot as plt
import os
import math
from math import sqrt
import copy
import random
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# --- 定义设备 ---
device = "cuda:0" if torch.cuda.is_available() else "cpu"


def TimeConvert(hmm):
    """ 
        转换时间格式，将字符串转换成 datatime 格式  
    Args:
        hmm: 待输入的原始时间数据 (ndarray)
    Returns:
        标准化后的时间数据
    """
    year, month, day, hour, minute, second = \
                int(hmm[0]), int(hmm[1]), int(hmm[2]), \
                int(hmm[3]), int(hmm[4]), int(hmm[5])
    return datetime(year=year, month=month, day=day, 
                    hour=hour, minute=minute, second=second)

def LoadMat(mat_file):
    """ 
        加载 mat 文件数据  
    Args:
        mat_file: 待加载的文件路径 (string)
    Returns:
        读取的数据 (list)，其中每个元素为一个嵌套的 dict 类型
    """
    # 函数返回一个字典，其中键是 mat 文件中变量的名称，值是对应的数据数组
    data = scipy.io.loadmat(mat_file)
    # 从文件路径中提取文件名(不包含扩展名),用于访问字典的值
    fileName = mat_file.split('/')[-1].split('.')[0]
    col = data[fileName] # 获取整个数据(一个(1 x N)的四层结构化数组)
    col = col[0][0][0][0] # 去除冗余维度，访问包含所有循环数据的(616,)结构化数组
    size = col.shape[0] # 获取数组的大小(cycle 的数量)
    # print("data['B0005'].dtype:",data['B0005'].dtype,"value:",data['B0005'])
    # print("data['B0005'][0][0][0][0].dtype:",data['B0005'][0][0][0][0].dtype,
    #       "value:",data['B0005'][0][0][0][0])
    # print("data['B0005'][0][0][0][0][0][3][0].dtype:",data['B0005'][0][0][0][0][0][3][0].dtype,
    #       "value:",data['B0005'][0][0][0][0][0][3][0])

    data = []
    for i in range(size): # 遍历每个 cycle 的数据
        """ dtype.fields 方法用于访问 NumPy 结构化数组的字段信息，它返回一个字典，其中：
        键: 是结构化数组中每个字段的名称（字符串）；
        值: 是描述每个字段的元组，包含字段的数据类型、字节偏移量以及可选的标题。 """
        k = list(col[i][3][0].dtype.fields.keys()) # 获取结构化数组(data 字段)中所有子字段名称的列表
        d1, d2 = {}, {}
        if str(col[i][0][0]) != 'impedance': # 去除 impedance 类型的数据
            for j in range(len(k)): # 遍历(data 字段)数组中的每个子字段
                t = col[i][3][0][0][j][0] # 获取该字段的数组数据
                l = [t[m] for m in range(len(t))] # 遍历提取数组中每个数据转为列表
                d2[k[j]] = l # 保存该数据及其对应的字段名称(以键值对的形式存在)
        # 将每个样本(cycle)的类型、温度、时间和数据存储到字典 d1 中
        d1['type'], d1['temp'], d1['time'], d1['data'] = \
            str(col[i][0][0]), int(col[i][1][0]), str(TimeConvert(col[i][2][0])), d2
        data.append(d1)

    return data

def GetBatteryCapacity(Battery):
    """ 
        获取单个锂电池的容量数据  
    Args:
        Battery: 单个电池的数据 (dict)
    Returns:
        获取的电池容量数据 (list)，包含两个元素，第一个为放电周期，第二个为容量数据
    """
    cycle, capacity = [], []
    i = 1
    for Bat in Battery:
        if Bat['type'] == 'discharge': # 放电状态下获取容量数据
            capacity.append(Bat['data']['Capacity'][0])
            cycle.append(i)
            i += 1
    return [cycle, capacity]

def GetBatteryValues(Battery, Type='charge'):
    """ 
        获取单个锂电池充电或放电时的测试数据(默认为充电状态的数据)  
    Args:
        Battery: 单个电池的数据 (dict)
        Type: 指定要读取的数据类型 (string)
    Returns:
        获取的电池数据， list 类型
    """
    data = []
    for Bat in Battery:
        if Bat['type'] == Type:
            data.append(Bat['data'])
    return data
# ------------------------------------------------------------------

# --- 定义数据集类 (TimeSeriesDataset) ---
class TimeSeriesDataset(Dataset):
    def __init__(self, data, window_size, pred_len=1):
        self.window_size = window_size
        self.pred_len = pred_len
        self.data = torch.tensor(data, dtype=torch.float32).to(device)
        self.max_index = self.data.shape[0] - self.window_size - pred_len

    def __len__(self):
        return self.max_index + 1

    def __getitem__(self, index):
        if index > self.max_index:
            raise IndexError(f"Index {index} is out of bounds."
                            f"Max index is {self.max_index}")
        source = self.data[index:index + self.window_size]
        target = self.data[index + 1:index + self.window_size + 1]
        return source.unsqueeze(1), target.unsqueeze(1)

# --- 生成自回归掩码张量 (gen_subsequent_mask) ---
def gen_subsequent_mask(size, n_head):
    mask_shape = (n_head, size, size)
    subsequent_mask = np.triu(np.ones(mask_shape), k=1).astype('uint8')
    return torch.from_numpy(1 - subsequent_mask).bool()

# --- 获取数据集函数 (get_data) ---
def get_data(data_dict, name, head=4, window_size=8, shuffle=True, batch_size=32):
    test_data = data_dict[name][1]
    test_dataset = TimeSeriesDataset(test_data, window_size)

    train_datasets = []
    for k, v in data_dict.items():
        if k != name:
            dataset = TimeSeriesDataset(v[1], window_size)
            train_datasets.append(dataset)
    train_dataset = ConcatDataset(train_datasets)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    src_mask = torch.ones((head, window_size, window_size), device=device).bool()
    tgt_mask = gen_subsequent_mask(window_size, head).to(device)
    return train_loader, test_loader, src_mask, tgt_mask

# --- 定义克隆函数 (clones) ---
def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

# --- 计算注意力张量 (attention) ---
def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask==False, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

# --- 定义降噪自编码类 (Autoencoder) ---
class Autoencoder(nn.Module):
    def __init__(self, input_size, auto_hidden, noise_level=0.01):
        super(Autoencoder, self).__init__()
        self.input_size = input_size
        self.auto_hidden = auto_hidden
        self.noise_level = noise_level
        self.fc1 = nn.Linear(self.input_size, self.auto_hidden)
        self.fc2 = nn.Linear(self.auto_hidden, self.input_size)

    def encoder(self, x):
        h1 = F.relu(self.fc1(x))
        return h1

    def mask(self, x):
        corrupted_x = x + self.noise_level * torch.randn_like(x)
        return corrupted_x

    def decoder(self, x):
        h2 = self.fc2(x)
        return h2

    def forward(self, x):
        out = self.mask(x)
        encode = self.encoder(out)
        decode = self.decoder(encode)
        return encode, decode

# --- 定义输入层类 (InputLayer) ---
class InputLayer(nn.Module):
    def __init__(self, features, d_model, noise_level=0.01, is_autoencoder=True):
        super(InputLayer, self).__init__()
        self.is_autoencoder = is_autoencoder
        self.noise_level = noise_level
        self.autoencoder = Autoencoder(features, d_model, noise_level)

    def forward(self, x):
        encode, decode = self.autoencoder(x)
        return encode if self.is_autoencoder else decode

# --- 定义位置编码类 (PositionalEncoding) ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=500):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)

# --- 定义多头注意力机制类 (MutiHeadAttention) ---
class MutiHeadAttention(nn.Module):
    def __init__(self, n_head, embedding_dim, dropout=0.1):
        super(MutiHeadAttention, self).__init__()
        assert embedding_dim % n_head == 0
        self.d_k = embedding_dim // n_head
        self.n_head = n_head
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        self.linears = clones(nn.Linear(embedding_dim, embedding_dim), 4)
        
    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(0)
        batch_size = query.size(0)
        query, key, value = \
            [model(x).view(batch_size, -1, self.n_head, self.d_k).transpose(1, 2) 
            for model, x in zip(self.linears, (query, key, value))]
        
        x, self.attn = attention(query, key, value, mask, dropout=self.dropout)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.n_head * self.d_k)
        return self.linears[-1](x)

# --- 位置编码的前馈全连接层类的定义 (PositionwiseFeedForward) ---
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        return self.w2(self.dropout(F.relu(self.w1(x))))
    
# --- 定义规范化层类 (NormLayer) ---
class NormLayer(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(NormLayer, self).__init__()
        self.parm1 = nn.Parameter(torch.ones(features))
        self.parm2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.parm1 * (x - mean) / (std + self.eps) + self.parm2
    
# --- 定义子层连接类 (SubLayerConnection) ---
class SubLayerConnection(nn.Module):
    def __init__(self, size, dropout=0.1):
        super(SubLayerConnection, self).__init__()
        self.norm = NormLayer(size)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(self.norm(sublayer(x)))
    
# --- 定义编码器层类 (EncoderLayer) ---
class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayers = clones(SubLayerConnection(size, dropout=dropout), 2)

    def forward(self, x, mask=None):
        x = self.sublayers[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayers[1](x, self.feed_forward)

# --- 定义编码器类 (Encoder) ---
class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = NormLayer(layer.size)

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
    
# --- 定义解码器层类 (DecoderLayer) ---
class DecoderLayer(nn.Module):
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayers = clones(SubLayerConnection(size, dropout=dropout), 3)

    def forward(self, x, memory, source_mask=None, target_mask=None):
        m = memory
        x = self.sublayers[0](x, lambda x: self.self_attn(x, x, x, target_mask))
        x = self.sublayers[1](x, lambda x: self.src_attn(x, m, m, source_mask))
        return self.sublayers[2](x, self.feed_forward)
    
# --- 定义解码器类 (Decoder) ---
class Decoder(nn.Module):
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = NormLayer(layer.size)

    def forward(self, x, memory, source_mask=None, target_mask=None):
        for layer in self.layers:
            x = layer(x, memory, source_mask, target_mask)
        return self.norm(x)

# --- 定义CNN层类 (CNNLayer) ---
class CNNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(CNNLayer, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=in_dim, 
                               out_channels=64, 
                               kernel_size=3, 
                               padding=1)
        self.conv2 = nn.Conv1d(in_channels=64, 
                               out_channels=out_dim, 
                               kernel_size=3, 
                               padding=1)
    
    def forward(self, x):
        x = F.relu(self.conv1(x.transpose(1, 2)))
        x = F.relu(self.conv2(x))
        return x.transpose(1, 2)
    
# --- 定义输出层类 (OutputLayer) ---
class OutputLayer(nn.Module):
    def __init__(self, in_cnn, out_cnn, output_dim=1):
        super(OutputLayer, self).__init__()
        self.cnn = CNNLayer(in_cnn, out_cnn)
        self.output = nn.Linear(out_cnn, output_dim)

    def forward(self, x):
        x = self.cnn(x)
        return self.output(x)[:, -1, :]

# --- 定义 TransformerCNN 模型类 (TransformerCNN) ---
class TransformerCNN(nn.Module):
    def __init__(self, features, d_model, out_cnn, head=4, d_ff=128,
                 output_dim=1, dropout=0.1, N=4, noise_level=0.01, is_autoencoder=False):
        super(TransformerCNN, self).__init__()
        c = copy.deepcopy
        self_attn = src_attn = MutiHeadAttention(head, d_model, dropout)
        ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.input = InputLayer(features, d_model, noise_level, is_autoencoder)
        self.pe = PositionalEncoding(d_model, dropout)
        self.encoder = Encoder(EncoderLayer(d_model, c(self_attn), c(ff), dropout), N)
        self.decoder = Decoder(DecoderLayer(d_model, c(self_attn), 
                                            c(src_attn), c(ff), dropout), N)
        self.output = OutputLayer(d_model, out_cnn, output_dim)

    def forward(self, source, target, source_mask, target_mask):
        source = self.pe(self.input(source))
        target = self.pe(self.input(target))
        memory = self.encoder(source, source_mask)
        decode = self.decoder(target, memory, source_mask, target_mask)
        return self.output(decode)

# --- 模型获取函数 (get_model) ---
def get_model(model_name, features, d_model, out_cnn, head=4, d_ff=2048, 
              output_dim=1, dropout=0.1, N=4, learn_rate=0.001, 
              noise_level=0.01, is_autoencoder=False):
    if model_name == 'TransformerCNN':
        model = TransformerCNN(features, d_model, out_cnn, head, d_ff, 
                            output_dim, dropout, N, noise_level, is_autoencoder)
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), 
                        lr=learn_rate, betas=(0.5,0.999))
    return model, loss_fn, optimizer

# --- 批训练函数 (train_batch) ---
def train_batch(source, target, source_mask, target_mask, model, optimizer, loss_fn):
    model.train()
    prediction = model(source, target, source_mask, target_mask)
    batch_loss = loss_fn(prediction, target[:, -1, :])
    batch_loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return batch_loss.item()

# --- 评估函数 (evaluation) ---
def relative_error(y_test, y_predict, threshold):
    true_re, pred_re = len(y_test), 0
    
    for i in range(len(y_test) - 1):
        if y_test[i] <= threshold >= y_test[i+1]:
            true_re = i - 1
            break
    for i in range(len(y_predict) - 1):
        if y_predict[i] <= threshold:
            pred_re = i - 1
            break
    score = abs(true_re - pred_re) / true_re
    if score > 1: score = 1
    
    return score

def evaluation(y_test, y_predict):
    rmse = sqrt(mean_squared_error(y_test, y_predict))
    crmsd = np.sqrt(np.mean(((y_test - np.mean(y_test)) - \
                             (y_predict - np.mean(y_predict)))**2))
    mad = np.median(np.abs(y_test - y_predict))
    mae = mean_absolute_error(y_test, y_predict)
    mbe = np.abs(np.mean(y_predict - y_test))
    rsquare = r2_score(y_test, y_predict)
    metrics_dict = {
        "RMSE": rmse,
        "CRMSD": crmsd,
        "MAD": mad,
        "MAE": mae,
        "MBE": mbe,
        "R2": rsquare
    }
    return metrics_dict

# --- 设置环境随机种子 (setup_seed) ---
def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

# -----------------------------------------------------------------------------
Battery_list = ['B0005', 'B0006', 'B0007', 'B0018']
dir_path = r'BatteryDataset/'

capacity, charge, discharge = {}, {}, {}
for name in Battery_list:
    print('Loading Dataset ' + name + '.mat ...')
    path = dir_path + name + '.mat'
    data = LoadMat(path)
    capacity[name] = GetBatteryCapacity(data) # 放电时的容量数据
    charge[name] = GetBatteryValues(data, 'charge') # 充电数据
    discharge[name] = GetBatteryValues(data, 'discharge') # 放电数据


# --- 性能测试参数设置 ---
epochs = 50 # 减少 epochs 以加快测试速度
seed = 42
model_name = 'TransformerCNN'

features = 1
output_dim = 1
dropout = 0.1
noise_level = 0.0
batch_size = 32
Rated_Capacity = 2.0
shuffle = True
is_autoencoder = True

window_size = 9
d_model = 16
out_cnn = 32
head = 2
d_ff = 121
N = 4
learn_rate = 0.00165

setup_seed(seed)

# --- 汇总性能指标的列表 ---
all_train_times_per_epoch = []
all_inference_times_per_batch = []
all_max_gpu_memory_train = []
all_max_gpu_memory_inference = []

print(f"--- 性能测试开始，运行在设备: {device} ---")

# --- GPU 内存初始化和清空 ---
if device.startswith('cuda'):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    initial_gpu_memory = torch.cuda.memory_allocated() / (1024 * 1024)
    print(f"初始 GPU 内存占用: {initial_gpu_memory:.2f} MB")
else:
    print("GPU 不可用，内存测量将针对 CPU (精度较低)。")


# --- 遍历所有电池数据进行训练和推理 (交叉验证模拟) ---
for i, bat_name in enumerate(Battery_list):
    print(f"\n======== 开始测试电池: {bat_name} ({i+1}/{len(Battery_list)}) ========")

    # 获取数据加载器和掩码
    train_loader, test_loader, src_mask, tgt_mask = \
        get_data(capacity, bat_name, head, window_size, shuffle, batch_size)

    # 每次循环都重新初始化模型，以确保独立性 (重要，否则模型会持续学习)
    model, loss_fn, optimizer = get_model(model_name, features, d_model, out_cnn,
                                          head, d_ff, output_dim, dropout, N,
                                          learn_rate, noise_level, is_autoencoder)
    model = model.to(device)

    print(f"\n--- 开始训练模型 ({model_name})，测试电池: {bat_name} ---")

    # --- 训练时间与内存测量 ---
    train_epoch_times = []
    if device.startswith('cuda'):
        torch.cuda.empty_cache() # 清空缓存
        torch.cuda.reset_peak_memory_stats() # 重置峰值内存统计
    
    train_start_time_overall = time.time()
    for epoch in range(int(epochs)):
        model.train() # 设置为训练模式
        epoch_start_time = time.time()
        for index, batch in enumerate(iter(train_loader)):
            src, tgt = batch
            # 归一化
            src /= torch.tensor(Rated_Capacity).to(device)
            tgt /= torch.tensor(Rated_Capacity).to(device)
            batch_loss = train_batch(src, tgt, src_mask, tgt_mask, model, optimizer, loss_fn)
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        train_epoch_times.append(epoch_duration)
        print(f"Epoch {epoch+1}/{epochs} - 训练耗时: {epoch_duration:.4f} s")
    train_end_time_overall = time.time()

    avg_train_time_per_epoch = np.mean(train_epoch_times)
    all_train_times_per_epoch.append(avg_train_time_per_epoch)

    if device.startswith('cuda'):
        max_gpu_memory_train_current = torch.cuda.max_memory_allocated() / (1024 * 1024) # 记录当前电池训练峰值内存
        all_max_gpu_memory_train.append(max_gpu_memory_train_current)
        torch.cuda.empty_cache() # 在推理前清空缓存，以便更准确测量推理内存

    print(f"--- 训练完成，测试电池: {bat_name} ---")

    # --- 推理时间与内存测量 ---
    model.eval() # 设置为评估模式
    if device.startswith('cuda'):
        torch.cuda.reset_peak_memory_stats() # 在推理前重置峰值内存统计
    inference_batch_times_current = []
    inference_batch_count_current = 0

    print(f"\n--- 开始推理模型 ({model_name}) ---")
    with torch.no_grad(): # 在推理时禁用梯度计算
        for index, batch in enumerate(iter(test_loader)):
            src, tgt = batch
            # 归一化
            src /= torch.tensor(Rated_Capacity).to(device)
            tgt /= torch.tensor(Rated_Capacity).to(device)

            batch_inference_start_time = time.time()
            pred = model(src, tgt, src_mask, tgt_mask)
            batch_inference_end_time = time.time()
            inference_batch_times_current.append(batch_inference_end_time - batch_inference_start_time)
            inference_batch_count_current += 1

    avg_inference_time_per_batch_current = (sum(inference_batch_times_current) / inference_batch_count_current) if inference_batch_count_current > 0 else 0
    all_inference_times_per_batch.append(avg_inference_time_per_batch_current)

    if device.startswith('cuda'):
        max_gpu_memory_inference_current = torch.cuda.max_memory_allocated() / (1024 * 1024) # 记录当前电池推理峰值内存
        all_max_gpu_memory_inference.append(max_gpu_memory_inference_current)

    print(f"--- 推理完成 ---")
    print(f"当前电池 {bat_name} - 平均训练时间/epoch: {avg_train_time_per_epoch:.4f} s")
    print(f"当前电池 {bat_name} - 平均推理时间/batch: {avg_inference_time_per_batch_current:.4f} s")
    if device.startswith('cuda'):
        print(f"当前电池 {bat_name} - 训练期间峰值 GPU 内存: {max_gpu_memory_train_current:.2f} MB")
        print(f"当前电池 {bat_name} - 推理期间峰值 GPU 内存: {max_gpu_memory_inference_current:.2f} MB")


# --- 打印最终汇总性能指标 ---
print(f"\n======== 总体性能指标 ({model_name} 在所有电池上) ========")
print(f"**注意：此结果基于 batch_size={batch_size}，更改 batch_size 将显著影响性能。**")
print(f"平均训练时间 (每 epoch): {np.mean(all_train_times_per_epoch):.4f} s/epoch")
print(f"平均推理时间 (每 batch): {np.mean(all_inference_times_per_batch):.4f} s/batch")

if device.startswith('cuda'):
    print(f"训练期间平均峰值 GPU 内存占用: {np.mean(all_max_gpu_memory_train):.2f} MB")
    print(f"推理期间平均峰值 GPU 内存占用: {np.mean(all_max_gpu_memory_inference):.2f} MB")
    print(f"所有训练阶段中最大峰值 GPU 内存占用: {np.max(all_max_gpu_memory_train):.2f} MB")
    print(f"所有推理阶段中最大峰值 GPU 内存占用: {np.max(all_max_gpu_memory_inference):.2f} MB")
else:
    print("当前运行在 CPU 上，无法提供精确的 GPU 内存使用量。")
print("==========================================================")