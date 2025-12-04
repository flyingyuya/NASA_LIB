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
    def __init__(self, data, window_size):
        self.window_size = window_size
        self.data = torch.tensor(data, dtype=torch.float32).to(device)
        # 计算最大索引
        self.max_index = self.data.shape[0] - self.window_size - 1

    def __len__(self):
        return self.max_index + 1 #  返回有效数据长度

    def __getitem__(self, index):
        if index > self.max_index:
            raise IndexError(f"Index {index} is out of bounds."
                            f"Max index is {self.max_index}")
        x = self.data[index:index + self.window_size]
        y = self.data[index + self.window_size]
        return x.unsqueeze(1), y.unsqueeze(0) # 添加一个特征维度

# --- 获取数据集函数 (get_data) ---
def get_data(data_dict, name, window_size=8, shuffle=True, batch_size=32):
    """ 
        留一法获取训练集和测试集 DataLoader,每次留一个电池的数据作为测试集  
    Args:  
        data_dict: 字典类型，键为电池名称，值为包含电池信息的元组，
                其中第二个元素是容量数据列表 (list)  
        name: 指定为测试集的电池数据名称 (str)  
        window_size: 用于创建时间序列的窗口大小 (int)  
        shuffle: 是否打乱训练集 (bool)  
        batch_size: 训练的批大小 (int)  
    Returns:
        包含训练数据和测试数据 DataLoader 的元组。
    """
    test_data = data_dict[name][1]
    test_dataset = TimeSeriesDataset(test_data, window_size)

    train_datasets = []
    for k, v in data_dict.items():
        if k != name:
            dataset = TimeSeriesDataset(v[1], window_size)
            train_datasets.append(dataset)
    train_dataset = ConcatDataset(train_datasets) # 使用 ConcatDataset 拼接多个数据集

    # 创建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 打印 DataLoader
    # print("train_loader_num:", len(train_loader.dataset))
    # print("test_loader_num:", len(test_loader.dataset))
    # for x, y in train_loader:
    #     print("x:", x.shape) # 输出: x: (batch_size, window_size, num_features)
    #     print("y:", y.shape) # 输出: y: (batch_size, 1)
    #     break
    return train_loader, test_loader


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
                         self.hidden_size).to(device) # 初始化隐藏状态h0
        c0 = torch.zeros(self.num_layers*2 if self.bidirectional else 
                         self.num_layers, x.size(0), 
                         self.hidden_size).to(device)  # 初始化记忆状态c0
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

# 建立组合模型 CNN-LSTM-Attention
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
    
    # 建立组合模型 CNN-LSTM
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

# 建立组合模型 LSTM-Attention
class LAM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, 
                 bidirectional, step_dim, output_dim):
        super(LAM, self).__init__()
        self.lstm = LSTMLayer(input_dim, hidden_dim, 
                              num_layers, bidirectional)
        self.attention = AttentionLayer(hidden_dim * 2 if bidirectional else 
                                        hidden_dim, step_dim)
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
    
    def forward(self, x):
        x = self.lstm(x)
        x, _ = self.attention(x)
        x = self.fc(x)
        # x = self.fc(x[:, -1, :]) # 取序列最后一个时间步的输出作为预测
        return x

    # 建立模型 LSTM
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

# --- 模型获取函数 (get_model) ---
def get_model(num_channels, out_dim, kernel_size, feature_dim, hidden_dim, num_layers, 
              bidirectional, step_dim, output_dim, learn_rate, model_name='BNO-CBiLAM'):
    """ 
        获取模型，并指定优化器和损失计算方法
    Args:
        num_channels: 输入模型的通道数，即窗口大小
        out_dim: 卷积的输出特征维度
        feature_dim: 输入数据的特征维度
        kernel_size: 卷积核大小
        hidden_dim: LSTM隐藏状态维度
        num_layers: LSTM层的数目
        bidirectional: 是否使用双向LSTM
        step_dim: 时间步维度
        output_dim: 输出维度(预测目标维度)
        model_name: 指定的模型名称
        learn_rate: 学习率
    Returns:
        指定的模型、损失函数和优化器的元组
    """
    if model_name == 'BNO-CBiLA':
        model = CLAM(num_channels, out_dim, kernel_size, hidden_dim, 
                    num_layers, bidirectional, step_dim, output_dim)
    elif model_name == 'CBiLA':
        model = CLAM(num_channels, out_dim, kernel_size, hidden_dim, 
                num_layers, bidirectional, step_dim, output_dim)
    elif model_name == 'CBiL':
        model = CLM(num_channels, out_dim, kernel_size, hidden_dim, 
                    num_layers, bidirectional, output_dim)
    elif model_name == 'BiLSTM':
        model = LM(feature_dim, hidden_dim, num_layers, 
                    bidirectional, output_dim)
    elif model_name == 'CNN':
        model = CNN(num_channels, out_dim, output_dim)
    else:
        model = LM(feature_dim, hidden_dim, num_layers, 
                   bidirectional=False, output_dim=output_dim)
    loss_fn = nn.MSELoss() # 使用均方误差
    optimizer = optim.Adam(model.parameters(), 
                           lr=learn_rate, betas=(0.5,0.999)) # 使用Adam优化器
    return model, loss_fn, optimizer

# --- 批训练函数 (train_batch) ---
def train_batch(x, y, model, optimizer, loss_fn):
    """ 
        批训练函数
    Args:
        x: 输入的训练数据
        y: 输入的真实目标数据
        model: 指定的模型
        optimizer: 指定的优化器
        loss_fn: 指定的损失函数
    Returns:
        计算的损失标量
    """
    model.train() # 设置为训练
    prediction = model(x) # 输入数据
    # print("Prediction shape:", prediction.shape)
    batch_loss = loss_fn(prediction, y) # 计算损失
    batch_loss.backward() # 进行反向传播
    optimizer.step() # 梯度下降
    optimizer.zero_grad() # 清空梯度
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
model_name = 'LSTM'

features = 1
output_dim = 1
dropout = 0.1
batch_size = 32
Rated_Capacity = 2.0
shuffle = True

window_size = 16
out_dim = 16
kernel_size = 3
num_layers = 1
step_dim = 32
learn_rate = 0.001
hidden_dim = 32
bidirectional = False
learn_rate = 0.001

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
    train_loader, test_loader = get_data(capacity, name, window_size, shuffle, batch_size)

    # 每次循环都重新初始化模型，以确保独立性 (重要，否则模型会持续学习)
    model, loss_fn, optimizer = get_model(features, out_dim, kernel_size, 
                    features, hidden_dim, num_layers, 
                    bidirectional, step_dim, output_dim, 
                    learn_rate, model_name)
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
            x, y = batch
            # 归一化
            x /= torch.tensor(Rated_Capacity).to(device)
            y /= torch.tensor(Rated_Capacity).to(device)
            batch_loss = train_batch(x, y, model, optimizer, loss_fn)
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
            x, y = batch
            # 归一化
            x /= torch.tensor(Rated_Capacity).to(device)
            y /= torch.tensor(Rated_Capacity).to(device)

            batch_inference_start_time = time.time()
            pred = model(x)
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