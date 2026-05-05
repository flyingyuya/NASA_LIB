import numpy as np
import scipy.io
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import ListedColormap

import matplotlib as mpl
mpl.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'mathtext.fontset': 'stix',
    'axes.unicode_minus': False
})


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
# -----------------------------------------------------------------------
def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=256):
    """截取 colormap 的中间部分，避免两端过深的颜色"""
    new_cmap = ListedColormap(cmap(np.linspace(minval, maxval, n)))
    return new_cmap
# -----------------------------------------------------------------------
Battery_list = ['B0005', 'B0006', 'B0007', 'B0018']
dir_path = r'BatteryDataset/'

capacity = {}
cc = {}
dc = {}
for name in Battery_list:
    print('Loading Dataset ' + name + '.mat ...')
    path = dir_path + name + '.mat'
    data = LoadMat(path)
    capacity[name] = GetBatteryCapacity(data) # 放电时的容量数据
    cc[name] = GetBatteryValues(data, 'charge') # 充电数据
    dc[name] = GetBatteryValues(data, 'discharge') # 放电数据
# -----------------------------------------------------------------------
fig = plt.figure(figsize=(14, 8))

gs = GridSpec(2, 2, figure=fig, width_ratios=[2.5, 1.0], height_ratios=[1.05, 1.0], wspace=0.15, hspace=0.15)
ax0 = fig.add_subplot(gs[:, 0])   # 第一列占满
ax1 = fig.add_subplot(gs[0, 1])   # 第一行第二列
ax2 = fig.add_subplot(gs[1, 1])   # 第二行第二列

color_list = ['blue','green','red','c']
c = 0
for name,color in zip(Battery_list, color_list):
    df_result = capacity[name]
    ax0.plot(df_result[0], df_result[1], 
            linestyle='--', linewidth=0.4, 
            marker='o', markerfacecolor=color, 
            markeredgecolor=color, markersize=6, 
            alpha=0.6, label=name)
# 临界点直线(电池容量下降30%则认为报废)
# plt.plot([-1,170],[2.0*0.7,2.0*0.7],c='black',lw=1,ls='--')
ax0.set(xlabel='Discharge Cycle', 
        ylabel='Capacity (Ah)', 
        # title='Capacity degradation at ambient temperature of 24°C'
)
ax0.set_facecolor("#fafafa00")                          # 可选：浅色背景
ax0.grid(True, which='major', linestyle='--', alpha=0.6, linewidth=0.8)
ax0.minorticks_on(); ax0.grid(True, which='minor', linestyle=':', alpha=0.3, linewidth=0.5)
ax0.legend()


name = 'B0005' # 仅绘制第 B0005 号电池
cmap = plt.get_cmap('viridis') # 选择合适的颜色映射, 'hsv', 'jet', 'viridis' 等
cmap = truncate_colormap(plt.get_cmap('viridis'), minval=0.2, maxval=0.8)

for i, cycle_data in enumerate(dc[name]):
    # 使用 plot 绘制，并根据循环次数着色
    ax1.plot(cycle_data['Time'], cycle_data['Voltage_measured'], 
            c=cmap(i / len(dc[name])), linewidth=5.0, alpha=0.5)
ax1.set(
    # xlabel='Time (s)', 
    ylabel='Voltage (V)', 
    # title='Voltage vs. Time for Discharge Cycles'
)

for i, cycle_data in enumerate(dc[name]):
    # 使用 plot 绘制，并根据循环次数着色
    ax2.plot(cycle_data['Time'], cycle_data['Current_measured'], 
            c=cmap(i / len(dc[name])), linewidth=5.0, alpha=0.5)
ax2.set(
    xlabel='Time (s)', 
    ylabel='Current (A)', 
    # title='Current vs. Time for Discharge Cycles'
)

# 单独给颜色条留一个独立轴，不挤压任何子图
cax = fig.add_axes([0.91, 0.11, 0.015, 0.77])
# 添加颜色条
sm = plt.cm.ScalarMappable(cmap=cmap, 
                           norm=plt.Normalize(vmin=0, vmax=len(dc[name])))
sm.set_array([])  # 这是为了让colorbar工作，即使没有明确的映射数组
cbar = fig.colorbar(sm, cax=cax)
cbar.set_label('Cycle')

for a in (ax0, ax1, ax2):
    a.tick_params(direction='in', which='both', top=True, right=True,
                  length=3, width=1)   # 主/次刻度都向内
    # a.minorticks_on()             # 若需要次刻度
# colorbar 的刻度也改为向内
cbar.ax.tick_params(direction='in', which='both', length=3, width=1)

fig.canvas.draw()                 # 强制渲染以获得正确位置
fig.align_ylabels([ax0, ax1, ax2])

# plt.show()
fig.savefig('img/capacity_dc_cc_for_battery.png', dpi=300, bbox_inches='tight')
# -----------------------------------------------------------------------