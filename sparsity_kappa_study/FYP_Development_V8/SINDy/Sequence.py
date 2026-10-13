from __future__ import annotations
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Sequence():
    """单条序列类(1xN)"""
    def __init__(self,Data,Name='DefaultSeq'):
        """序列数据"""
        self.Data = np.asarray(Data).flatten().tolist()
        """序列名称"""
        self.Name = Name
        """序列长度"""
        self.N = len(self.Data)

    def print(self):
        """打印序列信息"""
        print(f"序列：{self.Name}")
        print(f"序列长度:{self.N}")
        print(f"序列数据(前5个):{self.Data[0:5]}")

    def simpleplot(self):
        """简单绘制曲线"""
        plt.figure()
        x = np.linspace(0,self.N,self.N)
        plt.plot(x,self.Data,label = self.Name)
        plt.grid("on")
        plt.legend()
        plt.show()

    def plot(self,ax):
        """在给定的ax上简单绘制曲线"""
        x = np.linspace(0,self.N,self.N)
        ax.plot(x,self.Data,label = self.Name)
        ax.grid("on")
        ax.legend()

    def save(self,path = 'Output/DefaultSeq.csv', precision = 5):
        """保存为简单的序列CSV，按照列存储"""

        # 确保存储路径存在
        Path = os.path.dirname(path)
        if Path:
            os.makedirs(Path, exist_ok=True)

        # 转化为DataFrame,然后保存
        df = pd.DataFrame({self.Name: self.Data})
        float_format = f'%.{precision}e'
        df.to_csv(path, index=False, float_format=float_format)
        print(f"序列{self.Name}已保存到{path}")

    def sample(self, rate):
        """对序列按采样率进行采样"""
        
        sampled_data = self.Data[::rate]
        
        # 返回新的Sequence实例
        return Sequence(Data=sampled_data, Name=self.Name)

    def addseq(self, Seq: Sequence, k =1.0, NewName=None):
        """将当前序列与其他序列进行相加操作"""
        NewData = np.array(self.Data) + k*np.array(Seq.Data)
        if NewName==None:
            NewName = self.Name
        return Sequence(NewData,NewName)

    def addnoise(self, mu, sigma):
        """根据给定的均值和方差，对数据添加噪声"""
        noise = np.random.normal(loc=mu, scale=sigma, size=len(self.Data))
        return self.addseq(Sequence(noise),k=1.0,NewName=self.Name)

    def filter(self, Interval):
        """根据所需要的区间提取序列"""
        start,end = Interval
        return Sequence(self.Data[start:end],self.Name)

    def linearinterp(self,rate):
        """按照采样率对样本进行线性插值补全数据"""
        n = self.N
        
        # 原数据的索引坐标（视为等间隔的x轴）
        original_indices = np.arange(n)
        
        # 计算插值后的新索引坐标：在原索引间均匀插入点
        # 新数据长度 = (原长度-1)*rate + 1（保证首尾点与原数据一致）
        new_length = (n - 1) * rate + 1
        new_indices = np.linspace(start=0, stop=n-1, num=new_length, endpoint=True)
        
        # 执行线性插值
        interpolated_data = np.interp(new_indices, original_indices, self.Data)
        
        return Sequence(Data=interpolated_data, Name=self.Name)

    def smooth(self, window_size):
        """根据窗口大小对数据进行平滑处理"""
        from scipy.signal import savgol_filter
        
        # 参数验证
        if window_size > len(self.Data):
            raise ValueError(f"窗口大小({window_size})不能超过序列长度({len(self.Data)})")
        
        # 确保窗口大小为奇数
        if window_size % 2 == 0:
            window_size = window_size + 1
            print(f"警告：窗口大小自动调整为奇数: {window_size}")
        
        # 使用Savitzky-Golay滤波（自动处理边界）
        smoothed_data = savgol_filter(self.Data, window_length=window_size, polyorder=3,mode='mirror')

        return Sequence(Data=smoothed_data, Name=self.Name)

    def unify(self):
        """将序列归一化到[-1,1]之间"""
        coeff = np.max(np.abs(self.Data))

        return coeff,Sequence(self.Data/coeff,self.Name)

class SequenceData():
    """多条序列组类(NumxN)"""
    def __init__(self,SeqData:list[Sequence],Name='DefaultSeqData'):
        
        """序列组数据"""
        self.SeqData = SeqData
        """序列组的名称"""
        self.Name = Name
        """序列组个数Num"""
        self.Num = len(SeqData)
        """序列组的公共长度N"""
        self.N = SeqData[0].N

        # 检查序列的长度是否合法
        self._check_length()

    @ classmethod
    def from_csv(cls,Path,Name='DefaultSeqData',Col=None):
        """从CSV中读取指定的列并建立SequenceData类"""
        
        # 读取CSV文件
        df = pd.read_csv(Path)

        # 处理Col参数，确定需要读取的列
        all_cols = df.columns.tolist()
        
        if Col is None:
            selected_cols = all_cols
        else:
            if isinstance(Col, str):
                selected_cols = [Col]
            elif isinstance(Col, list):
                selected_cols = Col
            else:
                raise TypeError("Col参数必须是字符串、列表或None")
            
            # 验证选中的列是否都存在于CSV中
            invalid_cols = [col for col in selected_cols if col not in all_cols]
            if invalid_cols:
                raise ValueError(f"CSV中不存在以下列：{invalid_cols}，有效列名：{all_cols}")

        # 将每一列转换为Sequence对象
        seq_list = []
        for col_name in selected_cols:
            col_data = df[col_name].values
            seq = Sequence(Data=col_data, Name=col_name)
            seq_list.append(seq)
        
        # 用Sequence列表初始化SeqData实例并返回
        return cls(SeqData=seq_list, Name=Name)

    @ classmethod
    def from_concat(cls, seq_data1: 'SequenceData', seq_data2: 'SequenceData', Name: str = None):
        """合并两个SequenceData实例，返回新的SequenceData"""
        all_seqs = seq_data1.SeqData + seq_data2.SeqData
        
        # 检查序列名称是否重复
        seq_names = [seq.Name for seq in all_seqs]
        duplicate_names = [name for name in set(seq_names) if seq_names.count(name) > 1]
        if duplicate_names:
            raise ValueError(f"合并失败：存在重复序列名称 {duplicate_names}，请先重命名")
        
        # 生成新序列组名称
        if Name is None:
            Name = f"{seq_data1.Name}+{seq_data2.Name}"
        
        # 创建新的SequenceData
        return cls(all_seqs, Name)

    def to_tuple(self):
        """将序列数据转化为原先的元组格式"""
        return tuple(np.array(seq.Data) for seq in self.SeqData)

    def get(self, Name):
        """按名称获取序列"""
        for seq in self.SeqData:
            if seq.Name == Name:
                return seq
        raise KeyError(f"序列'{Name}'不存在")

    def print(self):
        """打印序列组信息"""
        print(f"序列组：{self.Name}")
        print(f"序列组序列个数:{self.Num}")
        print(f"序列组的公共长度:{self.N}")
        for i in range(self.Num):
            self.SeqData[i].print()

    def plot_all(self):
        """将所有的序列数据都简单绘制出来"""
        fig = plt.figure()
        for i in range(self.Num):
            ax = fig.add_subplot(self.Num,1,i+1)
            Seq = self.SeqData[i]
            Seq.plot(ax)
        plt.show()

    def plot_timeseries(self,idx):
        """指定某列为时间，绘制数据随时间的变化"""
        fig = plt.figure(figsize=(10,10))
        count = 0
        TimeLabel = self.SeqData[idx].Name
        for i in range(self.Num):
            if i == idx:
                continue
            count+=1

            ax = fig.add_subplot(self.Num-1,1,count)
            X = self.SeqData[idx].Data
            Y = self.SeqData[i].Data
            Label = self.SeqData[i].Name

            ax.plot(X,Y,label = Label)
            ax.set_xlabel(TimeLabel)
            ax.set_ylabel(Label)
            ax.grid("on")
            ax.legend()
        plt.tight_layout()
        plt.show()

    def save(self,path = 'Output/DefaultSeqData.csv', precision = 5):
        """保存为简单的序列CSV，按照列存储"""

        # 确保存储路径存在
        Path = os.path.dirname(path)
        if Path:
            os.makedirs(Path, exist_ok=True)

        # 建立数据字典，添加所有信息
        data_dict = {}
        for seq  in self.SeqData:
            data_dict[seq.Name] = seq.Data

        # 将数据字典转化为DataFrame并保存
        df = pd.DataFrame(data_dict)
        float_format = f'%.{precision}e'
        df.to_csv(path, index=False, float_format=float_format)
        print(f"序列组{self.Name}已保存到{path}")
    
    def sample(self, rate):
        """对序列组中所有序列进行采样"""
        sampled_seqs = [seq.sample(rate) for seq in self.SeqData]
        
        return SequenceData(SeqData=sampled_seqs, Name=self.Name)
    
    def addnoise(self, idx_list, mu_list, sigma_list):
        """对序列组中idx_list的序列按照顺序增加噪音"""
        new_seqs = self.SeqData.copy()
        for i, idx in enumerate(idx_list):
            new_seqs[idx] = self.SeqData[idx].addnoise(mu_list[i], sigma_list[i])
        return SequenceData(SeqData=new_seqs, Name=self.Name)
    
    def smooth(self,window,idx):
        """对整个数据除了idx列，都进行平滑处理"""

        if window !=0:
            new_seqs = []
            for i in range(self.Num):
                if i == idx:
                    new_seqs.append(self.SeqData[i])
                else:
                    smoothed_seq = self.SeqData[i].smooth(window_size=window)
                    new_seqs.append(smoothed_seq)
            
            return SequenceData(SeqData=new_seqs,Name=self.Name)

        else:
            return self

    def filter(self, interval):
        """对每个序列进行按照interval筛选"""
        # start,end = interval

        return SequenceData(SeqData = [seq.filter(interval) for seq in self.SeqData], Name=self.Name)
    
    def linearinterp(self,rate):
        """对序列组中所有序列进行线性插值"""
        interp_seqs = [seq.linearinterp(rate) for seq in self.SeqData]

        return SequenceData(SeqData=interp_seqs, Name=self.Name)

    def _check_length(self):
        """检测所有序列长度是否一致，不一致则抛出异常"""
        base_length = self.SeqData[0].N
        # 遍历所有序列检查长度
        for i, seq in enumerate(self.SeqData):
            if seq.N != base_length:
                # 若长度不一样，反馈并报错
                lengths = []
                for i, seq in enumerate(self.SeqData, 1):
                    seq_len = seq.N
                    lengths.append(seq_len)
                    print(f"第{i}条序列（名称：{seq.Name}）长度：{seq_len}")
                raise ValueError(f"序列长度不一致！")

    def extract_params(self, param_names: list, index: int = 0) -> dict:
        """批量提取指定参数在指定索引处的值，并转换为NumPy数组"""
        params = {}
        for name in param_names:
            params[name] = self.get(name).Data[index]
        return params

def Save_Matrix(matrix, File, row_headers=None, column_headers=None, precision = 8, output = True):
    """保存SINDy回归的矩阵信息"""
    # 构造 DataFrame
    if row_headers is not None and column_headers is not None:
        matrix_df = pd.DataFrame(matrix, index=row_headers, columns=column_headers)
    elif row_headers is not None:
        matrix_df = pd.DataFrame(matrix, index=row_headers)
    elif column_headers is not None:
        matrix_df = pd.DataFrame(matrix, columns=column_headers)
    else:
        matrix_df = pd.DataFrame(matrix)

    # 保存为 CSV，使用科学计数法格式化
    matrix_df.to_csv(File, encoding='utf-8-sig', float_format=f'%.{precision}e', index=(row_headers is not None))

    if output:
        print(f"矩阵已成功保存至: {File}")

