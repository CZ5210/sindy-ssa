from sympy import *
import numpy as np
from .Sequence import *
from Dynamics.MEEDynamics import MEE_Dynamics,MEEdot2Acc

def ObserveSat(SamplingRate, NoiseStd, FilePath, ObservePath):
    """模拟卫星观测，将所有可用数据集中起来"""
    # 读取数据
    DataCol = ['Time','p','f','g','h','k','L','Apn_r','Apn_t','Apn_n','Flag']
    ObserveData = SequenceData.from_csv(Path=FilePath,Col=DataCol,Name="ElementData")

    # 对MEE数据增加一个模拟的噪音
    ObserveData = ObserveData.addnoise(list(range(1,7)),np.zeros((6)),NoiseStd)

    # 根据观测频率进行采样后保存
    ObserveData = ObserveData.sample(SamplingRate)
    ObserveData.save(ObservePath)

    return

def GVE_MEE(ObserveData:SequenceData):
    """使用GVE摄动方法计算自然MEEdot"""

    _, p, f, g, h, k, L,apnr,apnt,apnn,_ = ObserveData.to_tuple()
    MEE = p, f, g, h, k, L
    Acc = apnr, apnt, apnn
    meedot = MEE_Dynamics(MEE,Acc)

    # 打包转化为SequenceData并返回
    TimeData = ObserveData.get("Time")
    cols1 = ['Time', 'p_dot', 'f_dot', 'g_dot', 'h_dot', 'k_dot', 'L_dot']
    MEEdot_natural = SequenceData([TimeData] + [Sequence(data, name) for data, name in zip(meedot, cols1[1:])],"MEEdotData")

    return MEEdot_natural

def FD_MEE(ObserveData:SequenceData):
    """混合差分方法求解MEE数据"""

    # 读取数据并展开角度L
    Time, p, f, g, h, k, L,_,_,_,_ = ObserveData.to_tuple()
    L = np.unwrap(L)
    dt = Time[1] - Time[0]

    # 中心差分方法，使用np.gradient更简洁表述
    pdot = np.gradient(p, dt)
    fdot = np.gradient(f, dt)
    gdot = np.gradient(g, dt)
    hdot = np.gradient(h, dt)
    kdot = np.gradient(k, dt)
    Ldot = np.gradient(L, dt)

    # 打包转化为SequenceData并返回
    MEEdot = [Time, pdot, fdot, gdot, hdot, kdot, Ldot]
    cols = ['Time', 'p_dot', 'f_dot', 'g_dot', 'h_dot', 'k_dot', 'L_dot']
    MEEdotData = SequenceData([Sequence(data, name) for data, name in zip(MEEdot, cols)],"MEEdotFD")

    return MEEdotData

def Element2AccData(ObserveData:SequenceData,MEEdotData:SequenceData):
    """将MEEdot转化为加速度"""

    Time,p,f,g,h,k,L,_,_,_,_ = ObserveData.to_tuple()
    Time,pdot,fdot,gdot,hdot,kdot,Ldot = MEEdotData.to_tuple()

    n = len(Time)
    Acc = np.zeros((n,3))
    for i in range(n):
        mee_i = np.array([p[i], f[i], g[i], h[i], k[i], L[i]])
        meedot_i = np.array([pdot[i], fdot[i], gdot[i], hdot[i], kdot[i], Ldot[i]])
        Acc[i] = MEEdot2Acc(mee_i,meedot_i)

    acc = [Time, Acc[:,0],Acc[:,1],Acc[:,2]]
    cols = ['Time', 'a_r', 'a_t', 'a_n']
    AccData = SequenceData([Sequence(data, name) for data, name in zip(acc, cols)],"AccData")

    return AccData

def Extract_Manoeuvre(Path):
    """提取机动段时间区间"""
    # 读取Flag列
    flag_data = SequenceData.from_csv(Path=Path,Col=['Flag'],Name='FlagData')
    flag_seq = np.array(flag_data.get('Flag').Data)

    # 识别机动段起止时间
    intervals = []
    count = 0
    in_manoeuvre = False
    current_start = None
    
    for i in range(len(flag_seq)):
        flag = flag_seq[i]
        
        # 进入机动段
        if flag == 'manoeuvre' and not in_manoeuvre:
            current_start = i
            in_manoeuvre = True
        
        # 退出机动段
        elif flag == 'normal' and in_manoeuvre:
            intervals.append((current_start, i))
            count += 1
            in_manoeuvre = False
            current_start = None
    
    # 处理末尾仍在机动段的情况
    if in_manoeuvre and current_start is not None:
        intervals.append((current_start, i))
        count += 1
    
    return count, intervals

def MinusAcc(acc_data1: SequenceData, acc_data2: SequenceData):
    """将两个加速度序列组的对应分量"""
    
    time = acc_data1.get('Time')
    # 提取加速度分量序列
    a1_r = acc_data1.get('a_r')
    a1_t = acc_data1.get('a_t')
    a1_n = acc_data1.get('a_n')
    
    a2_r = acc_data2.get('a_r')
    a2_t = acc_data2.get('a_t')
    a2_n = acc_data2.get('a_n')
    
    # 分量相减
    diff_r = a1_r.addseq(a2_r, k=-1.0, NewName='u_r')
    diff_t = a1_t.addseq(a2_t, k=-1.0, NewName='u_t')
    diff_n = a1_n.addseq(a2_n, k=-1.0, NewName='u_n')
    
    # 组合为新的序列组并返回
    return SequenceData([time, diff_r, diff_t, diff_n])

def GetMEEFDthrust(ThrustPath_raw):
    """读取MEEFD方法的得到的加速度,并将时间对齐和数据归一化"""
    AccData = SequenceData.from_csv(ThrustPath_raw,"Acc")

    # 时间对齐
    time = AccData.get("Time")
    coeff_time0= time.Data[0]
    Time_align = Sequence([time.Data[i] - coeff_time0 for i in range(time.N)],"Time")
    # 时间归一化
    coeff_time1,Time_u = Time_align.unify()
    # 参数归一化
    coeff_r,ur_u = AccData.get("u_r").unify()
    coeff_t,ut_u = AccData.get("u_t").unify()
    coeff_n,un_u = AccData.get("u_n").unify()
    AccDataTrain = SequenceData([Time_u,ur_u,ut_u,un_u])

    coeffs = coeff_time0,coeff_time1,coeff_r,coeff_t,coeff_n

    return coeffs,AccDataTrain
