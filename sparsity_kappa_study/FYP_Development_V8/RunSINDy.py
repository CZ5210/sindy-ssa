from SINDy import *
from Dynamics import *
from Plotter import *
from PlotResult import *
import gc

def RepropTest(FilePath,ThrustPath,OutDir):
    """基于重建的推力重新推演卫星动力学"""
    # 提取发生机动的区间
    _, Intervals = Extract_Manoeuvre(FilePath)

    # 初始化的卫星参数
    time_param = Time_Param.from_dict(Read_JSON('Output/TimeParam.json'))
    sat_param = Sat_StaticParam(**Read_JSON('Output/SatParam.json'))
    m = sat_param.m  #卫星的初始质量

    R0,V0 = InitOrbit(alt=530, inc=43)
    Sat = BaseSatellite(Sat_Param=sat_param, R0=R0, V0=V0, m=m)

    # 对于机动区间进行推演重建
    Sat.Repropagate_Orbit(Intervals[0],ThrustPath,time_param.dt)
    Sat.Save_Repropagate_CSV(OutDir,precision=8)

def __param_init():
    """主函数运行的参数"""

    # 一些运行路径
    global FilePath,ObservePath
    global ThrustPath_raw,ThrustPath0,ThrustPath1,ThrustPath2,ThrustPath3
    global OutDir0,OutDir1,OutDir2,OutDir3

    FilePath = './Output/Data_real.csv'
    ObservePath = "./Output/ObserveData.csv"

    ThrustPath_raw = './Output/MEEFD/U_MEEFD_raw.csv'
    ThrustPath0 = './Output/SINDyOri/U_SINDyOri.csv'
    ThrustPath1 = './Output/MEEFD/U_MEEFD.csv'
    ThrustPath2 = './Output/SINDyTIME/U_SINDyTIME.csv'
    ThrustPath3 = './Output/SINDyUMEE/U_SINDyUMEE.csv'

    OutDir0 = './Output/SINDyOri'
    OutDir1 = './Output/MEEFD'
    OutDir2 = './Output/SINDyTIME'
    OutDir3 = './Output/SINDyUMEE'

    # 设置参数
    window = 51
    SamplingRate = 12

    # 从星历分析的噪音信息
    NoiseStd = np.zeros((6))
    NoiseStd= [8.1045e+00,7.3764e-04,7.4911e-04,2.3713e-04,2.3602e-04,3.9802e-01]
    NoiseStd = [3.5668e-01,2.5515e-04,2.5553e-04,1.3696e-05,1.3653e-05,7.0751e-04]
    # NoiseStd = [seq/10 for seq in NoiseStd]

    return window,SamplingRate,NoiseStd

if __name__ == "__main__":
    # 初始化参数
    init_SINDy_plot()
    window,SamplingRate,NoiseStd = __param_init()
    # 模拟卫星观测数据
    ObserveSat(SamplingRate,NoiseStd,FilePath,ObservePath)

    # 传统SINDy方法
    SINDy_Ori(SINDy_Header1,ObservePath,window,SamplingRate,OutDir0,ThrustPath0)
    RepropTest(FilePath,ThrustPath0,OutDir0)
    Plot_Xi_Heatmap(filepath = "Output/SINDyOri/Xi.csv",figsize=(12,10), Anno=True, save_path="Figures/HeatMapSINDy.png", 
                    show=False, x_labels = [r'$\dot{p}$', r'$\dot{f}$', r'$\dot{g}$', r'$\dot{h}$', r'$\dot{k}$', r'$\dot{L}$'])
    gc.collect()
    print("--------------完成SINDy计算--------------")

    # MEE_FD方法
    MEE_FD(window,SamplingRate,ObservePath,ThrustPath1,ThrustPath_raw)
    RepropTest(FilePath,ThrustPath1,OutDir1)
    gc.collect()
    print("--------------完成MEEFD计算--------------")

    # SINDyTime方法
    SINDy_TIME(SINDyTIME_Header1,SamplingRate,ThrustPath_raw,OutDir2)
    RepropTest(FilePath,ThrustPath2,OutDir2)
    Plot_Xi_Heatmap(filepath = "Output/SINDyTIME/Xi.csv",figsize=(12,10), Anno=True, save_path="Figures/HeatMapTIME.png", 
                    show=False, x_labels= [r'$u_r$',r'$u_t$',r'$u_n$'])
    gc.collect()
    print("--------------完成SINDyTIME计算--------------")
    
    # SINDyUMEE方法
    SINDy_UMEE(SINDyUMEE_Header1,SamplingRate,ThrustPath_raw,ObservePath,OutDir3)
    RepropTest(FilePath,ThrustPath3,OutDir3)
    Plot_Xi_Heatmap(filepath = "Output/SINDyUMEE/Xi.csv",figsize=(12,10), Anno=True, save_path="Figures/HeatMapUMEE.png",
                    show=False, x_labels= [r'$u_r$',r'$u_t$',r'$u_n$'])
    gc.collect()
    print("--------------完成SINDyUMEE计算--------------")