from .Sequence import *
from .SINDy_Main import *
from Plotter import *

def MEE_FD(window,SamplingRate,ObservePath,ThrustPath,ThrustPath_raw):
    """使用无SINDy的直接方法，测试加速度信息的重建"""

    # 读取观测到的推力数据
    ObserveData = SequenceData.from_csv(ObservePath)

    # 筛选机动段数据
    count,interval = Extract_Manoeuvre(ObservePath)
    ObserveData = ObserveData
    print(f"共检测到 {count} 个机动段")

    # 使用GVE摄动方法计算自然MEEdot
    MEEdot_natural = GVE_MEE(ObserveData)

    # 从观测数据中有限差分得到总体的MEEdot
    MEEdot_FD = FD_MEE(ObserveData)

    # 将两个MEEdot转化到加速度数据
    u_natural = Element2AccData(ObserveData,MEEdot_natural)
    u_all = Element2AccData(ObserveData,MEEdot_FD)

    # 将加速度数据做减法转化到推力数据
    U = MinusAcc(u_all,u_natural).smooth(window=window,idx=0).filter(interval[0])
    U.save(path=ThrustPath_raw)
    U = U.linearinterp(SamplingRate)
    U.save(path=ThrustPath)
    print("[MEE-FD重建] 基于MEEFD有限差分的数据已保存。")

    return