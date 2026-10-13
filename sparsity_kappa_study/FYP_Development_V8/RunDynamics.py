from Dynamics import *
import time

def __dynamics_init():
    # 固定的卫星参数
    # 注，Tmax最大推力为170mN
    sat_param = Sat_StaticParam(Cd=2.5,Area=120,c=2500*9.81,m0=720,m=800,Tmax=0.17)
    m = sat_param.m  #卫星的初始质量

    # 初始化卫星信息
    R0,V0 = InitOrbit(alt=530, inc=43)
    
    # 保存初始化数据
    Save_JSON(sat_param, "Output/SatParam.json")

    DynamicsInit = R0, V0, sat_param, m
    return DynamicsInit

Clear_Folder("Output") # 清理文件夹
DynamicsInit = __dynamics_init() # 动力学参数初始化
print("===================================================")
print("【主程序】完成动力学初始化")
print("===================================================")
start_time = time.time()    

R0, V0, sat_param, m = DynamicsInit

# 场景1：轨道维持
Sat = SatelliteStationKeeping(Sat_Param=sat_param, R0=R0, V0=V0, m=m, BoxSize=[5,40,5])
time_param = Time_Param(hour=50,dt=10)

# # 场景2：紧急避障
# Sat = SatelliteCollisionAvoidance(Sat_Param=sat_param, R0=R0, V0=V0, m=m, TargetPos=[0,-20,0],TOF = 4)
# time_param = Time_Param(hour=6,dt=10)

Save_JSON(time_param, "Output/TimeParam.json")
Sat.Propagate_Orbit(time_param)
Sat.Save_CSV("Output", precision=8)

print("===================================================")
print("【主程序】完成动力学推演")
print("===================================================")
end_time = time.time()
print(f"总耗时: {end_time - start_time:.2f}秒")
