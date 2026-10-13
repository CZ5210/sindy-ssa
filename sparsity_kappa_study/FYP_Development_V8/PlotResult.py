from Plotter import *

def __PlotDynamics1():
    # 绘制场景1
    RealPath = 'Output/Data_real.csv'
    LVLHPath = 'Output/Data_LVLH.csv'

    PlotTrajectory_ECI(RealPath,save_path="Figures/ECI1.png")
    PlotTrajectory_LVLH_Scene1(LVLHPath,BoxSize=[5,40,5],shift=0,IsAspect=True,save_path="Figures/LVLH1.png")
    PlotAllAccs([RealPath],save_path="Figures/Thrust1.png")

def __PlotDynamics2():
    # 绘制场景2
    RealPath = 'Output/Data_real.csv'
    LVLHPath = 'Output/Data_LVLH.csv'

    PlotTrajectory_ECI(RealPath,save_path="Figures/ECI2.png")
    PlotTrajectory_LVLH_Scene2(LVLHPath, Target=[0,-20,0],save_path= "Figures/LVLH2.png", IsAspect=True, show=False)
    PlotAllAccs([RealPath],save_path="Figures/Thrust2.png")

def __PlotReprop():
    # SINDy重建后绘图
    Dir = 'Output/'
    PathList1 = ['MEEFD/Data_reLVLH.csv','SINDyOri/Data_reLVLH.csv','SINDyTIME/Data_reLVLH.csv','SINDyUMEE/Data_reLVLH.csv']
    # PathList2 = ['MEEFD/Data_reLVLH.csv','SINDyTIME/Data_reLVLH.csv','SINDyUMEE/Data_reLVLH.csv']
    PlotTrajectory_LVLH_Reprop_XY(Dir,PathList1,save_path = 'Figures/Reprop1.png',IsAspect=False,show=False)
    # PlotTrajectory_LVLH_Reprop_XY(Dir,PathList2,save_path = 'Figures/Reprop2.png',IsAspect=False,show=False)

def __PlotAcc():
    # 绘制所有的加速度
    Dir0 = 'Output/Data_real.csv'
    Dir1 = 'Output/MEEFD/U_MEEFD.csv'
    Dir2 = 'Output/SINDyOri/U_SINDyOri.csv'
    Dir3 = 'Output/SINDyTIME/U_SINDyTIME.csv'
    Dir4 = 'Output/SINDyUMEE/U_SINDyUMEE.csv'
    save_path1 = "Figures/AccCompare1.png"
    save_path2 = "Figures/AccCompare2.png"
    PlotAllAccs([Dir0,Dir1,Dir2,Dir3,Dir4],save_path1)
    PlotAllAccs([Dir0,Dir1,Dir2,Dir3],save_path2)

if __name__=="__main__":
    # 初始化全局颜色等
    init_dyn_plot()
    init_SINDy_plot()

    # 动力学绘制
    # __PlotDynamics1()
    # __PlotDynamics2()
    
    # 绘制重推演图像
    __PlotReprop()

    # 绘制所有加速度
    __PlotAcc()