# 添加同级目录到路径
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)
from Plotter import *

# LVLH场景2绘图
init_dyn_plot()
init_SINDy_plot() 
LVLHPath = 'ReportFigs/DynamicsPlots/Output/Data_LVLH2.csv'
RealPath = 'ReportFigs/DynamicsPlots/Output/Data_real2.csv'

PlotTrajectory_ECI(RealPath,save_path="ReportFigs/DynamicsPlots/Figures/ECI2.png")
PlotTrajectory_LVLH_Scene2(LVLHPath, Target=[0,-20,0],save_path= "ReportFigs/DynamicsPlots/Figures/LVLH2.png", IsAspect=True, show=False)
PlotAllAccs([RealPath],save_path="ReportFigs/DynamicsPlots/Figures/Thrust2.png")