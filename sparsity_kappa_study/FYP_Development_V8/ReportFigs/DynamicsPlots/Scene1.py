# 添加同级目录到路径
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)
from Plotter import *

# LVLH场景1绘图
init_dyn_plot()
init_SINDy_plot()
RealPath = 'ReportFigs/DynamicsPlots/Output/Data_real1.csv'
LVLHPath = 'ReportFigs/DynamicsPlots/Output/Data_LVLH1.csv'

PlotTrajectory_ECI(RealPath,save_path="ReportFigs/DynamicsPlots/Figures/ECI1.pdf")
PlotTrajectory_LVLH_Scene1(LVLHPath,BoxSize=[5,40,5],shift=0,IsAspect=True,show=True,save_path="ReportFigs/DynamicsPlots/Figures/LVLH1.pdf")
PlotAllAccs([RealPath],save_path="ReportFigs/DynamicsPlots/Figures/Thrust1.pdf")
