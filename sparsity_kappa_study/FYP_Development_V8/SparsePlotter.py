# 添加同级目录到路径
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from SINDy import *
from Plotter import *
from Evaluate import *

def Sparse_ComputeErrorIntegral():
    "计算四种方法对照原始推力的误差积分"

    # 提取加速度数据
    Dir0 = 'ReportFigs/SparsityPlots/Output-Full/Data_real.csv'

    # 静态单组测试路径
    Dir1 = 'ReportFigs/SparsityPlots/Output-Full/SINDyTIME/U_SINDyTIME.csv'
    Dir2 = 'ReportFigs/SparsityPlots/Output-Normal/SINDyTIME/U_SINDyTIME.csv'
    Dir3 = 'ReportFigs/SparsityPlots/Output-Sparse/SINDyTIME/U_SINDyTIME.csv'

    Dir1 = 'ReportFigs/SparsityPlots/Output-Full-Noise/SINDyTIME/U_SINDyTIME.csv'
    Dir2 = 'ReportFigs/SparsityPlots/Output-Normal-Noise/SINDyTIME/U_SINDyTIME.csv'
    Dir3 = 'ReportFigs/SparsityPlots/Output-Sparse-Noise/SINDyTIME/U_SINDyTIME.csv'

    # 批量测试路径
    Dir1 = 'Output/SINDyTIME-NonSparse/U_SINDyTIME.csv'
    Dir2 = 'Output/SINDyTIME-Normal/U_SINDyTIME.csv'
    Dir3 = 'Output/SINDyTIME-Sparse/U_SINDyTIME.csv'

    Cols = ['Time','u_r','u_t','u_n']
    AccTrue = SequenceData.from_csv(Dir0, Col=Cols)
    Acc1 = SequenceData.from_csv(Dir1, Col=Cols)
    Acc2 = SequenceData.from_csv(Dir2, Col=Cols)
    Acc3 = SequenceData.from_csv(Dir3, Col=Cols)

    # 筛选对应的区间
    def _find(time_data, target_time):
        """找到正确的区间"""
        times = np.array(time_data.Data)
        diffs = np.abs(times - target_time)
        min_idx = np.argmin(diffs)

        return min_idx
    
    startime = Acc1.get("Time").Data[0]
    endtime = Acc1.get("Time").Data[-1]
    startidx = _find(AccTrue.get("Time"),startime)
    endidx = _find(AccTrue.get("Time"),endtime)+1
    interval = startidx,endidx
    AccTrue = AccTrue.filter(interval)
    
    # 计算拟合的加速度的误差积分
    r1,t1,n1,total1 = ErrorIntegral(AccTrue,Acc1)
    r2,t2,n2,total2 = ErrorIntegral(AccTrue,Acc2)
    r3,t3,n3,total3 = ErrorIntegral(AccTrue,Acc3)

    return r1,t1,n1,total1,r2,t2,n2,total2,r3,t3,n3,total3

def Sparse_ComputeEndPosError():
    """计算四种方法的终端位置误差"""
    # 提取实际末端位置
    Dir0 = 'ReportFigs/SparsityPlots/Data_LVLH.csv'
    _,Rx,Ry,Flag = SequenceData.from_csv(Dir0,Col=["Time","R_x","R_y","Flag"]).to_tuple()
    TrueX = Rx[Flag=="manoeuvre"][-1]
    TrueY = Ry[Flag=="manoeuvre"][-1]
    TruePos = np.array([TrueX,TrueY])

    # 静态单组测试路径
    Dir1 = 'ReportFigs/SparsityPlots/Output-Full/SINDyTIME/Data_reLVLH.csv'
    Dir2 = 'ReportFigs/SparsityPlots/Output-Normal/SINDyTIME/Data_reLVLH.csv'
    Dir3 = 'ReportFigs/SparsityPlots/Output-Sparse/SINDyTIME/Data_reLVLH.csv'

    Dir1 = 'ReportFigs/SparsityPlots/Output-Full-Noise/SINDyTIME/Data_reLVLH.csv'
    Dir2 = 'ReportFigs/SparsityPlots/Output-Normal-Noise/SINDyTIME/Data_reLVLH.csv'
    Dir3 = 'ReportFigs/SparsityPlots/Output-Sparse-Noise/SINDyTIME/Data_reLVLH.csv'

    # 批量测试路径
    Dir1 = 'Output/SINDyTIME-NonSparse/Data_reLVLH.csv'
    Dir2 = 'Output/SINDyTIME-Normal/Data_reLVLH.csv'
    Dir3 = 'Output/SINDyTIME-Sparse/Data_reLVLH.csv'
    
    def _get_end_pos(Dir):
        """获取末端位置"""  
        Rx,Ry = SequenceData.from_csv(Dir,Col=["R_x","R_y"]).to_tuple()
        Pos = np.array([Rx[-1],Ry[-1]])
        return Pos

    Pos1 = _get_end_pos(Dir1)
    Pos2 = _get_end_pos(Dir2)
    Pos3 = _get_end_pos(Dir3)

    return Pos1, Pos2, Pos3, TruePos

def Sparse_PlotHeatMap():
    # 绘制SINDy热力图
    init_SINDy_plot()
    Plot_Xi_Heatmap(filepath = "ReportFigs/SparsityPlots/Output-Full/SINDyTIME/Xi.csv",figsize=(8,10), Anno=True, save_path="ReportFigs/SparsityPlots/HeatMap-Full.png", 
                        show=False, x_labels= [r'$u_r$',r'$u_t$',r'$u_n$'])

    Plot_Xi_Heatmap(filepath = "ReportFigs/SparsityPlots/Output-Normal/SINDyTIME/Xi.csv",figsize=(8,10), Anno=True, save_path="ReportFigs/SparsityPlots/HeatMap-Normal.png", 
                        show=False, x_labels= [r'$u_r$',r'$u_t$',r'$u_n$'])

    Plot_Xi_Heatmap(filepath = "ReportFigs/SparsityPlots/Output-Sparse/SINDyTIME/Xi.csv",figsize=(8,10), Anno=True, save_path="ReportFigs/SparsityPlots/HeatMap-Sparse.png", 
                        show=False, x_labels= [r'$u_r$',r'$u_t$',r'$u_n$'])
    
    Plot_Xi_Heatmap(filepath = "ReportFigs/SparsityPlots/Output-Full-Noise/SINDyTIME/Xi.csv",figsize=(8,10), Anno=True, save_path="ReportFigs/SparsityPlots/HeatMap-Full-Noise.png", 
                        show=False, x_labels= [r'$u_r$',r'$u_t$',r'$u_n$'])

    Plot_Xi_Heatmap(filepath = "ReportFigs/SparsityPlots/Output-Normal-Noise/SINDyTIME/Xi.csv",figsize=(8,10), Anno=True, save_path="ReportFigs/SparsityPlots/HeatMap-Normal-Noise.png", 
                        show=False, x_labels= [r'$u_r$',r'$u_t$',r'$u_n$'])

    Plot_Xi_Heatmap(filepath = "ReportFigs/SparsityPlots/Output-Sparse-Noise/SINDyTIME/Xi.csv",figsize=(8,10), Anno=True, save_path="ReportFigs/SparsityPlots/HeatMap-Sparse-Noise.png", 
                        show=False, x_labels= [r'$u_r$',r'$u_t$',r'$u_n$'])
    return

def Sparse_Visualize_Comparison(Pos1, Pos2, Pos3, TruePos,
                         r1, t1, n1, total1,
                         r2, t2, n2, total2,
                         r3, t3, n3, total3):
    """可视化终端位置误差和误差积分结果"""
    
    # 创建包含两个子图的图形
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    
    # ------------------- 左图：终端位置误差 -------------------
    methods = ["Non-Sparse","Sparse","Very-Sparse"]
    errors = [np.linalg.norm(Pos1 - TruePos),
              np.linalg.norm(Pos2 - TruePos),
              np.linalg.norm(Pos3 - TruePos)]
    
    x_pos = np.arange(len(methods))
    ax1.bar(x_pos, errors, 
            color=['lightcoral', 'skyblue', 'palegreen', 'peachpuff'], 
            # color=['red', 'darkblue', 'gold', 'cyan'],
            edgecolor='black', linewidth=1.2, alpha=0.7)
    
    for i, (error, method) in enumerate(zip(errors, methods)):
        ax1.text(i, error + 0.02, f'{error:.3f}', 
                ha='center', va='bottom')
    
    ax1.set_ylabel('End Position Error (km)')
    ax1.set_title('End Position Error')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(methods, rotation=0)
    ax1.grid(True, axis='y', alpha=0.3)
    ax1.set_ylim(0, max(errors) * 1.15)
    
    # ------------------- 中图：误差积分 -------------------
    error_data = np.array([
        [r1, t1, n1],
        [r2, t2, n2],
        [r3, t3, n3]
    ])
    
    direction_colors = colors = ['lightcoral', 'skyblue', 'palegreen']
    direction_names = ['R (m/s)', 'T (m/s)', 'N (m/s)']
    
    x = np.arange(len(methods))
    width = 0.6
    bottom = np.zeros(len(methods))
    for direction_idx in range(3):
        direction_errors = error_data[:, direction_idx]
        ax2.bar(x, direction_errors, width, 
               bottom=bottom,
               label=direction_names[direction_idx],
               color=direction_colors[direction_idx],
               edgecolor='black', linewidth=0.8,
               alpha=0.7)

        bottom += direction_errors
    
    # 添加总误差标签
    total_errors = error_data.sum(axis=1)
    for method_idx, total_error in enumerate(total_errors):
        ax2.text(method_idx, bottom[method_idx] + 0.02, 
               f'{total_error:.2f}', 
               ha='center', va='bottom')
    
    # 设置图表属性
    ax2.set_ylabel('Error Integral (m/s)')
    ax2.set_title('Error Integral')
    ax2.set_xticks(x)
    ax2.set_xticklabels(methods, rotation=0, fontsize=11)
    ax2.set_ylim(0, max(total_errors) * 1.15)
    ax2.legend(loc='upper right')
    ax2.grid(True, axis='y', alpha=0.3, linestyle='--')

    # ------------------- 右图：误差积分 -------------------
    error_data = np.array([
        [r1, t1],
        [r2, t2],
        [r3, t3],
    ])

    direction_colors = ['lightcoral', 'skyblue']
    direction_names = ['R (m/s)', 'T (m/s)']
    x = np.arange(len(methods))
    width = 0.6
    bottom = np.zeros(len(methods))
    for direction_idx in range(2):
        direction_errors = error_data[:, direction_idx]
        ax3.bar(x, direction_errors, width, 
               bottom=bottom,
               label=direction_names[direction_idx],
               color=direction_colors[direction_idx],
               edgecolor='black', linewidth=0.8,
               alpha=0.7)

        bottom += direction_errors
    
    # 添加总误差标签
    total_errors = error_data.sum(axis=1)
    for method_idx, total_error in enumerate(total_errors):
        ax3.text(method_idx, bottom[method_idx] + 0.02, 
               f'{total_error:.2f}', 
               ha='center', va='bottom')
    
    # 设置图表属性
    ax3.set_ylabel('Error Integral (m/s)')
    ax3.set_title('Error Integral (Excluding N)')
    ax3.set_xticks(x)
    ax3.set_xticklabels(methods, rotation=0, fontsize=11)
    ax3.set_ylim(0, max(total_errors) * 1.15)
    ax3.legend(loc='upper right')
    ax3.grid(True, axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    plt.savefig('ReportFigs/SparsityPlots/Errors.png', dpi=300)
    
    return

if __name__=="__main__":
    
    # # 绘制热力图
    # Sparse_PlotHeatMap()

    # 绘制推力对比图像
    init_SINDy_plot()
    # Dir0 = 'ReportFigs/SparsityPlots/Output-Normal/Data_real.csv'
    # Dir1 = 'ReportFigs/SparsityPlots/Output-Full/SINDyTIME/U_SINDyTIME.csv'
    # Dir2 = 'ReportFigs/SparsityPlots/Output-Normal/SINDyTIME/U_SINDyTIME.csv'
    # Dir3 = 'ReportFigs/SparsityPlots/Output-Sparse/SINDyTIME/U_SINDyTIME.csv'

    Dir0 = 'ReportFigs/SparsityPlots/Output-Normal-Noise/Data_real.csv'
    Dir1 = 'ReportFigs/SparsityPlots/Output-Full-Noise/SINDyTIME/U_SINDyTIME.csv'
    Dir2 = 'ReportFigs/SparsityPlots/Output-Normal-Noise/SINDyTIME/U_SINDyTIME.csv'
    Dir3 = 'ReportFigs/SparsityPlots/Output-Sparse-Noise/SINDyTIME/U_SINDyTIME.csv'
    save_path1 = "ReportFigs/SparsityPlots/SparseAcc.png"
    PlotAllAccs([Dir0,Dir1,Dir2,Dir3],save_path1)

    # 绘制重新推演的推力图像
    init_dyn_plot()
    Dir = 'ReportFigs/SparsityPlots/'
    PathList1 = ['Output-Full/SINDyTIME/Data_reLVLH.csv','Output-Normal/SINDyTIME/Data_reLVLH.csv','Output-Sparse/SINDyTIME/Data_reLVLH.csv']
    PathList1 = ['Output-Full-Noise/SINDyTIME/Data_reLVLH.csv','Output-Normal-Noise/SINDyTIME/Data_reLVLH.csv','Output-Sparse-Noise/SINDyTIME/Data_reLVLH.csv']
    PlotTrajectory_LVLH_Reprop_XY(Dir,PathList1,save_path = 'ReportFigs/SparsityPlots/SparseReprop.png',IsAspect=False,show=False)

    # # 计算评价误差
    # Pos1, Pos2, Pos3, TruePos = Sparse_ComputeEndPosError()
    # r1,t1,n1,total1,r2,t2,n2,total2,r3,t3,n3,total3 = Sparse_ComputeErrorIntegral()

    # Sparse_Visualize_Comparison(Pos1, Pos2, Pos3, TruePos,
    #                      r1, t1, n1, total1,
    #                      r2, t2, n2, total2,
    #                      r3, t3, n3, total3)