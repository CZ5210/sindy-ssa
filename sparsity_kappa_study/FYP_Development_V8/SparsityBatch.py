# 添加同级目录到路径
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from SINDy import *
from Dynamics import *
from Plotter import *
from PlotResult import *
from SparsePlotter import *
from RunSINDy import RepropTest
import gc

def __param_init():
    """主函数运行的参数"""

    # 一些运行路径
    global FilePath,ObservePath,EvalPath
    global ThrustPath_raw,ThrustPath0,ThrustPath1,ThrustPath2,ThrustPath3,ThrustPath4
    global OutDir0,OutDir1,OutDir2,OutDir3,OutDir4

    FilePath = './Output/Data_real.csv'
    ObservePath = "./Output/Observe/"

    ThrustPath_raw = './Output/MEEFD/U_MEEFD_raw.csv'
    ThrustPath1 = './Output/MEEFD/U_MEEFD.csv'
    ThrustPath2 = './Output/SINDyTIME-Normal/U_SINDyTIME.csv'
    ThrustPath3 = './Output/SINDyTIME-NonSparse/U_SINDyTIME.csv'
    ThrustPath4 = './Output/SINDyTIME-Sparse/U_SINDyTIME.csv'

    OutDir1 = './Output/MEEFD'
    OutDir2 = './Output/SINDyTIME-Normal'
    OutDir3 = './Output/SINDyTIME-NonSparse'
    OutDir4 = './Output/SINDyTIME-Sparse'

    EvalPath = './Output/SparseBatch_Evaluation.csv'

    # 设置参数
    window = 51
    SamplingRate = 6
    BatchSize = 100

    # 从星历分析的噪音信息
    NoiseStd = np.zeros((6))
    NoiseStd= [8.1045e+00,7.3764e-04,7.4911e-04,2.3713e-04,2.3602e-04,3.9802e-01]
    NoiseStd = [seq/10 for seq in NoiseStd]

    return BatchSize,window,SamplingRate,NoiseStd

def ObserveBatch(SamplingRate,NoiseStd):
    # 生成BatchSize组模拟卫星观测数据
    Clear_Folder(ObservePath)
    for i in range(BatchSize):
        ObserveFile = ObservePath+str(i+1)+".csv"
        ObserveSat(SamplingRate,NoiseStd,FilePath,ObserveFile)
    return

def RunBatch(BatchSize):
    """使用不同方法进行批量测试"""
    init_SINDy_plot()
    for i in range(BatchSize):
        ObserveFile = ObservePath+str(i+1)+".csv"
        print("========================================")
        print(f"开始第{i+1}组测试")
        print("========================================")

        # MEE_FD方法
        MEE_FD(window,SamplingRate,ObserveFile,ThrustPath1,ThrustPath_raw)
        RepropTest(FilePath,ThrustPath1,OutDir1)
        gc.collect()
        print("--------------完成MEEFD计算--------------")

        # SINDyTime方法
        SINDy_TIME(SINDyTIME_Header1,SamplingRate,ThrustPath_raw,OutDir2)
        RepropTest(FilePath,ThrustPath2,OutDir2)
        # Plot_Xi_Heatmap(filepath = "Output/SINDyTIME/Xi.csv",figsize=(12,10), Anno=True, save_path="Figures/HeatMapTIME.png", 
        #         show=False, x_labels= [r'$u_r$',r'$u_t$',r'$u_n$'])
        gc.collect()
        print("--------------完成SINDyTIME-Normal计算--------------")
        
        # SINDyTime方法
        SINDy_TIME(SINDyTIME_Header2,SamplingRate,ThrustPath_raw,OutDir3)
        RepropTest(FilePath,ThrustPath3,OutDir3)
        gc.collect()
        print("--------------完成SINDyTIME-Non-Sparse计算--------------")

        # SINDyTime方法
        SINDy_TIME(SINDyTIME_Header3,SamplingRate,ThrustPath_raw,OutDir4)
        RepropTest(FilePath,ThrustPath4,OutDir4)
        gc.collect()
        print("--------------完成SINDyTIME-Sparse计算--------------")

        # 完成三组测试后，评价指标分析
        r1,t1,n1,total1,r2,t2,n2,total2,r3,t3,n3,total3 = Sparse_ComputeErrorIntegral()
        Pos1, Pos2, Pos3, TruePos = Sparse_ComputeEndPosError()
        ErrorInte = [r1,t1,n1,total1,
                     r2,t2,n2,total2,
                     r3,t3,n3,total3]
        PosError = [np.linalg.norm(Pos1 - TruePos),
                    np.linalg.norm(Pos2 - TruePos),
                    np.linalg.norm(Pos3 - TruePos)]

        # 存储评价指标，每组测试结果追加写入
        
        if i == 0:
            with open (EvalPath,'w') as f:
                f.write("r1,t1,n1,total1,"+ "r2,t2,n2,total2," + 
                        "r3,t3,n3,total3," +
                        "enderr1, enderr2, enderr3 \n")
                f.write(','.join([f"{err:.2f}" for err in ErrorInte])+','+
                        ','.join([f"{err:.2f}" for err in PosError])+'\n')
        else:
            with open (EvalPath,'a') as f:
                f.write(','.join([f"{err:.2f}" for err in ErrorInte])+','+
                        ','.join([f"{err:.2f}" for err in PosError])+'\n')
    print("========================================")    
    print("==============完成批量测试===============")
    print("========================================")   

    return

def SparseBoxPlotVisualize(filename):
    """读取数据并绘制带有errorbar的boxplot"""
    from mpl_toolkits.axes_grid1.inset_locator import mark_inset
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    # 读取数据
    data = pd.read_csv(filename, header=None,skiprows=1)
    
    # 定义列名，注意这里为了绘制调整顺序调整了
    columns = [
        'r1', 't1', 'n1', 'total1',
        'r2', 't2', 'n2', 'total2',
        'r3', 't3', 'n3', 'total3',
        'enderr1', 'enderr2', 'enderr3'
    ]
    data.columns = columns
    
    # 创建包含三个子图的图形
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 4))
    
    # ------------------- 左图：最终位置误差的箱线图 -------------------
    methods = ["Non-Sparse","Sparse","Very-Sparse"]
    
    # 提取最终位置误差数据
    end_errors_data = []
    for i in range(3):
        end_errors = data[f'enderr{i+1}'].values
        end_errors_data.append(end_errors)
    
    # 绘制箱线图
    boxplot1 = ax1.boxplot(end_errors_data, tick_labels=methods, patch_artist=True, 
                          medianprops=dict(color='black', linewidth=1),
                          flierprops=dict(marker='', markersize=0, markeredgecolor='none'))
    
    # 设置箱线图颜色
    colors = ['lightcoral', 'skyblue', 'palegreen', 'peachpuff']
    # colors = ['indianred', 'skyblue', 'peachpuff', 'lightcyan']
    # colors=['red', 'darkblue', 'gold', 'cyan']
    for patch, color in zip(boxplot1['boxes'], colors):
        patch.set_facecolor(color)
    [median.set_alpha(0.5) for median in boxplot1['medians']]
    for box in boxplot1['boxes']:
        box.set_edgecolor('k')
        box.set_alpha(0.7)
    for whisker in boxplot1['whiskers']:
        whisker.set(color='black', alpha=1, linewidth=0.5)
    for cap in boxplot1['caps']:
        cap.set(color='black', alpha=1, linewidth=0.5)
    for flier in boxplot1['fliers']:
        flier.set(marker='*', color='black', alpha=1, markersize=4)
    
    # ax1.set_yscale('log')
    ax1.tick_params(axis='x', labelsize=10)
    ax1.set_ylabel('End Position Error (km)')
    ax1.set_title('End Position Errors')
    ax1.grid(True, axis='y', alpha=0.3, linestyle='--')
    
    # ------------------- 中图：总误差积分的箱线图 -------------------
    total_errors_data = []
    for i in range(3):
        total_errors = data[f'total{i+1}'].values
        total_errors_data.append(total_errors)
    
    # 绘制箱线图
    boxplot2 = ax2.boxplot(total_errors_data, tick_labels=methods, patch_artist=True,
                          medianprops=dict(color='black', linewidth=1))
    
    
    # 设置箱线图颜色
    for patch, color in zip(boxplot2['boxes'], colors):
        patch.set_facecolor(color)
    [median.set_alpha(0.5) for median in boxplot2['medians']]
    for box in boxplot2['boxes']:
        box.set_edgecolor('k')
        box.set_alpha(0.7)
    for whisker in boxplot2['whiskers']:
        whisker.set(color='black', alpha=1, linewidth=0.5)
    for cap in boxplot2['caps']:
        cap.set(color='black', alpha=1, linewidth=0.5)
    for flier in boxplot2['fliers']:
        flier.set(marker='*', color='black', alpha=1, markersize=4)
    
    # ax2.set_yscale('log')
    ax2.tick_params(axis='x', labelsize=10)
    ax2.set_ylabel('Error Integral (m/s)')
    ax2.set_title('Total Error Integral')
    ax2.grid(True, axis='y', alpha=0.3, linestyle='--')
    
    # ------------------- 右图：排除N方向的误差积分的箱线图 -------------------
    # 计算每个方法的r+t方向的误差积分
    r_t_errors_data = []
    for i in range(3):
        r_errors = data[f'r{i+1}'].values
        t_errors = data[f't{i+1}'].values
        r_t_errors = r_errors + t_errors  # r + t 方向
        r_t_errors_data.append(r_t_errors)
    
    # 绘制箱线图
    boxplot3 = ax3.boxplot(r_t_errors_data, tick_labels=methods, patch_artist=True,
                          medianprops=dict(color='black', linewidth=1))
    
    # 设置箱线图颜色
    for patch, color in zip(boxplot3['boxes'], colors):
        patch.set_facecolor(color)
    [median.set_alpha(0.5) for median in boxplot3['medians']]
    for box in boxplot3['boxes']:
        box.set_edgecolor('k')
        box.set_alpha(0.7)
    for whisker in boxplot3['whiskers']:
        whisker.set(color='black', alpha=1, linewidth=0.5)
    for cap in boxplot3['caps']:
        cap.set(color='black', alpha=1, linewidth=0.5)
    for flier in boxplot3['fliers']:
        flier.set(marker='*', color='black', alpha=1, markersize=4)
    
    # ax3.set_yscale('log')
    ax3.tick_params(axis='x', labelsize=10)
    ax3.set_ylabel('Error Integral (m/s)')
    ax3.set_title('Error Integral (Excluding N Direction)')
    ax3.grid(True, axis='y', alpha=0.3, linestyle='--')

    # # 局部放大图
    # axins = ax3.inset_axes((0.3, 0.4, 0.6, 0.5))
    # boxplot3 = axins.boxplot(r_t_errors_data, tick_labels=methods, patch_artist=True,
    #                       medianprops=dict(color='black', linewidth=1))
    # # 设置箱线图颜色
    # for patch, color in zip(boxplot3['boxes'], colors):
    #     patch.set_facecolor(color)
    # [median.set_alpha(0.5) for median in boxplot3['medians']]
    # for box in boxplot3['boxes']:
    #     box.set_edgecolor('k')
    #     box.set_alpha(0.7)
    # for whisker in boxplot3['whiskers']:
    #     whisker.set(color='black', alpha=1, linewidth=0.5)
    # for cap in boxplot3['caps']:
    #     cap.set(color='black', alpha=1, linewidth=0.5)
    # for flier in boxplot3['fliers']:
    #     flier.set(marker='*', color='black', alpha=1, markersize=4)
    # axins.tick_params(axis='x', labelsize=8)
    # axins.set_xlim(1.6, 4.4)
    # axins.set_ylim(0, 60)
    # axins.grid(True, axis='y', alpha=0.3, linestyle='--')
    # mark_inset(ax3, axins, loc1=1, loc2=2, fc="none", ec='k', lw=1)

    # 调整布局并保存
    plt.tight_layout()
    plt.savefig('Figures/Sparse_Boxplot.png', dpi=300)
    plt.show()
    
    return data


if __name__ == "__main__":
    # 初始化参数
    BatchSize,window,SamplingRate,NoiseStd = __param_init()
    
    # 模拟卫星观测数据
    ObserveBatch(SamplingRate,NoiseStd)

    # 批量测试
    RunBatch(BatchSize)

    # 绘图
    SparseBoxPlotVisualize(EvalPath)
    
        



