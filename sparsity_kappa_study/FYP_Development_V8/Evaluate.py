from SINDy import *

def ErrorIntegral(U1:SequenceData,U2:SequenceData):
    """计算误差积分"""
    time = U1.get('Time').Data
    dt = time[1]-time[0]
    
    r = dt*np.sum(np.abs(np.array(U1.get('u_r').Data)-np.array(U2.get('u_r').Data)))*1000
    t = dt*np.sum(np.abs(np.array(U1.get('u_t').Data)-np.array(U2.get('u_t').Data)))*1000
    n = dt*np.sum(np.abs(np.array(U1.get('u_n').Data)-np.array(U2.get('u_n').Data)))*1000
    total = r+t+n

    return r,t,n,total

def Visualize_ErrorIntegral(r1, t1, n1, total1,
                            r2, t2, n2, total2,
                            r3, t3, n3, total3,
                            r4, t4, n4, total4):
    """可视化误差积分结果"""
    methods = ['SINDy', 'MEEFD', 'SINDyTIME', 'SINDyUMEE']
    
    # 只使用R, T, N方向的误差，不使用total
    error_data = np.array([
        [r1, t1, n1],
        [r2, t2, n2],
        [r3, t3, n3],
        [r4, t4, n4]
    ])
    
    direction_colors = ['r', 'g', 'b']
    direction_names = ['R (m/s)', 'T (m/s)', 'N (m/s)']
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(6, 5))
    
    x = np.arange(len(methods))
    width = 0.6
    
    # 堆叠柱状图的底部位置
    bottom = np.zeros(len(methods))
    
    # 绘制堆叠柱状图
    for direction_idx in range(3):
        direction_errors = error_data[:, direction_idx]
        ax.bar(x, direction_errors, width, 
               bottom=bottom,
               label=direction_names[direction_idx],
               color=direction_colors[direction_idx],
               edgecolor='black', linewidth=0.8,
               alpha=0.7)

        bottom += direction_errors
    
    # 添加总误差标签
    total_errors = error_data.sum(axis=1)
    for method_idx, total_error in enumerate(total_errors):
        ax.text(method_idx, bottom[method_idx] + 0.02, 
               f'{total_error:.2f}', 
               ha='center', va='bottom')
    
    # 设置图表属性
    ax.set_xlabel('Methods')
    ax.set_ylabel('Error Integral (m/s)')
    ax.set_xticks(x)
    ax.set_title('Error Integral')
    ax.set_xticklabels(methods, rotation=0, fontsize=11)
    
    # 自动调整y轴范围，为顶部标签留出空间
    ax.set_ylim(0, max(total_errors) * 1.15)
    
    # 添加图例
    ax.legend(loc='upper right')
    
    # 添加网格线
    ax.grid(True, axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    plt.savefig("Figures/ErrorIntegral.png", dpi=300)
    
    return

def Visualize_EndPosError(Pos1, Pos2, Pos3, Pos4, TruePos):
    """可视化终端位置误差"""
    methods = ['SINDy', 'MEEFD', 'SINDyTIME', 'SINDyUMEE']
    errors = [np.linalg.norm(Pos1 - TruePos),
              np.linalg.norm(Pos2 - TruePos),
              np.linalg.norm(Pos3 - TruePos),
              np.linalg.norm(Pos4 - TruePos)]

    fig,ax = plt.subplots(1, 1, figsize=(6, 5))

    x_pos = np.arange(len(methods))
    ax.bar(x_pos, errors, color=['r', 'g', 'b', 'c'], 
            edgecolor='black', linewidth=1.2, alpha=0.7)
    
    for i, (error, method) in enumerate(zip(errors, methods)):
        ax.text(i, error + 0.05, f'{error:.3f}', 
                ha='center', va='bottom')
    
    ax.set_xlabel('Methods')
    ax.set_ylabel('End Position Error (km)')
    ax.set_title('End Position Error')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(methods, rotation=0)
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    
    plt.savefig('Figures/EndPosError.png', dpi=300)

def Visualize_Comparison(Pos1, Pos2, Pos3, Pos4, TruePos,
                         r1, t1, n1, total1,
                         r2, t2, n2, total2,
                         r3, t3, n3, total3,
                         r4, t4, n4, total4):
    """可视化终端位置误差和误差积分结果"""
    
    # 创建包含两个子图的图形
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    
    # ------------------- 左图：终端位置误差 -------------------
    methods = ['MEEFD', 'SINDy', 'SINDyTIME', 'SINDyUMEE']
    errors = [np.linalg.norm(Pos2 - TruePos),
              np.linalg.norm(Pos1 - TruePos),
              np.linalg.norm(Pos3 - TruePos),
              np.linalg.norm(Pos4 - TruePos)]
    
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
        [r2, t2, n2],
        [r1, t1, n1],
        [r3, t3, n3],
        [r4, t4, n4]
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
        [r2, t2],
        [r1, t1],
        [r3, t3],
        [r4, t4]
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
    
    plt.savefig('Figures/Errors.png', dpi=300)
    
    return

def ComputeErrorIntegral():
    "计算四种方法对照原始推力的误差积分"

    # 提取五组加速度数据
    Dir0 = 'Output/Data_real.csv'
    Dir1 = 'Output/SINDyOri/U_SINDyOri.csv'
    Dir2 = 'Output/MEEFD/U_MEEFD.csv'
    Dir3 = 'Output/SINDyTIME/U_SINDyTIME.csv'
    Dir4 = 'Output/SINDyUMEE/U_SINDyUMEE.csv'

    Cols = ['Time','u_r','u_t','u_n']
    AccTrue = SequenceData.from_csv(Dir0, Col=Cols)
    Acc1 = SequenceData.from_csv(Dir1, Col=Cols)
    Acc2 = SequenceData.from_csv(Dir2, Col=Cols)
    Acc3 = SequenceData.from_csv(Dir3, Col=Cols)
    Acc4 = SequenceData.from_csv(Dir4, Col=Cols)

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
    r4,t4,n4,total4 = ErrorIntegral(AccTrue,Acc4)

    # print("四种方法的误差积分结果如下（单位：m/s）:")
    # print(f"SINDyOri: r={r1:.4g}, t={t1:.4g}, n={n1:.4g}, total={total1:.4g}")
    # print(f"MEEFD: r={r2:.4g}, t={t2:.4g}, n={n2:.4g}, total={total2:.4g}")
    # print(f"SINDyTIME: r={r3:.4g}, t={t3:.4g}, n={n3:.4g}, total={total3:.4g}")
    # print(f"SINDyUMEE: r={r4:.4g}, t={t4:.4g}, n={n4:.4g}, total={total4:.4g}")

    # Visualize_ErrorIntegral(r1,t1,n1,total1,
    #                         r2,t2,n2,total2,
    #                         r3,t3,n3,total3,
    #                         r4,t4,n4,total4)

    return r1,t1,n1,total1,r2,t2,n2,total2,r3,t3,n3,total3,r4,t4,n4,total4

def ComputeEndPosError():
    """计算四种方法的终端位置误差"""
    # 提取实际末端位置
    Dir0 = 'Output/Data_LVLH.csv'
    _,Rx,Ry,Flag = SequenceData.from_csv(Dir0,Col=["Time","R_x","R_y","Flag"]).to_tuple()
    TrueX = Rx[Flag=="manoeuvre"][-1]
    TrueY = Ry[Flag=="manoeuvre"][-1]
    TruePos = np.array([TrueX,TrueY])

    Dir1 = 'Output/SINDyOri/Data_reLVLH.csv'
    Dir2 = 'Output/MEEFD/Data_reLVLH.csv'
    Dir3 = 'Output/SINDyTIME/Data_reLVLH.csv'
    Dir4 = 'Output/SINDyUMEE/Data_reLVLH.csv'
    
    def _get_end_pos(Dir):
        """获取末端位置"""  
        Rx,Ry = SequenceData.from_csv(Dir,Col=["R_x","R_y"]).to_tuple()
        Pos = np.array([Rx[-1],Ry[-1]])
        return Pos

    Pos1 = _get_end_pos(Dir1)
    Pos2 = _get_end_pos(Dir2)
    Pos3 = _get_end_pos(Dir3)
    Pos4 = _get_end_pos(Dir4)

    # print("四种方法的终端位置误差如下（单位：km）:")
    # print(f"SINDyOri: {np.linalg.norm(Pos1-TruePos):.4g}")
    # print(f"MEEFD: {np.linalg.norm(Pos2-TruePos):.4g}")
    # print(f"SINDyTIME: {np.linalg.norm(Pos3-TruePos):.4g}")
    # print(f"SINDyUMEE: {np.linalg.norm(Pos4-TruePos):.4g}")

    # Visualize_EndPosError(Pos1,Pos2,Pos3,Pos4,TruePos)

    return Pos1, Pos2, Pos3, Pos4, TruePos

if __name__=="__main__":

    r1,t1,n1,total1,r2,t2,n2,total2,r3,t3,n3,total3,r4,t4,n4,total4 = ComputeErrorIntegral()
    Pos1, Pos2, Pos3, Pos4, TruePos = ComputeEndPosError()

    Visualize_Comparison(Pos1, Pos2, Pos3, Pos4, TruePos,
                         r1, t1, n1, total1,
                         r2, t2, n2, total2,
                         r3, t3, n3, total3,
                         r4, t4, n4, total4)