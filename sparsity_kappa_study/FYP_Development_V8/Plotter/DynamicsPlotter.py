import numpy as np
import matplotlib.pyplot as plt
from SINDy.Sequence import *

def init_dyn_plot():
    """初始化所有颜色"""

    # 地球颜色
    global EarthColor,EarthAlpha
    EarthColor = "#2a68bc"
    EarthAlpha = 0.05

    # 轴颜色
    global AxisColor
    AxisColor = ["r","g","b"]

    # 弹道颜色和初始位置设定
    global TrajColor,TrajAlpha,TrajStyle,TrajWidth
    global PointSize,PointColor,PointMarker
    TrajColor = ["#3498db","#cb4335"]
    TrajAlpha = [1,1]
    TrajStyle = ['--','-']
    TrajWidth = [0.4,0.4]

    PointColor = ["#229954","#d4ac0d","purple"]
    PointMarker = ["o","d",'*']
    PointSize = 100

    # LVLH图的一些调整
    global LVLHTrajWidth,LVLHTrajAlpha
    LVLHTrajWidth = [2,3]
    LVLHTrajAlpha = [0.8,1]

    # 重新推演LVLH的全局信息
    global RepoMethod,RepoTrajColor,RepoTrajStyle,RepoTrajAlpha,RepoLinewidth
    global RepoMarker,RepoMarkerColor,RepoMarkerSize
    global TrueTrajColor,TrueTrajAlpha,TrueTrajStyle,TrueTrajwidth
    global TrueMarkerSize,TrueMarkerColor
    
    # RepoMethod=["MEEFD","SINDy","SINDy-TIME","SINDy-UMEE"]
    # 系数测试方法
    RepoMethod=["Non-Sparse","Sparse","Very-Sparse"]

    RepoTrajColor = ["r","g","b",'c']
    RepoTrajStyle = ['--',':',"-.",'-']
    RepoTrajAlpha = 0.7
    RepoLinewidth = 2

    RepoMarker = ['d','s','p','P']
    RepoMarkerColor = ["r","g","b",'c']
    RepoMarkerSize = 100

    TrueTrajColor = "k"
    TrueTrajAlpha = 0.5
    TrueTrajStyle = '-'
    TrueTrajwidth = 3

    TrueMarkerColor = ["k","purple"]
    TrueMarkerSize = 100

def PlotEarth(ax):
    """绘制地球"""
    Re = 6371
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 30)
    x = Re * np.outer(np.cos(u), np.sin(v))
    y = Re * np.outer(np.sin(u), np.sin(v))
    z = Re * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x, y, z, color=EarthColor, 
                    alpha=EarthAlpha)

    return

def PlotBox(ax, BoxSize, shift):
    """绘制LVLH控制盒 - 改进版：在边上生成更多点"""
    
    x = [-BoxSize[0] / 2, BoxSize[0] / 2]  # x 尺寸
    y = [(-BoxSize[1]) / 2 - shift, (BoxSize[1]) / 2 - shift]  # y 尺寸  
    z = [-BoxSize[2] / 2, BoxSize[2] / 2]  # z 尺寸
    
    edges = [
        # Bottom face
        [(x[0], y[0], z[0]), (x[1], y[0], z[0])],
        [(x[1], y[0], z[0]), (x[1], y[1], z[0])],
        [(x[1], y[1], z[0]), (x[0], y[1], z[0])],
        [(x[0], y[1], z[0]), (x[0], y[0], z[0])],
        # Top face
        [(x[0], y[0], z[1]), (x[1], y[0], z[1])],
        [(x[1], y[0], z[1]), (x[1], y[1], z[1])],
        [(x[1], y[1], z[1]), (x[0], y[1], z[1])],
        [(x[0], y[1], z[1]), (x[0], y[0], z[1])],
        # Vertical edges
        [(x[0], y[0], z[0]), (x[0], y[0], z[1])],
        [(x[1], y[0], z[0]), (x[1], y[0], z[1])],
        [(x[1], y[1], z[0]), (x[1], y[1], z[1])],
        [(x[0], y[1], z[0]), (x[0], y[1], z[1])],
    ]
    
    # 设置每边生成的中间点数
    n_points_per_edge = 100
    
    first_edge = True
    
    for edge in edges:
        p1, p2 = edge[0], edge[1]
        
        # 生成更多的点
        x_points = np.linspace(p1[0], p2[0], n_points_per_edge)
        y_points = np.linspace(p1[1], p2[1], n_points_per_edge)
        z_points = np.linspace(p1[2], p2[2], n_points_per_edge)
        
        # 绘制整条边
        if first_edge:
            ax.plot(
                x_points, y_points, z_points,
                color="black",
                linewidth=1.0,
                alpha=0.5,
                label="Station Keeping Box",
                axlim_clip=True
            )
            first_edge = False
        else:
            ax.plot(
                x_points, y_points, z_points,
                color="black",
                linewidth=1.0,
                alpha=0.5,
                axlim_clip=True
            )

def PlotAxis(ax, length=8000, buffer=500,label = 'R,T,N'):
    """使用plot绘制中心坐标轴 - 更好的zorder控制"""
    
    # 设置坐标轴线的属性
    line_width = 1.5
    
    # X轴（红色）
    ax.plot([0, length], [0, 0], [0, 0], 
            color=AxisColor[0], 
            linewidth=line_width,
            zorder=-5)  # 在底层
    
    # Y轴（绿色）
    ax.plot([0, 0], [0, length], [0, 0], 
            color=AxisColor[1], 
            linewidth=line_width,
            zorder=-5)
    
    # Z轴（蓝色）
    ax.plot([0, 0], [0, 0], [0, length], 
            color=AxisColor[2], 
            linewidth=line_width,
            zorder=-5)
    
    # 坐标轴标签
    if label == 'R,T,N':
        ax.text(length + buffer, 0, 0, 'R', 
                color=AxisColor[0], fontsize=12, zorder=-5)
        ax.text(0, length + buffer, 0, 'T', 
                color=AxisColor[1], fontsize=12, zorder=-5)
        ax.text(0, 0, length + buffer, 'N', 
                color=AxisColor[2], fontsize=12, zorder=-5)
    elif label == 'X,Y,Z':
        ax.text(length + buffer, 0, 0, 'X', 
                color=AxisColor[0], fontsize=12, zorder=-5)
        ax.text(0, length + buffer, 0, 'Y', 
                color=AxisColor[1], fontsize=12, zorder=-5)
        ax.text(0, 0, length + buffer, 'Z', 
            color=AxisColor[2], fontsize=12, zorder=-5)
    
    return

def PlotAxis2D(ax, length=8000, buffer=500, label='R,T'):
    """2D版本的PlotAxis - 绘制RT平面的坐标轴"""
    line_width = 1.5

    # X/T轴 (水平, 绿色)
    ax.plot([0, length], [0, 0],
            color=AxisColor[1],
            linewidth=line_width,
            zorder=-5)

    # Y/R轴 (竖直, 红色)
    ax.plot([0, 0], [0, length],
            color=AxisColor[0],
            linewidth=line_width,
            zorder=-5)

    # 坐标轴标签
    if label == 'R,T':
        ax.text(0, length + buffer, 'R',
                color=AxisColor[0], fontsize=12, zorder=-5)
        ax.text(length + buffer, 0, 'T',
                color=AxisColor[1], fontsize=12, zorder=-5)
    elif label == 'X,Y':
        ax.text(0, length + buffer, 'Y',
                color=AxisColor[0], fontsize=12, zorder=-5)
        ax.text(length + buffer, 0, 'X',
                color=AxisColor[1], fontsize=12, zorder=-5)

    return

def PlotTrajectory_ECI(RealPath,save_path = None,show=False):

    """绘制卫星在地心惯性坐标系下的轨迹"""

    # 读取数据
    Time,Rx,Ry,Rz,Flag = SequenceData.from_csv(RealPath,Col=["Time","R_x","R_y","R_z","Flag"]).to_tuple()

    Rx_f = Rx[Flag!="manoeuvre"]
    Ry_f = Ry[Flag!="manoeuvre"]
    Rz_f = Rz[Flag!="manoeuvre"]

    Rx_mano = Rx[Flag=="manoeuvre"]
    Ry_mano = Ry[Flag=="manoeuvre"]
    Rz_mano = Rz[Flag=="manoeuvre"]

    # 绘图
    plt.rcParams.update({'font.size': 14})
    plt.figure(figsize=(10,10))
    ax = plt.axes(projection='3d')

    PlotEarth(ax)
    PlotAxis(ax,length=9000,buffer=1500,label='X,Y,Z')
    
    ax.plot(Rx_f, Ry_f, Rz_f, label = "Free Flight Trajectory",linewidth=TrajWidth[0],color = TrajColor[0],alpha=LVLHTrajAlpha[0],linestyle = TrajStyle[0],zorder = 1)
    ax.plot(Rx_mano, Ry_mano, Rz_mano, label = "Maneuver Segment",linewidth=TrajWidth[1], color = TrajColor[1],alpha=LVLHTrajAlpha[1],linestyle = TrajStyle[1],zorder = 1)
    ax.scatter3D(Rx[0], Ry[0], Rz[0], color=PointColor[0], s=PointSize, label='Start',zorder=10,marker=PointMarker[0])
    ax.scatter3D(Rx[-1], Ry[-1], Rz[-1], color=PointColor[1], s=PointSize, label='End',zorder=10,marker=PointMarker[1])
    
    ax.set_xlabel('X (km)')
    ax.set_ylabel('Y (km)')
    ax.set_zlabel('Z (km)')

    lim = 8000
    ax.set_xlim3d([-lim,lim])
    ax.set_ylim3d([-lim,lim])
    ax.set_zlim3d([-lim,lim])
    ax.set_box_aspect([1, 1, 1])

    # ax.title.set_text('Satellite Trajectory in ECI Frame')
    ax.grid(False)
    ax.view_init(elev=45, azim=45)

    plt.legend()
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"图片已保存至: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()

    return

def PlotTrajectory_LVLH_Scene1(LVLHPath,BoxSize,shift,IsAspect=True,save_path = None,show=False):
    """绘制卫星在LVLH坐标系下的轨迹"""
    Time,Rx,Ry,Rz,Flag = SequenceData.from_csv(LVLHPath,Col=["Time","R_x","R_y","R_z","Flag"]).to_tuple()

    Rx_mano = Rx[Flag=="manoeuvre"]
    Ry_mano = Ry[Flag=="manoeuvre"]
    Rz_mano = Rz[Flag=="manoeuvre"]

    # 绘图
    plt.rcParams.update({'font.size': 14})

    x_min, x_max = -5, 5
    y_min, y_max = -10, 50
    z_min, z_max = -5, 5

    # --- 3D 图 ---
    plt.figure(figsize=(8,8))
    ax = plt.axes(projection='3d')

    ax.plot(Rx, Ry, Rz, label = "Free Flight Trajectory",linewidth=LVLHTrajWidth[0],alpha=TrajAlpha[0],linestyle = TrajStyle[0],color = TrajColor[0])
    ax.plot(Rx_mano, Ry_mano, Rz_mano, label = "Maneuver Segment",linewidth=LVLHTrajWidth[1],alpha=TrajAlpha[1],linestyle = TrajStyle[1],color = TrajColor[1])
    ax.scatter3D(Rx[0], Ry[0], Rz[0], color=PointColor[0], s=PointSize, label='Start',zorder=10,marker=PointMarker[0])
    ax.scatter3D(Rx[-1], Ry[-1], Rz[-1], color=PointColor[1], s=PointSize, label='End',zorder=10,marker=PointMarker[1])

    ax.set_xlabel('R (km)')
    ax.set_ylabel('T (km)')
    ax.set_zlabel('N (km)')

    ax.set_xlim3d(x_min, x_max)
    ax.set_ylim3d(y_min, y_max)
    ax.set_zlim3d(z_min, z_max)

    if IsAspect:
        PlotAxis(ax,length=5,buffer=0.7,label='R,T,N')
        x_range = x_max - x_min
        y_range = y_max - y_min
        z_range = z_max - z_min
        PlotBox(ax, BoxSize,shift)
        ax.set_box_aspect([x_range, y_range, z_range])
        ax.legend(loc = "upper left")
        ax.view_init(elev=-50, azim=-60, roll=65)
        ax.set_ylabel('T (km)',labelpad=50)
    else:
        PlotAxis(ax,length=5,buffer=0.7,label='R,T,N')
        ax.legend()

    ax.grid(False)
    plt.tight_layout()

    # 保存3D图
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"3D图已保存至: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()

    # --- RT 平面图 ---
    plt.figure(figsize=(10,6))
    ax_rt = plt.axes()
    ax_rt.plot(Ry, Rx, label="Free Flight Trajectory",
               linewidth=LVLHTrajWidth[0], alpha=TrajAlpha[0],
               linestyle=TrajStyle[0], color=TrajColor[0])
    ax_rt.plot(Ry_mano, Rx_mano, label="Maneuver Segment",
               linewidth=LVLHTrajWidth[1], alpha=TrajAlpha[1],
               linestyle=TrajStyle[1], color=TrajColor[1])
    ax_rt.scatter(Ry[0], Rx[0], color=PointColor[0], s=PointSize,
                  label='Start', zorder=10, marker=PointMarker[0])
    ax_rt.scatter(Ry[-1], Rx[-1], color=PointColor[1], s=PointSize,
                  label='End', zorder=10, marker=PointMarker[1])

    ax_rt.set_xlabel('T (km)')
    ax_rt.set_ylabel('R (km)')
    ax_rt.set_xlim(y_min, y_max)
    ax_rt.set_ylim(x_min, x_max)

    # 原点坐标轴
    PlotAxis2D(ax_rt, length=3, buffer=0.7, label='R,T')

    # Station Keeping Box (RT平面投影)
    box_t_half = BoxSize[1] / 2
    box_r_half = BoxSize[0] / 2
    t_min = -box_t_half - shift
    t_max =  box_t_half - shift
    r_min = -box_r_half
    r_max =  box_r_half
    ax_rt.add_patch(plt.Rectangle(
        (t_min, r_min), t_max - t_min, r_max - r_min,
        fill=False, edgecolor='black', linewidth=2.0, alpha=1,
        label='Station Keeping Box'
    ))

    if IsAspect:
        ax_rt.set_aspect('equal')

    ax_rt.legend(loc = "upper right")
    ax_rt.grid(True)
    plt.tight_layout()

    # 保存RT平面图
    if save_path:
        base, ext = os.path.splitext(save_path)
        save_path_rt = f"{base}_RT{ext}"
        plt.savefig(save_path_rt, dpi=300)
        print(f"RT平面图已保存至: {save_path_rt}")

    if show:
        plt.show()
    else:
        plt.close()
    return

def PlotTrajectory_LVLH_Scene2(LVLHPath,Target,IsAspect=True,save_path = None,show=False):
    """绘制卫星在LVLH坐标系下的轨迹"""
    Time,Rx,Ry,Rz,Flag = SequenceData.from_csv(LVLHPath,Col=["Time","R_x","R_y","R_z","Flag"]).to_tuple()
    # RefX,RefY,RefZ = SequenceData.from_csv('./Output/ReferenceTrajectory.csv',Col=["x","y","z"]).to_tuple()

    Rx_mano = Rx[Flag=="manoeuvre"]
    Ry_mano = Ry[Flag=="manoeuvre"]
    Rz_mano = Rz[Flag=="manoeuvre"]

    # 绘图
    plt.rcParams.update({'font.size': 14})
    plt.figure(figsize=(8,8))
    ax = plt.axes(projection='3d')
    
    # ax.plot(RefX, RefY, RefZ, label = "Reference Trajectory",linewidth=2,alpha=0.5,linestyle = '--',color = 'gray',marker = 'o',markersize=2)
    ax.plot(Rx, Ry, Rz, label = "Free Flight Trajectory",linewidth=LVLHTrajWidth[0],alpha=TrajAlpha[0],linestyle = TrajStyle[0],color = TrajColor[0])
    ax.plot(Rx_mano, Ry_mano, Rz_mano, label = "Maneuver Segment",linewidth=LVLHTrajWidth[1],alpha=TrajAlpha[1],linestyle = TrajStyle[1],color = TrajColor[1])
    ax.scatter3D(Rx[0], Ry[0], Rz[0], color=PointColor[0], s=PointSize, label='Start',zorder=10,marker=PointMarker[0])
    ax.scatter3D(Rx[-1], Ry[-1], Rz[-1], color=PointColor[1], s=PointSize, label='End',zorder=10,marker=PointMarker[1])
    ax.scatter3D(Target[0], Target[1], Target[2], color=PointColor[2], s=PointSize, label='Target',zorder=10,marker=PointMarker[2])

    ax.set_xlabel('X (km)')
    ax.set_ylabel('Y (km)')
    ax.set_zlabel('Z (km)')

    # ax.title.set_text('Satellite Trajectory in LVLH Frame')

    x_min, x_max = -5, 5
    y_min, y_max = -30, 10
    z_min, z_max = -5, 5

    ax.set_xlim3d(x_min, x_max)
    ax.set_ylim3d(y_min, y_max)
    ax.set_zlim3d(z_min, z_max)

    if IsAspect:
        x_range = x_max - x_min
        y_range = y_max - y_min
        z_range = z_max - z_min
        ax.set_box_aspect([x_range, y_range, z_range])
        ax.legend(loc = "upper left")
        # ax.view_init(elev=60, azim=-45)
        PlotAxis(ax,length=5,buffer=0.7)
        ax.set_ylabel('Y (km)',labelpad=50)
    else:
        PlotAxis(ax,length=5,buffer=0.7)
        ax.legend()

    ax.grid(False)
    plt.tight_layout()

    # 标准保存显示设置
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"图片已保存至: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    return

def PlotTrajectory_LVLH_Reprop(Dir,PathList,save_path = None,show =False):
    """绘制卫星重新推演下在LVLH坐标系下的轨迹"""

    def _PlotSingleLVLH3D(ax,RepropLVLHPath,idx):
        """绘制单独一组卫星重新推演下在LVLH坐标系下的轨迹"""
        Rxr,Ryr,Rzr = SequenceData.from_csv(RepropLVLHPath,Col=["R_x","R_y","R_z"]).to_tuple()

        ax.plot(Rxr, Ryr, Rzr, label = f"Repropagated Trajectory {RepoMethod[idx]}",linewidth=RepoLinewidth,alpha=RepoTrajAlpha,color=RepoTrajColor[idx])
        ax.scatter3D(Rxr[-1], Ryr[-1], Rzr[-1], label=f'{RepoMethod[idx]} End',marker = RepoMarker[idx],color=RepoMarkerColor[idx], s=RepoMarkerSize)

        return

    # 绘图设置
    plt.rcParams.update({'font.size': 14})
    plt.figure(figsize=(10,10))
    ax = plt.axes(projection='3d')

    # 解析真实推力
    LVLHPath = os.path.join(Dir, 'Data_LVLH.csv')
    _,Rx,Ry,Rz,Flag = SequenceData.from_csv(LVLHPath,Col=["Time","R_x","R_y","R_z","Flag"]).to_tuple()
    Rx_mano = Rx[Flag=="manoeuvre"]
    Ry_mano = Ry[Flag=="manoeuvre"]
    Rz_mano = Rz[Flag=="manoeuvre"]

    # 绘制真实机动
    ax.plot(Rx_mano, Ry_mano, Rz_mano, label = "True Maneuver",alpha=TrueTrajAlpha,linewidth=TrueTrajwidth,color = TrueTrajColor,linestyle = TrueTrajStyle)
    ax.scatter3D(Rx_mano[0], Ry_mano[0], Rz_mano[0], color=TrueMarkerColor[0], s=TrueMarkerSize, label='Maneuver Start')
    ax.scatter3D(Rx_mano[-1], Ry_mano[-1], Rz_mano[-1], color=TrueMarkerColor[1], s=TrueMarkerSize, label='True End')
    
    # 基于PathList解析并绘图
    n = len(PathList)
    for idx in range(n):
        RepropLVLHPath = os.path.join(Dir, PathList[idx])
        _PlotSingleLVLH3D(ax,RepropLVLHPath,idx)

    # 图像后处理
    ax.set_xlabel('X (km)')
    ax.set_ylabel('Y (km)')
    ax.set_zlabel('Z (km)')
    ax.grid(False)
    ax.axis('equal')
    plt.legend()
    plt.tight_layout()

    # 标准保存显示设置
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"图片已保存至: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return

def PlotTrajectory_LVLH_Reprop_XY(Dir,PathList,IsAspect=True,save_path=None,show=False):
    """在XY平面上绘制卫星重新推演下在LVLH坐标系下的轨迹"""

    def _PlotSingleLVLH(ax,RepropLVLHPath,idx):
        """绘制单独一组卫星重新推演下在LVLH坐标系下的轨迹"""
        Rxr,Ryr = SequenceData.from_csv(RepropLVLHPath,Col=["R_x","R_y"]).to_tuple()

        ax.plot(Ryr, Rxr, label = f"{RepoMethod[idx]} Trajectory ",linewidth=RepoLinewidth,linestyle = RepoTrajStyle[idx],alpha=RepoTrajAlpha,color=RepoTrajColor[idx],zorder=1)
        ax.scatter(Ryr[-1], Rxr[-1], label=f'{RepoMethod[idx]} End',marker = RepoMarker[idx],color=RepoMarkerColor[idx], s=RepoMarkerSize,zorder = 3)

        return

    # 绘图设置
    plt.rcParams.update({'font.size': 12})
    fig = plt.figure(figsize=(10, 6))

    ## 第一张子图：等比
    ax = fig.add_subplot(2,1,1)

    # 解析真实推力
    LVLHPath = os.path.join(Dir, 'Data_LVLH.csv')
    _,Rx,Ry,Flag = SequenceData.from_csv(LVLHPath,Col=["Time","R_x","R_y","Flag"]).to_tuple()
    Rx_mano = Rx[Flag=="manoeuvre"]
    Ry_mano = Ry[Flag=="manoeuvre"]

    # 绘制真实机动
    ax.plot(Ry_mano, Rx_mano, label = "True Manoeuvre",alpha=TrueTrajAlpha,linewidth=TrueTrajwidth,color = TrueTrajColor,linestyle = TrueTrajStyle)
    ax.scatter(Ry_mano[0], Rx_mano[0], color=TrueMarkerColor[0], s=TrueMarkerSize, label='Manoeuvre Start',zorder=3)
    ax.scatter(Ry_mano[-1], Rx_mano[-1], marker="*",color=TrueMarkerColor[1], s=TrueMarkerSize, label='True End',zorder=3)

    # 基于PathList解析并绘图
    n = len(PathList)
    for idx in range(n):
        RepropLVLHPath = os.path.join(Dir, PathList[idx])
        _PlotSingleLVLH(ax,RepropLVLHPath,idx)

    # 图像后处理
    ax.set_ylabel('R (km)')
    ax.set_xlabel('T (km)')
    ax.grid(True,alpha=0.3)
    ax.set_title("Equal Aspect")
    plt.tight_layout()
    plt.legend(bbox_to_anchor=(1.05, 1),loc="upper left")
    ax.axis('equal')

    # ax.set_xlim([-25,15])
    # ax.set_ylim([-10,10])

    ax.set_xlim([-35,5])
    ax.set_ylim([-10,10])

    # ax.set_xlim([-5,40])
    # ax.set_ylim([-2,2])

    ## 第二张子图：非等比
    ax = fig.add_subplot(2,1,2)

    # 绘制真实机动
    ax.plot(Ry_mano, Rx_mano, label = "True Maneuver",alpha=TrueTrajAlpha,linewidth=TrueTrajwidth,color = TrueTrajColor,linestyle = TrueTrajStyle)
    ax.scatter(Ry_mano[0], Rx_mano[0], color=TrueMarkerColor[0], s=TrueMarkerSize, label='Maneuver Start',zorder=3)
    ax.scatter(Ry_mano[-1], Rx_mano[-1], color=TrueMarkerColor[1], s=TrueMarkerSize, label='True End',zorder=3)

    # 基于PathList解析并绘图
    n = len(PathList)
    for idx in range(n):
        RepropLVLHPath = os.path.join(Dir, PathList[idx])
        _PlotSingleLVLH(ax,RepropLVLHPath,idx)

    # 图像后处理
    ax.set_ylabel('R (km)')
    ax.set_xlabel('T (km)')
    ax.set_title("Auto-Scaled")
    ax.grid(True,alpha=0.3)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"图片已保存至: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()