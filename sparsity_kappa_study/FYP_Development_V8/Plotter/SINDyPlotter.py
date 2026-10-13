import matplotlib.pyplot as plt
from SINDy.Sequence import *
import seaborn as sns

def init_SINDy_plot():
    """初始化所有颜色"""

    # 原始元素颜色选项
    global ElementMethods,ElementLabels,ElementColor,ElementStyle,ElementAlpha,ElementWidth
    ElementMethods=["Observed","Reconstructed"]
    ElementLabels = [r'$\dot{p}$', r'$\dot{f}$', r'$\dot{g}$', r'$\dot{h}$', r'$\dot{k}$', r'$\dot{L}$']
    ElementColor = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    ElementStyle = ['-','--',"-.",':']
    ElementAlpha = [0.5,1.0,0.5]
    ElementWidth = [2,2]

    # Xi矩阵颜色选项
    global cmap
    cmap = "plasma"
    # cmap = "coolwarm"

    # 加速度曲线选项
    global AccMethods, AccColor, AccStyle, AccLabels, AccAlpha,AccWidth
    AccMethods = ["True","MEEFD","SINDy","SINDy-TIME","SINDy-UMEE"]
    AccColor = ["k","r","g","b","c"]
    AccStyle = ['-','--',"-.",':',"-"]
    AccLabels = [r'$u_r$', r'$u_t$', r'$u_n$']
    AccAlpha = [1.0,0.5,0.5,0.5,0.5]
    AccWidth = [2,2,2,2,2]

    # 绘制Sparse曲线时用
    # AccMethods = ["True","Non-Sparse","Sparse","Very-Sparse"]

def PlotElementData(ElementDatas:list[SequenceData],save_path = None, show = False):
    """六元素信息序列绘图函数，支持n个不同计算方式的数据对比"""

    n = len(ElementDatas)
    
    # 创建画布
    plt.figure(figsize=(18, 10))
    
    # 循环创建6个子图
    for i in range(6):
        plt.subplot(2, 3, i+1)
        plt.grid('on',alpha=0.5)
        
        # 循环绘制当前子图中所有数据的序列
        for j in range(n):
            labels = ElementLabels[i] + "(" +ElementMethods[j] + ")"
            time,e1,e2,e3,e4,e5,e6 = ElementDatas[j].to_tuple()

            if i==0:
                data = e1
            elif i==1:
                data = e2
            elif i==2:
                data = e3
            elif i==3:
                data = e4
            elif i==4:
                data = e5
            elif i==5:
                data = e6
                
            # 绘制曲线
            plt.plot(time / 3600, data, color=ElementColor[i],label=labels,linewidth=ElementWidth[j], alpha=ElementAlpha[j])
        
        # 子图配置
        plt.legend(loc = 'upper left')
        plt.xlabel("Time(h)")
        plt.ylabel(ElementLabels[i])
    
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

def PlotAccData(idx,AccDatas:list[SequenceData],save_path=None, show=False):
    """加速度数据绘图函数，支持多组加速度数据对比"""

    plt.figure(figsize=(8, 6))
    
    # 循环创建3个子图
    for i in range(3):
        plt.subplot(3, 1, i+1)
        plt.grid('on', alpha=0.5)
        
        # 循环绘制当前子图中所有组数据
        for k in idx:
            labels = AccMethods[k]
            time,ur,ut,un = AccDatas[k].to_tuple()

            if i==0:
                data = ur
            elif i==1:
                data = ut
            else:
                data = un

            # 绘制曲线
            plt.plot(time / 3600,data,color=AccColor[k],linestyle=AccStyle[k],label=labels,linewidth=AccWidth[k],alpha=AccAlpha[k])
        
        # 子图配置
        if i ==0:
            plt.legend(bbox_to_anchor=(1, 1,0,0.2),loc="upper left",fontsize=12)
        plt.xlabel("Time (h)",fontsize=12)
        plt.ylabel(AccLabels[i] + r"$\quad(km/s^2)$",fontsize=12)
        # if i == 2:
            # plt.ylim([-1e-8,1e-8])
            # plt.ylim([-1e-7,1e-7])   

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

def PlotAllAccs(Dirs,save_path=None):
    """绘制所有加速度数据的对比图"""

    Cols = ['Time','u_r','u_t','u_n']

    acc_datas = []
    for dir_path in Dirs:
        acc_data = SequenceData.from_csv(dir_path, Col=Cols)
        acc_datas.append(acc_data)

    # 1. 绘制所有数据的整体对比图
    n = len(acc_datas)
    indices_all = list(range(n))
    PlotAccData(indices_all, acc_datas, save_path=save_path)
    
    # 2. 绘制第一个数据与其余每个数据的单独对比图
    if n >= 2:
        for i in range(1, n):
            indices_single = [0, i]
            save_path = f"Figures/Acc{i}.png"
            PlotAccData(indices_single, acc_datas, save_path=save_path)

def Plot_Xi_Heatmap(filepath, textsize =8, figsize=(12, 18), Anno=True, save_path = None,show = False, x_labels = None):
    """绘制SINDy回归Xi矩阵的绝对值热图"""

    # 构建文件路径
    df = pd.read_csv(filepath)
    feature_names = df.iloc[:, 0].values
    derivatives = df.columns[1:]
    xi_matrix = df.iloc[:, 1:].values
    xi_abs = abs(xi_matrix)
    # xi_abs = xi_matrix
    mask = abs(xi_abs) < 1e-90

    # 自定义X标签
    if x_labels is not None:
        x_ticklabels = x_labels
    else:
        x_ticklabels = derivatives
    
    # 创建图像
    plt.figure(figsize=figsize)
    
    # 绘制热图
    if not Anno:
        feature_names = []

    ax = sns.heatmap(
        xi_abs,
        annot=Anno,
        fmt='.3e',
        mask=mask,
        cmap=cmap,
        cbar=True,
        cbar_kws={'label': 'Coefficients Absolute Value'},
        linewidths=0,
        xticklabels=x_ticklabels,
        yticklabels=feature_names,
        annot_kws={'size': textsize, 'va': 'center_baseline'}
    )

    cbar = ax.collections[0].colorbar

    cbar.set_label(
        label='Coefficients Absolute Value',
        fontsize=12
    )

    # 美化标签
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(rotation=0, fontsize=10)
    plt.ylabel('Features', fontsize=12)
    
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