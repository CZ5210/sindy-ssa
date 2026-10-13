from sympy import *
from .SINDy_Main import *
from Plotter import *
import pysindy as ps

def SINDy_Header1():
    p, f, g, h, k, L = symbols('p f g h k L')

    n=20
    alpha = 1

    header_text = [
        1, 
        f, g, h, k, L,
        f**2, g**2, h**2, k**2,
        f*g, f*h, f*k, f*L,
        g*h, g*k, h*k,
        f**3, g**3, h**3, k**3, L**3,
        f*g*h, f*g*k, f*h*k, g*h*k,

        h*sin(L), h*cos(L),
        f*sin(L), g*cos(L),

        sin(L), cos(L),
        sin(L)**2, cos(L)**2,
        sin(L)**3, cos(L)**3,
    ]

    # 转换为数值函数
    header = [lambdify([p, f, g, h, k, L], expr, modules=['numpy']) for expr in header_text]
    return header, header_text, n, alpha

def SINDy_Ori(Library,ObservePath,window,SamplingRate,OutDir,ThrustPath):
    """原始的SINDy方法,使用PySINDy"""

    # 读取观测到的所有数据
    ObserveData = SequenceData.from_csv(ObservePath)

    # 从观测数据中做有限差分得到MEEdot
    MEEdot_FD = FD_MEE(ObserveData)

    # 提取MEEData
    MEEData = SequenceData.from_csv(ObservePath,Col = ["Time","p","f","g","h","k","L"])

    # 筛选机动段数据
    count,interval = Extract_Manoeuvre(ObservePath)
    print(f"共检测到 {count} 个机动段")
    MEEData = MEEData.filter(interval[0])
    MEEdot_FD = MEEdot_FD.filter(interval[0])

    ## 进行SINDy回归

    # 构建X,Xdot
    TIME,p,f,g,h,k,L = MEEData.to_tuple()
    X = np.column_stack((p/6371, f, g, h, k, L))

    _,pdot,fdot,gdot,hdot,kdot,Ldot = MEEdot_FD.to_tuple()
    Xdot = np.column_stack((pdot,fdot,gdot,hdot,kdot,Ldot))

    # 构建图书馆Theta矩阵
    header, header_text, n,alpha = Library() 
    library = ps.CustomLibrary(library_functions=header).fit(X)
    Theta = library.transform(X)

    # 使用SINDy优化器拟合回归方程,获取系数矩阵Xi
    optimizer = ps.SSR(max_iter=n, alpha=alpha)
    optimizer.fit(Theta, Xdot)
    Xi = optimizer.coef_.T

    # 重建SINDy的状态矩阵
    Theta = library.fit_transform(X)
    XRec = optimizer.predict(Theta)

    # 保存XRec,Theta,Xi矩阵
    header = ['pdot','fdot','gdot','hdot','kdot','Ldot']
    os.makedirs(OutDir,exist_ok=True)
    Save_Matrix(Xi, f'{OutDir}/Xi.csv', row_headers=header_text, column_headers=header, precision = 8, output = True)
    Save_Matrix(XRec, f'{OutDir}/XRec.csv', column_headers=header, precision = 8, output = True)
    Save_Matrix(Theta, f'{OutDir}/Theta.csv', column_headers=header_text, precision = 8, output = True)

    # 创建通过SINDy重建的MEEdotRec
    MEEdotRecData = SequenceData.from_concat(SequenceData([Sequence([TIME],'Time')]),
                                             SequenceData.from_csv(f'{OutDir}/XRec.csv',
                                                                   Col=["pdot","fdot","gdot","hdot","kdot","Ldot"]))
    
    # 使用GVE摄动方法计算自然MEEdot
    MEEdot_natural = GVE_MEE(ObserveData).filter(interval[0])

    # 将两个MEEdot转化到加速度数据
    u_natural = Element2AccData(ObserveData,MEEdot_natural)
    u_all = Element2AccData(ObserveData,MEEdotRecData)

    # 将加速度数据做减法转化到推力数据
    U = MinusAcc(u_all,u_natural).smooth(window=window,idx=0)
    U = U.linearinterp(SamplingRate)
    U.save(path=ThrustPath)
    print("[SINDy重建] 基于原始SINDy的重建数据已保存。")

    return


