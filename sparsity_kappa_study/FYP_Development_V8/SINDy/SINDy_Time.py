import numpy as np
import pysindy as ps
from sympy import *
from Plotter import *
from .SINDy_Main import *

def SINDyTIME_Header1():
    """SINDy基函数库"""

    n = 25
    alpha = 1e-3
    kappa = 1e-19

    t = symbols('t')
    epi = 1e-7
    header_text = [
        1,t,t**2,t**3,
        (t+epi)**(0.5),(t+epi)**(1.5),(t+epi)**(2.5),
        (t+epi)**(3.5),(t+epi)**(4.5),(t+epi)**(5.5),
        sin((t+epi)**(0.5)),sin((t+epi)**(1.5)),sin((t+epi)**(2.5)),
        sin((t+epi)**(3.5)),sin((t+epi)**(4.5)),sin((t+epi)**(5.5)),
        cos((t+epi)**(0.5)),cos((t+epi)**(1.5)),cos((t+epi)**(2.5)),
        cos((t+epi)**(3.5)),cos((t+epi)**(4.5)),cos((t+epi)**(5.5)),
    ]

    # 转换为数值函数
    header = [lambdify([t], expr, modules=['numpy']) for expr in header_text]
    return header, header_text, n, alpha, kappa

def SINDyTIME_Header2():
    """SINDy基函数库"""

    n = 5
    alpha = 1e-4
    kappa = 0

    t = symbols('t')
    epi = 1e-7
    header_text = [
        1,t,t**2,t**3,
        (t+epi)**(0.5),(t+epi)**(1.5),(t+epi)**(2.5),
        (t+epi)**(3.5),(t+epi)**(4.5),(t+epi)**(5.5),
        sin((t+epi)**(0.5)),sin((t+epi)**(1.5)),sin((t+epi)**(2.5)),
        sin((t+epi)**(3.5)),sin((t+epi)**(4.5)),sin((t+epi)**(5.5)),
        cos((t+epi)**(0.5)),cos((t+epi)**(1.5)),cos((t+epi)**(2.5)),
        cos((t+epi)**(3.5)),cos((t+epi)**(4.5)),cos((t+epi)**(5.5)),
    ]

    # 转换为数值函数
    header = [lambdify([t], expr, modules=['numpy']) for expr in header_text]
    return header, header_text, n, alpha, kappa

def SINDyTIME_Header3():
    """SINDy基函数库"""

    n = 50
    alpha = 1e-1
    kappa = 1e-15

    t = symbols('t')
    epi = 1e-7
    header_text = [
        1,t,t**2,t**3,
        (t+epi)**(0.5),(t+epi)**(1.5),(t+epi)**(2.5),
        (t+epi)**(3.5),(t+epi)**(4.5),(t+epi)**(5.5),
        sin((t+epi)**(0.5)),sin((t+epi)**(1.5)),sin((t+epi)**(2.5)),
        sin((t+epi)**(3.5)),sin((t+epi)**(4.5)),sin((t+epi)**(5.5)),
        cos((t+epi)**(0.5)),cos((t+epi)**(1.5)),cos((t+epi)**(2.5)),
        cos((t+epi)**(3.5)),cos((t+epi)**(4.5)),cos((t+epi)**(5.5)),
    ]

    # 转换为数值函数
    header = [lambdify([t], expr, modules=['numpy']) for expr in header_text]
    return header, header_text, n, alpha, kappa

def SINDy_TIME(Library,SamplingRate,ThrustPath_raw,OutDir):
    """SINDy_Time方法,使用PySINDy"""

    # 获取嘈杂的推力数据
    coeffs,AccDataTrain = GetMEEFDthrust(ThrustPath_raw)

    # 构建训练数据
    ur = AccDataTrain.get("u_r")
    ut = AccDataTrain.get("u_t")
    un = AccDataTrain.get("u_n")
    t = np.array([AccDataTrain.get("Time").Data]).T
    u = np.array([ur.Data,ut.Data,un.Data]).T

    # 构建图书馆Theta矩阵
    header, header_text, n, alpha,kappa = Library()
    library = ps.CustomLibrary(library_functions=header)
    Theta = library.fit_transform(t)

    # 使用SINDy的优化器直接拟合方程,获取系数矩阵Xi
    # optimizer = ps.SSR(max_iter=n, alpha=alpha, kappa=1e-15)
    optimizer = ps.SSR(max_iter=n, alpha=alpha, kappa= kappa)
    optimizer.fit(Theta, u)
    Xi = optimizer.coef_.T

    # 重建SINDy的状态矩阵
    t = np.array([AccDataTrain.get("Time").linearinterp(SamplingRate).Data]).T
    Theta = library.fit_transform(t)
    XRec = optimizer.predict(Theta)

    # 保存XRec,Theta,Xi矩阵
    u_header = [r'u_r',r'u_t',r'u_n']
    os.makedirs(OutDir,exist_ok=True)
    Save_Matrix(Xi, f'{OutDir}/Xi.csv', row_headers=header_text, column_headers=u_header, precision = 8, output = True)
    Save_Matrix(XRec, f'{OutDir}/XRec.csv', column_headers=u_header, precision = 8, output = True)
    Save_Matrix(Theta, f'{OutDir}/Theta.csv', column_headers=header_text, precision = 8, output = True)

    # 将重建状态反归一化转化回原始推力并保存
    coeff_time0,coeff_time1,coeff_r,coeff_t,coeff_n = coeffs

    t = Sequence(coeff_time1 * t + coeff_time0,"Time")
    ur = Sequence(coeff_r * XRec[:,0],"u_r")
    ut = Sequence(coeff_t * XRec[:,1],"u_t")
    un = Sequence(coeff_n * XRec[:,2],"u_n")

    AccRecData = SequenceData([t,ur,ut,un],"AccRecData")
    AccRecData.save(f"{OutDir}/U_SINDyTIME.csv")

    return