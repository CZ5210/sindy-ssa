from sympy import *
import numpy as np
from .SINDy_Main import *
from Plotter import *
import pysindy as ps

def UnwrapL(L, threshold: float = -np.pi):
    """计算真近点角L的累积值L_total"""

    # 初始化圈数和累积真近点角
    N = np.zeros_like(L, dtype=int)
    L_total = np.copy(L)
    current_N = 0  # 当前圈数
    
    # 计算相邻元素的差值（ΔL = L[i+1] - L[i]）
    delta_L = np.diff(L)
    
    # 检测跳变索引：ΔL < threshold（从2π附近跳回0附近）
    jump_indices = np.where(delta_L < threshold)[0]  # jump_indices是i的索引，对应L[i+1]是跳变点
    
    # 遍历跳变点，更新圈数和累积L
    for idx in jump_indices:
        current_N += 1
        N[idx+1:] = current_N
        L_total[idx+1:] = L[idx+1:] + 2 * np.pi * current_N
    
    return N, L_total

def SINDyUMEE_Header1():

    n = 30
    alpha = 1e-3

    p, f, g, h, k, L = symbols('p f g h k L')
    header_text_sympy = [
        p**0, p, f, g, h, k, L,
        p**2,f**2,g**2,h**2,k**2,L**2,
        (f*f+g*g)*sin(L),
        sin(L), cos(L), sin(L)*cos(L),
        sin(2*L), cos(2*L), sin(2*L)*cos(2*L),
        p*sin(L),f*sin(L), g*cos(L) ,h*sin(L), k*cos(L), L*sin(L),
        p*sin(2*L),f*sin(2*L), g*cos(2*L) ,h*sin(2*L), k*cos(2*L), L*sin(2*L),
    ]
    header = []
    header_text = []
    for expr in header_text_sympy:
        func = lambdify((p, f, g, h, k, L), expr, modules=['numpy'])
        header.append(func)
        header_text.append(str(expr).replace('**', '^'))
    
    return header, header_text, n,alpha

def SINDy_UMEE(Library,SamplingRate,ThrustPath_raw,ObservePath,OutDir):
    """SINDy_UMEE方法,使用PySINDy"""

    # 获取嘈杂的推力数据
    coeffs,AccDataTrain = GetMEEFDthrust(ThrustPath_raw)
    # 构建训练数据
    ur = AccDataTrain.get("u_r")
    ut = AccDataTrain.get("u_t")
    un = AccDataTrain.get("u_n")
    t = np.array([AccDataTrain.get("Time").Data]).T
    # u = np.array([ur.Data,ut.Data,un.Data]).T

    u = np.array([ur.Data,ut.Data]).T

    # 获取观测段MEE数据
    MEEData = SequenceData.from_csv(ObservePath,Col=["Time","p","f","g","h","k","L"])
    _,interval = Extract_Manoeuvre(ObservePath)
    MEEData = MEEData.filter(interval[0])

    _,p,f,g,h,k,L = MEEData.to_tuple()
    # N, L = UnwrapL(L)
    X = np.column_stack((p/6371, f, g, h, k, L))

    TIME,p,f,g,h,k,L = MEEData.linearinterp(SamplingRate).to_tuple()
    # N, L = UnwrapL(L)
    Xf = np.column_stack((p/6371, f, g, h, k, L))

    # 构建图书馆Theta矩阵
    header, header_text, n,alpha = Library() 
    library = ps.CustomLibrary(library_functions=header).fit(X)
    Theta = library.transform(X)

    # 使用SINDy的优化器直接拟合方程,获取系数矩阵Xi
    optimizer = ps.SSR(max_iter=n, alpha=alpha)
    optimizer.fit(Theta, u)
    Xi = optimizer.coef_.T

    # 重建SINDy的状态矩阵
    Theta = library.fit_transform(Xf)
    XRec = optimizer.predict(Theta)

    # 保存XRec,Theta,Xi矩阵
    # u_header = [r'u_r',r'u_t',r'u_n']
    u_header = [r'u_r',r'u_t']
    os.makedirs(OutDir,exist_ok=True)
    Save_Matrix(Xi, f'{OutDir}/Xi.csv', row_headers=header_text, column_headers=u_header, precision = 8, output = True)
    Save_Matrix(XRec, f'{OutDir}/XRec.csv', column_headers=u_header, precision = 8, output = True)
    Save_Matrix(Theta, f'{OutDir}/Theta.csv', column_headers=header_text, precision = 8, output = True)

    # 将重建状态转化回原始推力并保存

    _,_,coeff_r,coeff_t,coeff_n = coeffs
    t = Sequence(TIME,"Time")
    ur = Sequence(coeff_r * XRec[:,0],"u_r")
    ut = Sequence(coeff_t * XRec[:,1],"u_t")
    # un = Sequence(coeff_n * XRec[:,2],"u_n")
    un = Sequence(np.zeros(len(TIME)), "u_n")

    AccRecData = SequenceData([t,ur,ut,un],"AccRecData")
    AccRecData.save(f"{OutDir}/U_SINDyUMEE.csv")

    return