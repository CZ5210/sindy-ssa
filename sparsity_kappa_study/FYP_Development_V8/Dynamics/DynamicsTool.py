import numpy as np
from numpy.linalg import norm

def Outside_Box(R_LVLH,BoxSize,shift):
    """用于判断当前实际卫星是否在轨道维持盒内"""

    Sx,Sy,Sz = BoxSize[0],BoxSize[1],BoxSize[2]
    x,y,z = R_LVLH

    if (abs(x)>Sx/2) or (abs(y+shift)>Sy/2) or (abs(z)>Sz/2):
        return True
    else:
        return False

def ComputeLQRControl(A,B,Q,R):
    import scipy.linalg as la
    # 求解连续时间代数Riccati方程
    P = la.solve_continuous_are(A, B, Q, R)
    # 计算控制增益
    K = np.linalg.inv(R) @ B.T @ P
    return K

def CWEquation(r):

    # 平均运动
    omega = np.sqrt(398600/ r**3)

    # 计算CW方程的状态空间表示
    # 状态向量 [x, y, z, vx, vy, vz]
    A = np.array([
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1],
        [3*omega**2, 0, 0, 0, 2*omega, 0],
        [0, 0, 0, -2*omega, 0, 0],
        [0, 0, -omega**2, 0, 0, 0]
    ])

    B = np.array([
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])

    return A, B

def ECI2LVLH(R,V,Vec):
    """将 ECI 坐标系下的矢量转换为 LVLH 坐标系下的矢量"""

    # 计算 LVLH 方向单位向量
    i_r = R / norm(R)
    i_n = np.cross(R, V) / norm(np.cross(R, V))
    i_t = np.cross(i_n, i_r) / norm(np.cross(i_n, i_r))

    # 变换到 LVLH 坐标系
    Rot = np.array([i_r,i_t,i_n])
    Vec_LVLH = np.dot(Rot,Vec).flatten()

    return Vec_LVLH

def LVLH2ECI(R,V,U_LVLH):

    """
    该函数将 LVLH 坐标系下的推力转换为 ECI 坐标系下推力
    """

    # 计算 LVLH 方向单位向量
    i_r = R / norm(R)
    i_n = np.cross(R, V) / norm(np.cross(R, V))
    i_t = np.cross(i_n, i_r)

    # 变换到 ECI 坐标系
    U_ECI = U_LVLH[0]*i_r + U_LVLH[1]*i_t + U_LVLH[2]*i_n

    return U_ECI

def RV2MEE(R, V, mu=398600):
    """
    将RV状态向量转化为春分点六根数
    """
    
    r = norm(R)
    v = norm(V)

    h_vec = np.cross(R, V)
    h = norm(h_vec)

    i = np.arccos(h_vec[2] / h)

    e_vec = (np.cross(V, h_vec) / mu) - R / r
    e = norm(e_vec)

    energy = v**2 / 2 - mu / r
    a = -mu / (2 * energy)
    p = a * (1 - e**2)

    K = np.array([0, 0, 1])
    n_vec = np.cross(K, h_vec)
    n = norm(n_vec)

    # 计算角度
    if n != 0:
        RAAN = np.arccos(n_vec[0] / n)
        if n_vec[1] < 0:
            RAAN = 2 * np.pi - RAAN
    else:
        RAAN = 0

    if e > 1e-10 and n != 0:
        omega = np.arccos(np.dot(n_vec, e_vec) / (n * e))
        if e_vec[2] < 0:
            omega = 2 * np.pi - omega
    else:
        omega = 0

    if e > 1e-10:
        nu = np.arccos(np.dot(e_vec, R) / (e * r))
        if np.dot(R, V) < 0:
            nu = 2 * np.pi - nu
    else:
        nu = 0

    # 构造 MEE
    L = (RAAN + omega + nu) % (2 * np.pi)
    f = e * np.cos(RAAN + omega)
    g = e * np.sin(RAAN + omega)
    h = np.tan(i / 2) * np.cos(RAAN)
    k = np.tan(i / 2) * np.sin(RAAN)

    return np.array([p, f, g, h, k, L])

def MEE2RV(MEE, mu =398600):
    """将春分点根数转化为RV"""

    p,f,g,h,k,L = MEE

    sinL = np.sin(L)
    cosL = np.cos(L)

    alpha = h*h - k*k
    s2 = 1 + h*h + k*k
    w = 1 + f*cosL + g*sinL
    r = p/w
    SQRT = np.sqrt(mu/p)

    Rx = (r/s2)*(cosL+alpha*cosL+2*h*k*sinL)
    Ry = (r/s2)*(sinL-alpha*sinL+2*h*k*cosL)
    Rz = (2*r/s2)*(h*sinL-k*cosL)

    Vx = (-1/s2)*SQRT*(sinL+alpha*sinL-2*h*k*cosL+g-2*f*h*k+alpha*g)
    Vy = (-1/s2)*SQRT*(-cosL+alpha*cosL+2*h*k*sinL-f+2*g*h*k+alpha*f)
    Vz = (2/s2)*SQRT*(h*cosL+k*sinL+f*h+g*k)

    R = np.array([Rx,Ry,Rz])
    V = np.array([Vx,Vy,Vz])

    return R,V

def InitOrbit(alt=500,inc=0):

    """给定卫星的初始轨道高度，倾角，返回ECI下的R,V"""

    inc = np.deg2rad(inc)
    R = np.array([alt+6371,0,0])
    v = np.sqrt(398600/(alt+6371))
    V = np.array([0,v*np.cos(inc),v*np.sin(inc)])

    return R,V    