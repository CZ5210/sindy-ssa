import numpy as np
from numpy.linalg import norm
from .DataClass import *
from .DynamicsTool import *
from .MEEDynamics import *

def Acc_Gravity(R,mu=398600):
    '''
    返回引力加速度(km/s^2)
    Return acceleration due gravity
    '''
    return -(mu/(norm(R)**3))*R

def __Acc_J2_ECI(R,J2=0.00108263,mu=398600,Re=6378):
    '''
    给定当前的位置R,返回地球的J2摄动加速度 (km/s^2)
    Given the current position R, return the J2 perturbation (km/s^2)
    '''

    x,y,z=R
    r=norm(R)
    Coeff=(3/2)*(J2*mu*Re*Re)/(r**4)
    ax = Coeff * (x/r) * (5*z**2/r**2 - 1)
    ay = Coeff * (y/r) * (5*z**2/r**2 - 1)
    az = Coeff * (z/r) * (5*z**2/r**2 - 3)

    J2acc = np.array([ax,ay,az])
    return J2acc

def __Acc_Drag(R,V,Cd,A,m):
    '''
    给定当前的位置R,速度V,和相关参数,返回大气阻力摄动加速度 (km/s^2)
    Given the current position R, Velocity V, and relavent parameter, return the drag perturbation (km/s^2)
    '''

    def __RelativeVelocity(R,V,omega=2*np.pi/24/3600):
        '''
        给定航天器当前的位置矢量R, 得到当前位置对应的航天器与地球大气的相对速度
        Given the current position R, obtain the relative velocity between spacecraft and Earth's atmosphere
        '''
        Omega = np.array ([0,0,omega])
        VEarth = np.cross (Omega,R)
        Vrel = V-VEarth

        return Vrel
    
    def __atmosphere_density(z):
        """
        **描述 Description**:
        - 计算给定高度的大气密度(kg/m^3)，基于US 1976标准大气模型  
        - Compute the atmospheric density (kg/m^3) at a given altitude using the US Standard Atmosphere 1976 model

        **输入 Input**:
        - z: 高度 (km) | Altitude in kilometers

        **输出 Output**:
        - 返回值: 对应高度的大气密度 (kg/m^3) | Atmospheric density at the given altitude (kg/m^3)
        """
        
        # [1] H. D. Curtis. Orbital Mechanics for Engineering Students Third Edition. Elsevier,2014.

        # Geometric altitudes (km)
        h = [0, 25, 30, 40, 50, 60, 70, 
            80, 90, 100, 110, 120, 130, 140, 
            150, 180, 200, 250, 300, 350, 400, 
            450, 500, 600, 700, 800, 900, 1000]

        # Corresponding densities (kg/m^3) from USSA76
        r = [1.225, 4.008e-2, 1.841e-2, 3.996e-3, 1.027e-3, 3.097e-4, 8.283e-5, 
            1.846e-5, 3.416e-6, 5.606e-7, 9.708e-8, 2.222e-8, 8.152e-9, 3.831e-9, 
            2.076e-9, 5.194e-10, 2.541e-10, 6.073e-11, 1.916e-11, 7.014e-12, 2.803e-12, 
            1.184e-12, 5.215e-13, 1.137e-13, 3.070e-14, 1.136e-14, 5.759e-15, 3.561e-15]

        # Scale heights (km)
        H = [7.310, 6.427, 6.546, 7.360, 8.342, 7.583, 6.661, 
            5.927, 5.533, 5.703, 6.782, 9.973, 13.243, 16.322, 
            21.652, 27.974, 34.934, 43.342, 49.755, 54.513, 58.019, 
            60.980, 65.654, 76.377, 100.587, 147.203, 208.020]

        # Handle altitudes outside of the range
        if z > 1000:
            # If larger than 1000km, ignore drag
            return 0
        if z < 0:
            # If crashed on Earth, return 1.225, and continue to execute
            return 1.225

        # Determine the interpolation interval
        i = 0
        for j in range(27):
            if z >= h[j] and z < h[j + 1]:
                i = j
                break
        if z == 1000:
            i = 26

        # Exponential interpolation
        density = r[i] * np.exp(-(z - h[i]) / H[i])

        return density

    Vrel = __RelativeVelocity(R,V)
    
    vrel = np.linalg.norm(Vrel)
    
    density = __atmosphere_density(norm(R) - 6371)  # 返回 kg/m^3

    Dacc = -0.5 * (density*(10**9)) * vrel * (Cd * (A/(10**6)) / m) * Vrel  # 转换为 km/s^2

    return Dacc

def __Acc_Drag_MEE(MEE,R,V,Cd,A,m,mu=398600):
    '''
    给定当前的位置R,速度V,和相关参数,返回大气阻力摄动加速度 (km/s^2)
    '''

    def __RelativeVelocity(R,V,omega=2*np.pi/24/3600):
        '''
        给定航天器当前的位置矢量R, 得到当前位置对应的航天器与地球大气的相对速度
        '''
        Omega = np.array ([0,0,omega])
        VEarth = np.cross (Omega,R)
        Vrel = V-VEarth

        return Vrel
    
    def __atmosphere_density(z):
        """
        **描述 Description**:
        - 计算给定高度的大气密度(kg/m^3)，基于US 1976标准大气模型  
        - Compute the atmospheric density (kg/m^3) at a given altitude using the US Standard Atmosphere 1976 model

        **输入 Input**:
        - z: 高度 (km) | Altitude in kilometers

        **输出 Output**:
        - 返回值: 对应高度的大气密度 (kg/m^3) | Atmospheric density at the given altitude (kg/m^3)
        """
        
        # [1] H. D. Curtis. Orbital Mechanics for Engineering Students Third Edition. Elsevier,2014.

        # Geometric altitudes (km)
        h = [0, 25, 30, 40, 50, 60, 70, 
            80, 90, 100, 110, 120, 130, 140, 
            150, 180, 200, 250, 300, 350, 400, 
            450, 500, 600, 700, 800, 900, 1000]

        # Corresponding densities (kg/m^3) from USSA76
        r = [1.225, 4.008e-2, 1.841e-2, 3.996e-3, 1.027e-3, 3.097e-4, 8.283e-5, 
            1.846e-5, 3.416e-6, 5.606e-7, 9.708e-8, 2.222e-8, 8.152e-9, 3.831e-9, 
            2.076e-9, 5.194e-10, 2.541e-10, 6.073e-11, 1.916e-11, 7.014e-12, 2.803e-12, 
            1.184e-12, 5.215e-13, 1.137e-13, 3.070e-14, 1.136e-14, 5.759e-15, 3.561e-15]

        # Scale heights (km)
        H = [7.310, 6.427, 6.546, 7.360, 8.342, 7.583, 6.661, 
            5.927, 5.533, 5.703, 6.782, 9.973, 13.243, 16.322, 
            21.652, 27.974, 34.934, 43.342, 49.755, 54.513, 58.019, 
            60.980, 65.654, 76.377, 100.587, 147.203, 208.020]

        # Handle altitudes outside of the range
        if z > 1000:
            # If larger than 1000km, ignore drag
            return 0
        if z < 0:
            # If crashed on Earth, return 1.225, and continue to execute
            return 1.225

        # Determine the interpolation interval
        i = 0
        for j in range(27):
            if z >= h[j] and z < h[j + 1]:
                i = j
                break
        if z == 1000:
            i = 26

        # Exponential interpolation
        density = r[i] * np.exp(-(z - h[i]) / H[i])

        return density

    Vrel = __RelativeVelocity(R,V)
    
    vrel = np.linalg.norm(Vrel)

    p,f,g,h,k,L = MEE
    sinL = np.sin(L)
    cosL = np.cos(L)
    density = __atmosphere_density(norm(R) - 6371)  # 返回 kg/m^3

    Dacc = -0.5 * (density*(10**9)) * vrel * (Cd * (A/(10**6)) / m)  # 转换为 km/s^2

    Vr = np.sqrt(mu/p)*(f*sinL-g*cosL)
    Vt = np.sqrt(mu/p)*(1+f*cosL+g*sinL)
    Dacc_r = Dacc * Vr
    Dacc_t = Dacc * Vt
    Dacc_n = 0

    return np.array([Dacc_r,Dacc_t,Dacc_n])

def __Acc_J2_MEE(MEE,J2=0.00108263,mu=398600,Re=6378):
    """MEE摄动方程直接给出的J2摄动表达式"""
    
    p,f,g,h,k,L = MEE
    sinL = np.sin(L)
    cosL = np.cos(L)
    s2 = 1+ h*h +k+k
    w = 1 + f*cosL + g*sinL
    r = p/w #经后续验证，这里的r和航天器到地心距离一致，不需要额外转化计算

    R = (-(3*mu*J2*Re*Re)/(r**4))*(1-(12*(h*sinL-k*cosL)**2)/(s2*s2))
    T = (-(12*mu*J2*Re*Re)/(r**4))*((h*sinL-k*cosL)*(h*cosL+k*sinL)/(s2*s2))
    N = (-(6*mu*J2*Re*Re)/(r**4))*((1-h*h-k*k)*(h*sinL-k*cosL)/(s2*s2))

    Acc_J2 = R,T,N
    return Acc_J2

def Dynamics_Reference(t, s, para, u, IfAcc = False):
    """
    返回仅有引力+J2摄动下参考的MEEdot动力学
    输入:
    - t: 时间(s)
    - s: s = [p,f,g,h,k,L,m] | 系统状态
    - para: para = [Cd,Area,c] | 系统动力学参数
    - u: u = [0,0,0] | RTN系下航天器推力加速度
    输出:
    - Sdot: Sdot = [pdot,fdot,gdot,hdot,kdot,Ldot,mdot] | 系统状态时间导数
    """

    p,f,g,h,k,L,m = s

    MEE = p,f,g,h,k,L

    Apn = __Acc_J2_MEE(MEE)
    MEEdot = MEE_Dynamics(MEE,Apn)
    pdot,fdot,gdot,hdot,kdot,Ldot = MEEdot
    mdot=0

    sdot=np.array([pdot,fdot,gdot,hdot,kdot,Ldot,mdot])

    if IfAcc:
        return sdot,Apn

    return sdot

def Dynamics_Perturbed(t, s, para, u, IfAcc = False):
    """
    返回引力+J2+阻力+推力的实际MEEdot动力学
    输入:
    - t: 时间(s)
    - s: s = [p,f,g,h,k,L,m] | 系统状态
    - para: para = [Cd,Area,c] | 系统动力学参数
    - u: u = [Tax,Tay,Taz] | RTN系下航天器推力加速度
    输出:
    - Sdot: Sdot = [pdot,fdot,gdot,hdot,kdot,Ldot,mdot] | 系统状态时间导数
    """

    p,f,g,h,k,L,m = s
    Cd,A,c = para

    MEE = p,f,g,h,k,L
    R,V = MEE2RV(MEE)

    Acc1 = __Acc_J2_MEE(MEE)
    Acc2 = __Acc_Drag_MEE(MEE,R,V,Cd,A,m)
    Acc3 = u

    Apn = Acc1 + Acc2
    Ap = Acc1 + Acc2 + Acc3

    MEEdot = MEE_Dynamics(MEE,Ap)

    pdot,fdot,gdot,hdot,kdot,Ldot = MEEdot
    mdot = - m * norm(u) *1000/c

    sdot=np.array([pdot,fdot,gdot,hdot,kdot,Ldot,mdot])

    if IfAcc:
        return sdot,Apn

    return sdot

def RK4_Step(dynamics, dt, t, s, para, u):
    
    """
    **描述 Description**:\n
    动力学龙格-库塔4次演算方法\n
    4th Order Runge-Kutta Method\n

    **输入 Input**:\n
         - dynamics: 动力学函数 | Dynamics | f(t, s, p, u)
         - dt: 时间步长 | Time step
         - t: 当前时间 | Current time
         - s: 当前状态向量 | Current system states
         - p: 参数向量 | System dynamics parameter
         - u: 输入向量 | Input control vector

    **输出 Output**:\n
         - s_next: 下一步的状态向量 | Next system states
         - s_dot: 当前系统状态时间导数 | Current system states derivative 
    """

    k1 = dynamics(t, s, para, u)
    k2 = dynamics(t + dt / 2, s + k1 * dt / 2, para, u)
    k3 = dynamics(t + dt / 2, s + k2 * dt / 2, para, u)
    k4 = dynamics(t + dt, s + k3 * dt, para, u)

    n = s + (k1 + 2 * k2 + 2 * k3 + k4) * dt / 6
    return n

def Next_Reference(dt,sat_param:Sat_StaticParam,Data_ref:Sat_DynamicParam):
     
    """输入当前标定卫星的状态, 返回下一个标定状态"""

    # 解包数据
    dt = dt
    t = Data_ref.t
    MEE = Data_ref.MEE
    m = Data_ref.m

    Cd = sat_param.Cd
    Area = sat_param.Area
    c = sat_param.c

    # 构建需要的输入数据并求解一步
    p = [Cd,Area,c]
    s = [*MEE,m]
    u = [0,0,0]
    s_next = RK4_Step(Dynamics_Reference,dt,t,s,p,u)

    # 对于L元素，范围处理
    s_next[5] %= (2*np.pi)

    # 得到下一步数据并返回
    MEE = np.array(s_next[0:6])
    m = s_next[6]
    R,V = MEE2RV(MEE)
    sdot,Apn = Dynamics_Reference(t+dt,s_next,p,u,IfAcc=True)
    MEEdot = sdot[0:6]

    Next_Data_ref = Sat_DynamicParam(t+dt,MEE,MEEdot,Apn,u,R,V,m)

    return Next_Data_ref

def Next_Perturbed(dt,sat_param:Sat_StaticParam,Data_real:Sat_DynamicParam,u):
    """输入当前实际卫星的状态, 返回下一个实际状态"""

    # 解包数据
    dt = dt
    t = Data_real.t
    MEE = Data_real.MEE
    m = Data_real.m

    Cd = sat_param.Cd
    Area = sat_param.Area
    c = sat_param.c

    # 构建需要的输入数据并求解一步
    p = [Cd,Area,c]
    s = [*MEE,m]
    s_next = RK4_Step(Dynamics_Perturbed,dt,t,s,p,u)

    # 对于L元素，范围处理
    s_next[5] %= (2*np.pi)
    
    # 得到下一步数据并返回
    MEE = np.array(s_next[0:6])
    m = s_next[6]
    R,V = MEE2RV(MEE)
    sdot,Apn = Dynamics_Perturbed(t+dt,s_next,p,u,IfAcc=True)
    MEEdot = sdot[0:6]
    Next_Data_real = Sat_DynamicParam(t+dt,MEE,MEEdot,Apn,u,R,V,m)

    return Next_Data_real

def Next_LVLH(Data_real:Sat_DynamicParam, Data_ref:Sat_DynamicParam):
    """
    给定下一个的实际和标定卫星动态数据, 计算LVLH坐标系中的相对位置与速度状态
    """

    R = Data_real.R
    V = Data_real.V
    R_ref = Data_ref.R
    V_ref = Data_ref.V

    Rdrift = R-R_ref
    Vdrift = V-V_ref

    # 漂移位置与速度定义相对于参考卫星
    R_LVLH = ECI2LVLH(R_ref,V_ref,Rdrift)
    V_LVLH = ECI2LVLH(R_ref,V_ref,Vdrift)
    
    U_LVLH = Data_real.U_LVLH

    # 返回LVLH坐标系下的状态数据
    Next_Data_LVLH = Sat_LVLHParam(R_LVLH, V_LVLH,U_LVLH)

    return Next_Data_LVLH