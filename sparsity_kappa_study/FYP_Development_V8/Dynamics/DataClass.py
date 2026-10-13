class Time_Param():
    """时间参数类"""
    def __init__(self,hour,dt):
        self.tspan = (0,3600 * hour)
        self.n = (3600 * hour) // dt
        self.dt = dt
        self.hour = hour 
        self.PointPerHour = 3600 // dt
    
    @classmethod
    def from_dict(cls, data):
        """从字典创建Time_Param对象"""
        return cls(hour=data['hour'], dt=data['dt'])

class Sat_StaticParam():
    """卫星静态参数类"""
    def __init__(self,Cd,Area,c,m0,m,Tmax):
        self.Cd = Cd
        """ 阻力系数 | Drag Coefficient"""
        self.Area = Area
        """ 迎风面积 | Frontal Area"""
        self.c = c
        """ 比冲 | Specific Impulse"""
        self.m0 = m0
        """ 空质量 | Empty Mass"""
        self.m = m
        """ 初始质量 | Initial Mass"""
        self.Tmax = Tmax
        """ 最大推力(N) | Maximum Thrust"""

    def __repr__(self):
        return f"Sat_StaticParam(Cd={self.Cd}, Area={self.Area}, c={self.c}, m0={self.m0}, m={self.m}, Tmax={self.Tmax}, BoxSize={self.BoxSize})"

class Sat_DynamicParam():
    """卫星的动态数据类"""
    def __init__(self,t,MEE,MEEdot,Apn,U_LVLH,R,V,m):
        self.t = t
        self.MEE = MEE
        self.MEEdot = MEEdot
        self.Apn = Apn
        self.U_LVLH = U_LVLH
        self.R = R
        self.V = V
        self.m = m
        
class Sat_LVLHParam():
    """卫星的LVLH数据类"""
    def __init__(self,R_LVLH,V_LVLH,U_LVLH):
        self.R = R_LVLH
        """ LVLH坐标系下的位置"""
        self.V = V_LVLH
        """ LVLH坐标系下的速度"""
        self.U = U_LVLH
        """ LVLH坐标系下的推力(U)"""
        