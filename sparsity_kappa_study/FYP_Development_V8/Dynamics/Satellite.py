from .DataClass import *
from .DynamicsCore import *
from .DynamicsTool import *
from .MEEDynamics import *
from SINDy.Sequence import *
from types import SimpleNamespace
import csv,os

class BaseSatellite():
    """ 卫星类 """
    
    def __init__(self,Sat_Param:Sat_StaticParam,R0,V0,m):
        
        """初始化卫星类"""

        # 卫星ID和静态参数
        self.Param = Sat_Param
        R0 = np.array(R0)
        V0 = np.array(V0)
        MEE0 = RV2MEE(R0,V0)

        # 初始条件
        t0 = 0
        s0 = [*MEE0,m]

        p = [self.Param.Cd,self.Param.Area,self.Param.c]
        u0 = [0,0,0]

        sdot_ref,Apn_ref = Dynamics_Reference(t0,s0,p,u0,IfAcc=True)
        sdot_real,Apn_real = Dynamics_Perturbed(t0,s0,p,u0,IfAcc=True)
        MEEdot_ref = sdot_ref[0:6]
        MEEdot_real = sdot_real[0:6]

        # 初始化存储信息
        self.Data_ref = Sat_DynamicParam(t0,MEE0,MEEdot_ref,Apn_ref,u0,R0,V0,m)
        self.Data_ref_list=[self.Data_ref]

        self.Data_real = Sat_DynamicParam(t0,MEE0,MEEdot_real,Apn_real,u0,R0,V0,m)
        self.Data_real_list = [self.Data_real]

        # 计算相对状态信息
        self.Data_LVLH = Next_LVLH(self.Data_real,self.Data_ref)
        self.Data_LVLH_list = [self.Data_LVLH]

        self.Flag_list = []

        self.Manoeuvre = SimpleNamespace()
        self.Manoeuvre.freq = 0
        self.Manoeuvre.start_time = []
        self.Manoeuvre.TOF = []
        self.Manoeuvre.DV = []

        return
    
    def update(self,time_param:Time_Param, u_LVLH):
        """卫星类状态更新函数"""
        
        # 基本轨道预报
        Data_ref_new = Next_Reference(time_param.dt,self.Param,self.Data_ref)
        Data_real_new = Next_Perturbed(time_param.dt,self.Param,self.Data_real,u = u_LVLH)
        Data_LVLH_new = Next_LVLH(self.Data_real,self.Data_ref)

        self.Data_ref = Data_ref_new
        self.Data_real = Data_real_new
        self.Data_LVLH = Data_LVLH_new

        self.Data_ref_list.append(self.Data_ref)
        self.Data_real_list.append(self.Data_real)
        self.Data_LVLH_list.append(self.Data_LVLH)
        
        self.Flag_list.append(self.Flag)

        return

    def Check_Conditions(self, t):
        """检查终止条件"""
        Alt = norm(self.Data_real.R) - 6371
        if Alt <= 0:
            print(f"[终止] 时间: {t:.1f}秒 | 卫星撞击地球")
            return True
        if self.Data_real.m < self.Param.m0:
            print(f"[终止] 时间: {t:.1f}秒 | 燃料耗尽")
            return True
        return False
    
    def Repropagate_Orbit(self,interval,ThrustPath,dt):
        """重新推演轨道函数"""
        # 在给定的区间内，获取同样的初始状态和重建推力重新推演动力学

        def _initRealRef(time0):
            # 提取Real的初始数据
            real_scalar_params = ['Time', 'Mass']  # 标量参数
            real_mee_params = ['p', 'f', 'g', 'h', 'k', 'L']  # MEE轨道要素
            real_meedot_params = ['pdot', 'fdot', 'gdot', 'hdot', 'kdot', 'Ldot']  # 轨道要素导数
            real_apn_params = ['Apn_r', 'Apn_t', 'Apn_n']  # 自然摄动加速度
            real_u_params = ['u_r', 'u_t', 'u_n']  # 推力
            real_r_params = ['R_x', 'R_y', 'R_z']  # 位置矢量
            real_v_params = ['V_x', 'V_y', 'V_z']  # 速度矢量
            
            Real_Data_full = SequenceData.from_csv(RealPath)

            # 找到最接近time0的数据点
            real_times = np.array(Real_Data_full.get('Time').Data)
            # 找到最接近time0的索引
            idx_real = np.argmin(np.abs(real_times - time0))

            # 使用最接近的索引来提取参数
            real_params = Real_Data_full.extract_params(
                real_scalar_params + real_mee_params + real_meedot_params +
                real_apn_params + real_u_params + real_r_params + real_v_params,
                index=idx_real
            )

            # 确保使用time0作为初始时间，而不是原始数据的时间
            MEE0 = np.array([real_params[name] for name in real_mee_params])
            MEEdot_real = np.array([real_params[name] for name in real_meedot_params])
            Apn_real = np.array([real_params[name] for name in real_apn_params])
            u0 = np.array([real_params[name] for name in real_u_params])
            R0 = np.array([real_params[name] for name in real_r_params])
            V0 = np.array([real_params[name] for name in real_v_params])
            t0 = time0
            m0 = real_params['Mass']

            # 初始化实际轨道状态
            self.Data_real = Sat_DynamicParam(
                t=t0, MEE=MEE0, MEEdot=MEEdot_real, Apn=Apn_real,
                U_LVLH=u0, R=R0, V=V0, m=m0
            )
            
            # 提取Ref的初始数据
            ref_mee_params = ['p', 'f', 'g', 'h', 'k', 'L']
            ref_meedot_params = ['pdot', 'fdot', 'gdot', 'hdot', 'kdot', 'Ldot']
            ref_apn_params = ['Apn_r', 'Apn_t', 'Apn_n']
            ref_r_params = ['R_x', 'R_y', 'R_z']
            ref_v_params = ['V_x', 'V_y', 'V_z']

            Ref_Data_full = SequenceData.from_csv(RefPath)

            # 找到最接近time0的数据点
            ref_times = np.array(Ref_Data_full.get('Time').Data)
            # 找到最接近time0的索引
            idx_ref = np.argmin(np.abs(ref_times - time0))

            ref_params = Ref_Data_full.extract_params(
                ref_mee_params + ref_meedot_params +
                ref_apn_params + ref_r_params + ref_v_params,
                index=idx_ref
            )

            # 组合为向量参数
            MEE0_ref = np.array([ref_params[name] for name in ref_mee_params])
            MEEdot_ref = np.array([ref_params[name] for name in ref_meedot_params])
            Apn_ref = np.array([ref_params[name] for name in ref_apn_params])
            R0_ref = np.array([ref_params[name] for name in ref_r_params])
            V0_ref = np.array([ref_params[name] for name in ref_v_params])

            # 初始化参考轨道状态
            self.Data_ref = Sat_DynamicParam(
                t=t0, MEE=MEE0_ref, MEEdot=MEEdot_ref, Apn=Apn_ref,
                U_LVLH=np.array([0, 0, 0]), R=R0_ref, V=V0_ref, m=m0
            )

            self.Data_LVLH = Next_LVLH(self.Data_real, self.Data_ref)

            return idx_real

        # 生成对应路径
        RealPath = './Output/Data_real.csv'
        RefPath = './Output/Data_ref.csv'

        # 取出区间和对应的重建加速度
        start_idx,end_idx = interval
        U_rec = SequenceData.from_csv(ThrustPath)
        time0 = U_rec.get('Time').Data[0]
        u_r_list = np.array(U_rec.get('u_r').Data)
        u_t_list = np.array(U_rec.get('u_t').Data)
        u_n_list = np.array(U_rec.get('u_n').Data)

        # 获取重建推力的总时间步数
        thrust_steps = len(u_r_list)
        
        # 计算推演的时间区间（基于重建推力数据）
        t_start = time0
        t_end = time0 + (thrust_steps - 1) * dt
        
        print(f"[重新推演] 对齐时间：{t_start:.2f}s, 推演时长：{(t_end-t_start):.2f}s, 步数：{thrust_steps}")

        # 提取初始化时刻状态
        aligned_idx = _initRealRef(time0)

        # 重新推演数据存储
        self.Data_reprop_real_list = [self.Data_real]  # 重新推演的实际轨道
        self.Data_reprop_ref_list = [self.Data_ref]    # 重新推演的参考轨道
        self.Data_reprop_LVLH_list = [self.Data_LVLH]  # 重新推演的LVLH相对状态

        for step in range(thrust_steps - 1):
            
            # 获取当前步的重建推力
            # u_LVLH = [u_r_list[step], u_t_list[step], u_n_list[step]]
            u_LVLH = [u_r_list[step], u_t_list[step], 0]
            
            # 更新轨道状态（使用重建推力）
            Data_ref_new = Next_Reference(dt, self.Param, self.Data_ref)
            Data_real_new = Next_Perturbed(dt, self.Param, self.Data_real, u=u_LVLH)
            Data_LVLH_new = Next_LVLH(Data_real_new, Data_ref_new)
            
            # 更新当前状态
            self.Data_real = Data_real_new
            self.Data_ref = Data_ref_new
            self.Data_LVLH = Data_LVLH_new
            
            # 存储重新推演的结果
            self.Data_reprop_real_list.append(Data_real_new)
            self.Data_reprop_ref_list.append(Data_ref_new)
            self.Data_reprop_LVLH_list.append(Data_LVLH_new)
        
        print(f"[重新推演] 完成区间：{start_idx}~{end_idx}重新推演")
                
        return
    
    def Save_Repropagate_CSV(self, Dir, precision=8):
        """保存重新推演的轨道数据"""
        os.makedirs(Dir, exist_ok=True)
        reprop_csv_path = os.path.join(Dir, "Data_rereal.csv")
        lvlh_csv_path = os.path.join(Dir, "Data_reLVLH.csv")
        
        # 列标题
        headers = ['Time',  
                'p', 'f', 'g', 'h', 'k', 'L', 
                'pdot', 'fdot', 'gdot', 'hdot', 'kdot', 'Ldot',
                'Apn_r','Apn_t','Apn_n',
                'R_x', 'R_y', 'R_z', 'V_x', 'V_y', 'V_z', 
                'Mass','u_r','u_t','u_n']
        headers_lvlh = ['Time', 'R_x', 'R_y', 'R_z', 'V_x', 'V_y', 'V_z', 
                        'u_x', 'u_y', 'u_z']
        
        float_format = f"{{:.{precision}e}}"

        with open(reprop_csv_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            for idx, data in enumerate(self.Data_reprop_real_list):
                writer.writerow([
                    data.t,
                    *[float_format.format(v) for v in data.MEE],
                    *[float_format.format(v) for v in data.MEEdot],
                    *[float_format.format(v) for v in data.Apn],
                    *[float_format.format(v) for v in data.R],
                    *[float_format.format(v) for v in data.V],
                    float_format.format(data.m),
                    *[float_format.format(v) for v in data.U_LVLH],
                ])

        # 写入 Data_LVLH_list
        with open(lvlh_csv_path, mode='w', newline='') as lvlh_csv:
            writer = csv.writer(lvlh_csv)
            writer.writerow(headers_lvlh)  # 写入标题
            for idx, lvlh_data in enumerate(self.Data_reprop_LVLH_list):
                current_time = self.Data_reprop_real_list[idx].t
                writer.writerow([
                    current_time,
                    *[float_format.format(v) for v in lvlh_data.R],  
                    *[float_format.format(v) for v in lvlh_data.V],
                    *[float_format.format(v) for v in lvlh_data.U],
                ])
        print(f"[重新推演] 数据保存到{reprop_csv_path}")
        print(f"[重新推演] 数据保存到{lvlh_csv_path}")

    def Save_CSV(self, Output_Dir, precision =8):
        """
        将 Data_real_list、Data_ref_list 和 Data_LVLH_list 写入 CSV 文件
        :param output_dir: 输出文件的统一目录路径
        """

        # 确保输出目录存在
        os.makedirs(Output_Dir, exist_ok=True)

        # 生成文件路径
        real_csv_path = os.path.join(Output_Dir, "Data_real.csv")
        ref_csv_path = os.path.join(Output_Dir, "Data_ref.csv")
        lvlh_csv_path = os.path.join(Output_Dir, "Data_LVLH.csv")
        mano_csv_path = os.path.join(Output_Dir, "Data_mano.csv")

        # 定义列标题
        headers_real = ['Time',  
                        'p', 'f', 'g', 'h', 'k', 'L', 
                        'pdot', 'fdot', 'gdot', 'hdot', 'kdot', 'Ldot',
                        'Apn_r','Apn_t','Apn_n',
                        'R_x', 'R_y', 'R_z', 'V_x', 'V_y', 'V_z', 
                        'Mass','u_r','u_t','u_n','Flag']
        headers_ref = headers_real[:-5]
        headers_lvlh = ['Time', 'R_x', 'R_y', 'R_z', 'V_x', 'V_y', 'V_z', 
                        'u_x', 'u_y', 'u_z', 'Flag']
        headers_mano = ['Frequency','StartTime','TOF','DV']

        float_format = f"{{:.{precision}e}}"

        # 写入 Data_real_list
        with open(real_csv_path, mode='w', newline='') as real_csv:
            writer = csv.writer(real_csv)
            writer.writerow(headers_real)  # 写入标题
            for idx, data in enumerate(self.Data_real_list):
                flag = self.Flag_list[idx] if idx < len(self.Flag_list) else "unknown"
                writer.writerow([
                    data.t,
                    *[float_format.format(v) for v in data.MEE],
                    *[float_format.format(v) for v in data.MEEdot],
                    *[float_format.format(v) for v in data.Apn],
                    *[float_format.format(v) for v in data.R],
                    *[float_format.format(v) for v in data.V],
                    float_format.format(data.m),
                    *[float_format.format(v) for v in data.U_LVLH],
                    flag
                ])

        # 写入 Data_ref_list
        with open(ref_csv_path, mode='w', newline='') as ref_csv:
            writer = csv.writer(ref_csv)
            writer.writerow(headers_ref)  # 写入标题
            for data in self.Data_ref_list:
                writer.writerow([
                    data.t,
                    *[float_format.format(v) for v in data.MEE],
                    *[float_format.format(v) for v in data.MEEdot],
                    *[float_format.format(v) for v in data.Apn],
                    *[float_format.format(v) for v in data.R],
                    *[float_format.format(v) for v in data.V]
                ])

        # 写入 Data_LVLH_list
        with open(lvlh_csv_path, mode='w', newline='') as lvlh_csv:
            writer = csv.writer(lvlh_csv)
            writer.writerow(headers_lvlh)  # 写入标题
            for idx, lvlh_data in enumerate(self.Data_LVLH_list):
                # 使用 Data_ref_list 或 Data_real_list 的时间数据
                current_time = self.Data_ref_list[idx].t
                flag = self.Flag_list[idx] if idx < len(self.Flag_list) else "unknown"
                writer.writerow([
                    current_time,
                    *[float_format.format(v) for v in lvlh_data.R],  
                    *[float_format.format(v) for v in lvlh_data.V],
                    *[float_format.format(v) for v in lvlh_data.U],
                    flag
                ])

        # 写入 Manoeuvre 数据
        with open(mano_csv_path, mode='w', newline='') as mano_csv:
            writer = csv.writer(mano_csv)
            writer.writerow(headers_mano)  # 写入标题
            for freq, (start_time, tof, dv) in enumerate(zip(self.Manoeuvre.start_time, self.Manoeuvre.TOF, self.Manoeuvre.DV), start=1):
                writer.writerow([
                    freq,  
                    float_format.format(start_time),  
                    float_format.format(tof),  
                    float_format.format(dv)
                ])

        print(f"[保存成功] ECI实际数据保存到 {real_csv_path}")
        print(f"[保存成功] ECI参考数据保存到 {ref_csv_path}")
        print(f"[保存成功] LVLH相对运动保存到 {lvlh_csv_path}")
        print(f"[保存成功] 轨道机动数据保存到 {mano_csv_path}")

class SatelliteStationKeeping (BaseSatellite):
    """基本轨道维持场景"""

    def __init__(self, Sat_Param, R0, V0, m, BoxSize):
        super().__init__(Sat_Param, R0, V0, m)
        self.BoxSize = BoxSize
        self.Flag = "normal"
        self.Flag_list = [self.Flag]

    def __Normal(self,t,time_param:Time_Param):
        """正常轨道预报"""
        
        # 更新状态
        self.update(time_param,[0,0,0])

        # 离开控制区判定
        R_LVLH = self.Data_LVLH.R
        if Outside_Box(R_LVLH, self.BoxSize,shift = 0):
            print(f"[事件] 时间: {t:.1f}秒 | 卫星离开控制区，准备机动")
            self.Flag = 'stationkeeping'
        return

    def __Generate_Manoeuvre_LQR(self):
        """初始化LQR机动数据"""

        self.terminalpos_error = 2
        self.terminalvel_error = 1
        r = np.linalg.norm(self.Data_real.R)
        A, B = CWEquation(r)
        Q = np.diag([10, 1, 0.01, 2, 3, 2])
        R = np.diag([2,4,2])
        self.K1 = ComputeLQRControl(A, B, Q, R)

        self.Flag = "manoeuvre"  # 切换到机动状态
        self.Manoeuvre.start_time.append(self.Data_real.t)  # 记录机动开始时间
        self.thisDV = 0  # 初始化机动增量

        return
    
    def __Manoeuvre_LQR(self,t, time_param:Time_Param):
        """LQR控制律执行机动: 实时计算控制输入并更新轨道"""
        
        # 获取当前LVLH状态
        S_LVLH = self.Data_LVLH_list[-1]  # 最新状态：[x, y, z, vx, vy, vz]
        s = [*S_LVLH.R,*S_LVLH.V]

        # 基于时间窗口的状态平均
        time_window = 3000  # 时间窗口（秒）
        dt = time_param.dt   # 时间步长（秒）
        max_iter = 200       # 最大迭代次数（避免循环过多，可根据需求调整）
        nds = np.zeros(6)
        count = 0

        # 步骤1：计算理论上覆盖时间窗口需要的步数（i_theo步的总时间 ≤ time_window）
        if dt > 0:  # 避免dt=0的异常
            i_theo = int(np.floor(time_window / dt))  # 理论步数（向下取整）
        else:
            i_theo = 0  # dt=0时不遍历历史

        # 步骤2：确定实际最大步数（不能超过历史数据长度，且不超过理论步数）
        # 历史数据长度：len(self.Data_LVLH_list)，当前状态是最后一个，往前最多取len-1步
        max_possible_steps = len(self.Data_LVLH_list) - 1  # 避免越界（i从0开始）
        i_max = min(i_theo, max_possible_steps)

        # 步骤3：自适应取样步长（如果i_max超过max_iter，通过步长减少循环次数）
        if i_max > max_iter:
            step = max(1, i_max // max_iter)  # 步长至少为1，确保覆盖窗口
        else:
            step = 1  # 步数较少时，逐个取样

        # 步骤4：按步长遍历历史状态（从最近的开始，间隔step步取样）
        # i的范围：0, step, 2*step, ..., ≤i_max（确保覆盖时间窗口）
        for i in range(0, i_max + 1, step):
            hist_state = self.Data_LVLH_list[-i - 1]  # 取第i步前的状态（i=0是当前前一步）
            # 累加状态（无需计算hist_time，因为i*dt已通过i_theo限制在窗口内）
            nds[:3] += hist_state.R
            nds[3:] += hist_state.V
            count += 1

        # 计算平均值（处理边界情况）
        if count > 0:
            nds /= count
        else:
            nds = np.array([*S_LVLH.R, *S_LVLH.V])  # 无历史数据时用当前状态


        # 尝试无量纲化到一个级别
        nds[0:3] = nds[0:3]/15
        nds[3] = nds[3]*100
        nds[4] = nds[4]*1000
        nds[5] = -nds[5]*10000

        # 提取误差状态
        pos_error = np.linalg.norm(s[0:3])
        vel_error = np.linalg.norm(s[3:6])
        
        # 计算LVLH坐标系下的控制输入 u = -K * s，并限制推力
        max_thrust = (self.Param.Tmax/self.Param.m/1000)  # km/s²
        K = self.K1
        u_LVLH = K @ nds * 10**-8
        u_LVLH = np.clip(u_LVLH,-max_thrust,max_thrust)
        u_LVLH[2]=0
        self.thisDV += np.linalg.norm(u_LVLH) * time_param.dt * 1000  # 累计机动增量
        
        # 更新状态
        self.update(time_param,u_LVLH)
        
        # 检查终止条件：位置偏差小于阈值时结束机动
        if pos_error < self.terminalpos_error and vel_error*1000 < self.terminalvel_error:
            print(f"[LQR机动结束] 时间: {t:.1f}秒 | 位置偏差: {pos_error:.2f}km | 速度偏差: {vel_error*1000:.2f}m/s")
            self.Flag = "normal"  # 切换回正常模式
            self.Manoeuvre.freq += 1  # 记录机动次数
            self.Manoeuvre.TOF.append(t - self.Manoeuvre.start_time[-1])
            self.Manoeuvre.DV.append(self.thisDV)

        return

    def Propagate_Orbit(self,time_param:Time_Param):
        """推演轨道函数"""
        
        for i in range(0,time_param.n+1):
            
            t = self.Data_ref.t
            
            # 检查末端状态
            if self.Check_Conditions(t):
                break

            if self.Flag =='normal':
                self.__Normal(t, time_param)

            if self.Flag == 'stationkeeping':
                self.__Generate_Manoeuvre_LQR()
             
            if self.Flag == "manoeuvre":
                self.__Manoeuvre_LQR(t, time_param)
                
        return

class SatelliteCollisionAvoidance(BaseSatellite):
    """紧急避障场景"""

    def __init__(self, Sat_Param, R0, V0, m, TargetPos,TOF):
        super().__init__(Sat_Param, R0, V0, m)
        self.TargetPos = TargetPos
        self.TOF = TOF
        self.Flag = "ConjunctionDetected"

    def __Normal(self,t,time_param:Time_Param):
        """正常轨道预报"""
        
        # 更新状态
        self.update(time_param,[0,0,0])

        return

    def __generate_Ref(self,time_param:Time_Param):
        """生成一组参考弹道"""
        n = 2 #多少个半椭圆
        a = (abs(self.TargetPos[1]))/n #椭圆长轴
        b = 4 #椭圆短轴
        len = int(3600*self.TOF//time_param.dt) #s 单位
        # hold_time = 3600*1//time_param.dt #最后静止时间
        # len = len - hold_time
        x = np.zeros(len)
        y = np.zeros(len)
        z = np.zeros(len)
        
        for i in range(n):
            start_idx = i * (len // n)
            end_idx = (i + 1) * (len // n)
            theta = np.linspace(0, np.pi, end_idx - start_idx)
            x[start_idx:end_idx] = (b / 2) * np.sin(theta)
            y[start_idx:end_idx] = (a / 2) * np.cos(theta) - (a / 2) - (a * i)

        # # 最后添加一段时间的静止，确保到达目标位置
        # x = np.concatenate([x, np.full(hold_time, x[-1])])
        # y = np.concatenate([y, np.full(hold_time, y[-1])])
        # z = np.concatenate([z, np.full(hold_time, z[-1])])

        # # 最后添加反向半椭圆，回到初始位置
        # x = np.concatenate([x, x[::-1]])
        # y = np.concatenate([y, y[::-1]])
        # z = np.concatenate([z, z[::-1]])

        self.RefPos = np.vstack([x,y,z]).T

        # 保存数据
        X = Sequence(x,'x')
        Y = Sequence(y,'y')
        Z = Sequence(z,'z')
        ReferData = SequenceData([X,Y,Z])
        ReferData.save('./Output/ReferenceTrajectory.csv',precision=8)

        return 
        

    def __Generate_Manoeuvre_LQR(self,time_param:Time_Param):
        """初始化LQR机动数据"""

        # 生成一组参考弹道
        self.__generate_Ref(time_param)

        # 设定LQR参数
        self.terminalpos_error = 1
        self.terminalvel_error = 30
        r = np.linalg.norm(self.Data_real.R)
        A, B = CWEquation(r)
        Q = np.diag([1e2, 1e2, 1, 1e3, 5e7, 1])
        R = np.diag([2,2,2])
        self.K1 = ComputeLQRControl(A, B, Q, R)

        # Q = np.diag([1, 1, 0.01, 2e20, 2e20, 2e20])
        # R = np.diag([2,2,2])
        # self.K2 = ComputeLQRControl(A, B, Q, R)

        self.Flag = "manoeuvre"  # 切换到机动状态
        self.Manoeuvre.start_time.append(self.Data_real.t)  # 记录机动开始时间
        self.thisDV = 0  # 初始化机动增量

        return
    
    def __Manoeuvre_LQR(self,t, time_param:Time_Param):
        """LQR控制律执行机动: 实时计算控制输入并更新轨道"""

        idx = int(t / time_param.dt)
        RefPos = self.RefPos[idx-1]
        
        # 获取当前LVLH状态
        S_LVLH = self.Data_LVLH_list[-1]  # 最新状态：[x, y, z, vx, vy, vz]
        # PosErr = self.TargetPos-S_LVLH.R
        PosErr = RefPos-S_LVLH.R
        VelErr = S_LVLH.V
        s = [*PosErr,*VelErr]
        pos_error = np.linalg.norm(PosErr)
        vel_error = np.linalg.norm(VelErr)

        # 计算LVLH坐标系下的期望控制输入 u = -K * s
        # if pos_error>self.terminalpos_error:
        K = self.K1
        u_LVLH = - K @ s

        # 期望控制输入转化到控制角度
        ux_desired = u_LVLH[0]
        uy_desired = u_LVLH[1]
        theta = np.arctan2(ux_desired, uy_desired)

        # 根据期望输入，计算实际三轴推力
        max_thrust = (self.Param.Tmax/self.Param.m/1000)# km/s²
        ux = max_thrust * np.sin(theta)
        uy = max_thrust * np.cos(theta)
        u_LVLH = np.array([ux, uy, 0])
            
        # else:
        #     # 刹车阶段
        #     print(abs(VelErr))
        #     # print(norm(VelErr))
        #     # K = self.K2
        #     # u_LVLH = - K @ s
        #     scale = 1
        #     max_thrust = (self.Param.Tmax/self.Param.m/1000)# km/s²
        #     u_LVLH = np.array([0,-1,0]) * max_thrust
        #     u_LVLH[2]=0
        
        self.thisDV += np.linalg.norm(u_LVLH) * time_param.dt * 1000  # 累计机动增量
        
        # 更新状态
        self.update(time_param,u_LVLH)
        
        # 检查终止条件
        # print(pos_error,vel_error)
        # if pos_error < self.terminalpos_error and vel_error*1000 < self.terminalvel_error:

        if t>=self.TOF*3600:
            print(f"[LQR机动结束] 时间: {t:.1f}秒 | 位置偏差: {pos_error:.2f}km | 速度偏差: {vel_error*1000:.2f}m/s")
            self.Flag = "normal"  # 切换回正常模式
            self.Manoeuvre.freq += 1  # 记录机动次数
            self.Manoeuvre.TOF.append(t - self.Manoeuvre.start_time[-1])
            self.Manoeuvre.DV.append(self.thisDV)

        return

    def Propagate_Orbit(self,time_param:Time_Param):
        """推演轨道函数"""
        
        for i in range(0,time_param.n+1):
            
            t = self.Data_ref.t
            
            if self.Check_Conditions(t):
                break

            if self.Flag =='normal':
                self.__Normal(t, time_param)

            if self.Flag == 'ConjunctionDetected':
                self.__Generate_Manoeuvre_LQR(time_param)
             
            if self.Flag == "manoeuvre":
                self.__Manoeuvre_LQR(t, time_param)
                
        return
    