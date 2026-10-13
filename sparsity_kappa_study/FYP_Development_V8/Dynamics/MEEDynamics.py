import numpy as np

def MEE_Dynamics(MEE, Acc, mu = 398600):
    """MEE修正春分点根数摄动方程"""

    p,f,g,h,k,L = MEE
    Pr,Pt,Pn = Acc

    sinL=np.sin(L)
    cosL=np.cos(L)
    w = 1+f*cosL+g*sinL
    SQRT = np.sqrt(p/mu)
    s2 = 1+h*h+k*k

    pdot = 2*p/w*SQRT*Pt
    fdot = SQRT*(Pr*sinL + ((w+1)*cosL+f)*Pt/w - (h*sinL-k*cosL)*g*Pn/w)
    gdot = SQRT*(-Pr*cosL + ((w+1)*sinL+g)*Pt/w + (h*sinL-k*cosL)*g*Pn/w)
    hdot = SQRT*s2*Pn*cosL/(2*w)
    kdot = SQRT*s2*Pn*sinL/(2*w)
    Ldot = np.sqrt(mu*p)*(w/p)**2 + SQRT*(h*sinL-k*cosL)*Pn/w
    MEEdot = pdot,fdot,gdot,hdot,kdot,Ldot
    
    return MEEdot

def MEEdot2Acc(MEE,MEEdot):
    """ 反向MEE, Acc = A(MEE)^{-1}(MEEdot-B(MEE))"""

    mu=398600
    p,f,g,h,k,L = MEE
    sinL=np.sin(L)
    cosL=np.cos(L)
    w = 1+f*cosL+g*sinL
    SQRT = np.sqrt(p/mu)
    S = np.sqrt(1+h*h+k*k)

    A = np.array([
        [0         , (2*p/w)*SQRT         , 0                        ],
        [ SQRT*sinL, SQRT/w*((w+1)*cosL+f), -SQRT*g/w*(h*sinL-k*cosL)],
        [-SQRT*cosL, SQRT/w*((w+1)*sinL+g)  ,SQRT*f/w*(h*sinL-k*cosL)],
        [0         , 0                    , SQRT*S*S*cosL/w/2        ],
        [0         , 0                    , SQRT*S*S*sinL/w/2        ],
        [0         , 0                    , SQRT*(h*sinL-k*cosL)/w     ]
    ])

    b = np.array([0,0,0,0,0,np.sqrt(mu*p)*(w*w/p/p)])

    # 使用伪逆求解
    # P = np.dot(np.linalg.pinv(A),(MEEdot-b))
    # Acc = P.flatten()

    # 使用最小二乘不带正则化
    A = np.array(A, dtype=np.float64)
    MEEdot = np.asarray(MEEdot, dtype=np.float64)
    P = np.linalg.lstsq(A, MEEdot - b, rcond=None)[0]

    # # 使用最小二乘法求解添加正则化项以提高数值稳定性
    # A = np.array(A, dtype=np.float64)
    # MEEdot = np.asarray(MEEdot, dtype=np.float64)
    # lambda_reg = 1e-10
    # AtA = A.T @ A  # 3x3
    # Atb = A.T @ (MEEdot - b)  # 3x1
    # A_reg = AtA + lambda_reg * np.eye(3)
    # P = np.linalg.solve(A_reg, Atb)

    Acc = P.flatten()

    return Acc
