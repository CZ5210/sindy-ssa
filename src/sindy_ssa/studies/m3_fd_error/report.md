# 无噪声站位保持中未平滑 MEEFD 推力反演的有限差分误差

**Finite-Difference Error in Unsmoothed MEEFD Thrust Reconstruction for Noise-Free Station-Keeping**

## 摘要 / Abstract

**60 秒差分造成 2.70280261 m/s，10 秒差分造成 0.0754744082 m/s。** 这两个数值是无噪声站位保持案例中，有限差分本身引入的平面推力误差积分。将步长从 60 秒缩短到 10 秒后，该指标下降了 97.21%。本研究为仓库配套论文审稿意见 M3 提供补充数值证据。 [论文与项目背景](../../../../README.md)

**Finite differencing introduced 2.70280261 m/s of planar integrated thrust error at a 60 s step and 0.0754744082 m/s at a 10 s step.** The reduction was 97.21% in the noise-free station-keeping case. This supplementary numerical study addresses review comment M3 on the companion manuscript. [Manuscript and project overview](../../../../README.md)

## 数据与方法 / Data and methods

输入为 `data/stationkeeping/truth/Data_real.csv`，由站位保持动力学仿真生成。文件在 0–180,020 秒内含 18,003 个等间隔样本，采样间隔为 10 秒。唯一连续的 `Flag = manoeuvre` 区间为 56,280–95,810 秒，含 3,954 个样本。三组共用 **56,340–95,700 秒**内的 **657 个**评价时刻，间隔为 60 秒，含首尾。每个评价时刻及其前后各 60 秒均处于同一机动段，确保中心差分模板避开推力开关。 [站位保持配置](../../../../configs/config.yaml)

The simulated station-keeping trajectory in `data/stationkeeping/truth/Data_real.csv` contains 18,003 samples from 0 to 180,020 s at a uniform 10 s cadence. The single contiguous interval marked `Flag = manoeuvre` spans 56,280 to 95,810 s and contains 3,954 samples. All three reconstructions are evaluated at the same **657 timestamps from 56,340 to 95,700 s**, inclusive, at 60 s intervals. Each evaluation timestamp and its two neighbors at ±60 s lie within the same manoeuvre, so the centered stencil avoids thrust switching. [Station-keeping configuration](../../../../configs/config.yaml)

本研究使用修正春分点根数（MEE）变分方程的有限差分反演（MEEFD）。在共同评价时刻，三组使用相同的 MEE 状态和自然摄动加速度 `Apn_r/Apn_t/Apn_n`，并在 `ideal` 无噪声配置下采用普通最小二乘反演。三组仅改变总 MEE 变化率的来源。 [反演实现](../../sindy/meefd.py) [预处理实现](../../sindy/preprocessing.py)

MEEFD is the finite-difference inversion of the modified equinoctial element (MEE) variational equations. At each evaluation timestamp, all three runs use the same MEE state and natural perturbing acceleration `Apn_r/Apn_t/Apn_n`, with ordinary least-squares inversion under the noise-free `ideal` configuration. The only experimental variable is the source of the total MEE rates. [Inversion implementation](../../sindy/meefd.py) [Preprocessing implementation](../../sindy/preprocessing.py)

| 组别 / Run | 变化率来源 | Rate source |
| --- | --- | --- |
| FD60 | 从真值首行开始每 6 行取 1 行，在 60 秒网格上调用 `meefd_thrust()`。 | Keep every sixth truth row starting with the first, then call `meefd_thrust()` on the 60 s grid. |
| FD10 | 在原始 10 秒网格上调用 `meefd_thrust()`，随后按 `Time` 对齐到评价网格。 | Call `meefd_thrust()` on the native 10 s grid, then select the evaluation timestamps by `Time`. |
| Dynamics rates | 读取同一时刻的六个瞬时导数 `pdot` 至 `Ldot`，分别反演总加速度与自然加速度，再相减得到推力加速度。 | Read the six instantaneous rates `pdot` through `Ldot` at the evaluation timestamps, invert the total and natural accelerations separately, then subtract to recover the thrust acceleration. |

## 误差指标 / Error metric

以动力学导数组为参考，先计算每个时刻和各分量的推力加速度差，再对径向和沿轨分量的绝对值求和。三个组对 CSV 真实推力的误差积分采用同一口径，此时以真值分量为参考。

The dynamics-rate reconstruction serves as the reference for isolating finite-difference error. Compute the pointwise acceleration difference first, then sum the absolute radial and along-track errors. The truth-based error integrals use the same calculation with the CSV thrust acceleration as the reference.

$$
\Delta u_{\mathrm{FD},c}(t_i) = u_{\mathrm{FD},c}(t_i) - u_{\mathrm{dyn},c}(t_i), \qquad c \in \{r,t\},
$$

$$
EI_{\mathrm{FD},2D} = 1000 \times 60 \sum_{i=1}^{657}
  \left(\left|\Delta u_{\mathrm{FD},r}(t_i)\right| +
        \left|\Delta u_{\mathrm{FD},t}(t_i)\right|\right).
$$

每个评价样本的权重为 60 秒。因输入单位为 km/s²，乘 1,000 后积分结果以 m/s 表示。 [误差积分实现](../../metrics/evaluation.py)

Every evaluation sample receives a weight of 60 s. The factor of 1,000 converts the acceleration from km/s² to m/s², so the integral is expressed in m/s. [Error integral implementation](../../metrics/evaluation.py)

## 结果与核查 / Results and validation

前三列指标以 CSV 真值为参考，最后一列以动力学导数组为参考。所有数值的单位均为 m/s。

The first three metrics use the CSV thrust truth as the reference. The last uses the dynamics-rate reconstruction. All values are in m/s.

| 组别 / Run | `EI_r` | `EI_t` | `EI_2D` | `EI_FD_2D` |
| --- | ---: | ---: | ---: | ---: |
| FD60 | 2.30602747 | 0.396775194 | 2.70280266 | 2.70280261 |
| FD10 | 0.0641756002 | 0.0112988583 | 0.0754744585 | 0.0754744082 |
| 动力学导数 / Dynamics rates | 1.00231199e-6 | 1.31651639e-7 | 1.13396363e-6 | — |

步长从 60 秒缩短到 10 秒后，纯差分 `EI_FD_2D` 和对真值的 `EI_2D` 均下降约 2.627328 m/s（97.21%）。纯差分误差的径向积分从 2.30603 降至 0.06418 m/s，约占平面降幅的 85%。缩短中心差分步长可减小平滑机动段内的导数截断误差。本窗口内两组 `EI_FD_2D` 的实测比值为 35.81，接近二阶截断误差主项给出的步长平方比 36。 [完整数值](summary.csv)

Reducing the step from 60 to 10 s decreased both the finite-difference and truth-based planar integrals by approximately 2.627328 m/s (97.21%). The radial contribution to the finite-difference integral fell from 2.30603 to 0.06418 m/s, accounting for about 85% of the planar reduction. A shorter centered-difference step reduces derivative truncation error within the smooth manoeuvre arc. The measured ratio of the two planar finite-difference integrals was 35.81 on this window, close to the squared step-size ratio of 36 suggested by the leading second-order truncation term. [Full numerical results](summary.csv)

动力学导数组对真值的平面误差积分为 1.13e-6 m/s，径向、沿轨和法向的逐点最大绝对残差分别为 1.00e-13、1.18e-14 和 3.72e-11 km/s²。这些残差用于核查反演流程及输入保存精度。重算的 FD60 与已有无噪声归档 `data/stationkeeping/{typical,batch}/ideal/methods/MEEFD-SG/U_MEEFD_inv.csv` 分别在 659 个重叠时刻上核对，三分量最大绝对差均为 4.98e-16 km/s²，与 CSV 保存精度一致。

The dynamics-rate reconstruction gave a planar error integral of 1.13e-6 m/s against truth. The maximum pointwise residuals in the radial, along-track, and normal components were 1.00e-13, 1.18e-14, and 3.72e-11 km/s², respectively. These residuals check the inversion and the precision of the stored inputs. The recomputed FD60 output was also checked against the existing noise-free files at `data/stationkeeping/{typical,batch}/ideal/methods/MEEFD-SG/U_MEEFD_inv.csv`, each at 659 shared timestamps. Both comparisons had a maximum absolute discrepancy of 4.98e-16 km/s² across the three components, consistent with the CSV storage precision.

## 适用范围 / Scope

上述数值量化此无噪声案例中有限差分对未平滑 MEEFD 反演的整体影响。论文图 7 的误差积分对应经过平滑或回归的输出，应结合相应的后处理流程解读。

These values quantify the aggregate effect of finite differencing on unsmoothed MEEFD reconstruction in this noise-free case. The error integrals in Figure 7 of the companion manuscript refer to outputs after smoothing or regression and should be interpreted in the context of those processing steps.

## 复现 / Reproduction

在仓库根目录使用 Python 3.10 或更高版本，以可编辑模式安装项目。在没有真值 CSV 的检出中，用第二条命令按仓库配置生成站位保持轨迹。已有真值 CSV 时可直接运行第三条命令。 [真值生成实现](../../experiments/run_dynamics.py)

From the repository root, use Python 3.10 or later and install the project in editable mode. In a checkout without the truth CSV, use the second command to generate the station-keeping trajectory from the repository configuration. When the truth CSV is already available, proceed to the third command. [Truth generation implementation](../../experiments/run_dynamics.py)

```bash
python -m pip install -e .
python -c "from sindy_ssa.config import load_config; from sindy_ssa.experiments.run_dynamics import run_dynamics; run_dynamics(load_config('configs/config.yaml', scenario='stationkeeping'), 'data/stationkeeping/truth')"
python -m sindy_ssa.studies.m3_fd_error
```

研究脚本输出评价窗口和基准残差，并生成汇总 CSV。若本地已有无噪声反演归档，脚本也会按时间戳进行核对。验证时，在独立临时目录重新生成的真值 CSV 和汇总 CSV 均与报告所用文件逐字节一致。 [运行入口](__main__.py) [汇总 CSV](summary.csv)

The study prints the evaluation window and reference residuals and writes the summary CSV. It also checks the saved noise-free inversions by timestamp when the files are available locally. Both the truth CSV and the summary CSV regenerated in an isolated temporary directory matched the files used for this report byte for byte. [Study entry point](__main__.py) [Summary CSV](summary.csv)
