# OpenSim Python 学习路线

> 学完 `tutorial_opensim_python.py`（9 章：模型构建、加载检查、NumPy 互转、数据 I/O、仿真后分析、I/O 连接、Moco 基础、肌肉优化、关节速查）之后的进阶内容。

---

## 一、Python 进阶脚本（直接衔接教程）

| 主题 | 文件 | 学什么 |
|------|------|--------|
| 模型构建 | `Code/Python/build_simple_arm_model.py` | 独立的手臂模型构建示例 |
| NumPy 互转 | `Code/Python/numpy_conversions.py` | 更完整的数据转换用例 |
| 类扩展 | `Code/Python/extend_OpenSim_Vec3_class.py` | 如何扩展 OpenSim 类 |
| 轨迹分析 | `Code/Python/posthoc_StatesTrajectory_example.py` | 后处理轨迹结果 |
| Reporter | `Code/Python/wiring_inputs_and_outputs_with_TableReporter.py` | TableReporter 数据记录 |

---

## 二、Moco 轨迹优化（核心进阶方向）

按难度递进：

1. **入门** — `Code/Python/Moco/exampleSlidingMass.py`：无摩擦滑块最优控制
2. **肌肉动力学** — `Code/Python/Moco/exampleHangingMuscle.py`：肌肉悬挂优化
3. **参数优化** — `Code/Python/Moco/exampleOptimizeMass.py`：优化质量参数
4. **运动学约束** — `Code/Python/Moco/exampleKinematicConstraints.py`：添加运动学约束
5. **预测与追踪** — `Code/Python/Moco/examplePredictAndTrack.py`：双摆预测与追踪结合
6. **EMG 追踪** — `Code/Python/Moco/exampleEMGTracking/`：基于肌电信号的追踪
7. **深蹲起立** — `Code/Python/Moco/exampleSquatToStand/`：力矩驱动→肌肉驱动完整流程（含学生版和答案版）
8. **3D 步行** — `Code/Python/Moco/example3DWalking/`：80 块肌肉的 MocoInverse 和 MocoTrack

### Moco 求解原理速记

核心不是反复前向仿真试控制，而是把整条轨迹一次性优化。

1. 先定义最优控制问题：状态、控制、动力学方程、边界条件、目标函数。
2. 将时间区间离散为很多 `mesh intervals`。
3. 把每个时间点上的状态、控制、终止时间都当成优化变量。
4. 用 `direct collocation` 把微分方程转成代数约束，常见转录是 `Trapezoidal` 或 `Hermite-Simpson`。
5. 这样会得到一个稀疏非线性规划 `NLP`，再交给 `CasADi` / `Tropter` 后端和 `IPOPT` 这类优化器求解。
6. 输出结果是整条最优轨迹 `states(t)` 和 `controls(t)`；通常保存为 `.sto`，问题定义可保存为 `.omoco`。

对教程第 8 章（滑块）：

- 目标：让滑块从 `x=0, v=0` 到 `x=1, v=0`，并最小化总时间。
- 典型最优控制：先最大加速，再最大减速，也就是 `bang-bang control`。

### DGF2016 肌肉参数速记

`DeGrooteFregly2016Muscle` 是 Hill 型肌肉-肌腱模型。核心几何和力学关系：

```text
l_MT = l_t + l_f cos(alpha)
l_f_tilde = l_f / l_opt
l_t_tilde = l_t / l_slack
v_f_tilde = v_f / (v_max * l_opt)

F_f = F0 * [a * f_L(l_f_tilde) * f_V(v_f_tilde) + f_PE(l_f_tilde) + beta * v_f_tilde]
F_t = F0 * f_T(l_t_tilde)
F_t = F_f * cos(alpha)
```

其中：

- `max_isometric_force = F0`：最大等长力，决定整块肌肉的力尺度。
- `optimal_fiber_length = l_opt`：最佳纤维长度，用于归一化纤维长度，并决定主动长度曲线峰值位置。
- `tendon_slack_length = l_slack`：肌腱松弛长度，决定肌腱何时开始明显受力。
- `tendon_strain_at_one_norm_force = eps_t1`：满足 `f_T(1 + eps_t1) = 1`；值越大，肌腱越软。
- `ignore_activation_dynamics = False`：保留激活动力学，控制量 `e` 不直接等于激活 `a`，而是满足 `a_dot = g(e, a)`。
- `ignore_tendon_compliance = False`：保留肌腱弹性，不把肌腱当刚体。
- `fiber_damping = beta`：纤维阻尼项，`F_damp = F0 * beta * v_f_tilde`；常用于改善 Moco 收敛。
- `tendon_compliance_dynamics_mode = implicit`：用隐式形式处理肌腱柔顺动力学，更适合 Moco 优化；不适合常规 forward time stepping。
- `max_contraction_velocity = v_max`：最大收缩速度，单位是“最佳纤维长度/秒”；绝对速度上限是 `Vmax = v_max * l_opt`。
- `pennation_angle_at_optimal = alpha_opt`：最佳长度处的羽状角；沿肌腱方向有效传力为 `F_f * cos(alpha)`。

直觉上：

- 增大 `max_isometric_force`：更容易把负载抬起来。
- 增大 `optimal_fiber_length`：改变长度和速度归一化尺度。
- 增大 `tendon_slack_length` 或 `tendon_strain_at_one_norm_force`：整体上让肌腱更“软”。
- 增大 `pennation_angle_at_optimal`：更多纤维力损失在几何投影上。

---

## 三、OpenSense — IMU 惯性传感器追踪

| 文件 | 内容 |
|------|------|
| `Code/Python/OpenSenseExample/OpenSense_CalibrateModel.py` | 用 IMU 数据校准模型 |
| `Code/Python/OpenSenseExample/OpenSense_IMUDataConverter.py` | IMU 数据格式转换 |
| `Code/Python/OpenSenseExample/OpenSense_OrientationTracking.py` | 基于朝向数据的逆运动学 |

---

## 四、GUI Python 脚本（OpenSim GUI 环境中运行）

- `Code/Python/GUI/runTutorialOne.py` / `runTutorialTwo.py` / `runTutorialThree.py` — GUI 引导式教程
- `Code/Python/GUI/runScaling.py` — 人体测量学缩放
- `Code/Python/GUI/runMultipleIKTrials.py` — 批量逆运动学
- `Code/Python/GUI/alterTendonSlackLength.py`、`strengthenModel.py` — 模型参数修改

---

## 五、完整分析流水线 (Pipelines/)

`Pipelines/` 目录包含多个步态模型的完整分析工作流：

- **IK**（逆运动学）→ **ID**（逆动力学）→ **RRA**（残差缩减）→ **CMC**（计算肌肉控制）
- 可用模型：Gait10dof18musc、Gait2354、Gait2392、Rajagopal、Hamner
- 每个都附带实验数据和参考输出

---

## 六、专题教程 (Tutorials/)

| 教程目录 | 主题 |
|----------|------|
| `Tutorials/Intro_to_Musculoskeletal_Modeling/` | 肌骨建模基础 |
| `Tutorials/Inverese_Kinematics_with_IMUs/` | IMU 逆运动学 |
| `Tutorials/Working_with_Static_Optimization/` | 静态优化 |
| `Tutorials/Computed_Muscle_Control/` | 计算肌肉控制 |
| `Tutorials/Estimating_Joint_Reaction_Loads/` | 关节反力估计 |
| `Tutorials/Design_to_Reduce_Metabolic_Cost/` | 代谢成本优化 |
| `Tutorials/Soccer_Kick/` | 足球踢球生物力学 |
| `Tutorials/Sky_High_Optimal_Jump_Performance/` | 跳跃优化 |
| `Tutorials/Building_a_Passive_Dynamic_Walker/` | 被动动态行走 |

---

## 七、Jupyter Notebook（云端学习）

- `Copy_of_Tutorial_7_Set_up_OpenSim_Moco_in_Google_Colab.ipynb` — 在 Colab 中设置和使用 Moco

---

## 八、预建模型（Models/）

从简单到复杂：

Pendulum → TugOfWar → Arm26 → Gait10dof18musc → Gait2392 → Rajagopal（全身模型）

共 19 个模型、43 个 .osim 文件可供实验。

---

## 推荐学习顺序

1. **Moco 系列**（exampleSlidingMass → exampleSquatToStand → example3DWalking）— 教程第 8-9 章的自然延伸
2. **Pipelines 中的 Rajagopal 流水线** — 学习完整的实验数据处理流程
3. **OpenSense** — 如果涉及 IMU 数据采集
4. **专题教程** — 根据研究方向选择（关节载荷、代谢优化等）


## 等长收缩为什么产力最大？

微观机制：肌球蛋白横桥（cross-bridge）与肌动蛋白的结合循环。
- 等长(v=0)：横桥有充足时间结合，同时处于结合状态的横桥数最多 → 力最大
- 缩短(v>0)：结合位点一闪而过，有效横桥数减少 → 力下降
- 离心(v<0)：横桥被强拉产生更大抵抗力(~1.5F₀)，但属于被动抵抗而非主动产力

## 为什么爆发力比静止发力更大？

Hill 模型说的是"单块肌肉、单一瞬间"的产力上限，真实运动通过多种机制突破：
1. **拉长-缩短循环(SSC)**：反向预拉伸（离心）→ 瞬间转换 → 缩短（向心），离心阶段的力+弹性势能释放叠加
2. **肌腱弹弓效应**：肌腱慢慢蓄能、瞬间释放，峰值力/功率远超肌肉本身
3. **神经因素**：爆发动作可实现更高的运动单元募集率和发放频率
4. **多关节动力链**：多块肌肉依次发力，力量层层叠加

**Why:** 用户在学习 OpenSim Hill-type 肌肉模型过程中的关键理解点
**How to apply:** 后续讨论肌肉模型参数或仿真结果时，可引用这些概念帮助解释

---

## DeGrooteFregly2016Muscle 详解

> 来源：De Groote et al. (2016) *Annals of Biomedical Engineering* 44(10), 1–15.
> C++ 头文件：`Library/sdk/include/OpenSim/Actuators/DeGrooteFregly2016Muscle.h`
> Python 绑定：`Lib/opensim/actuators.py` (SWIG 自动生成)

### 设计哲学

专为 **Direct Collocation 最优控制**设计的 Hill 型肌肉模型：
- 所有曲线用**解析函数**表达，保证连续可微（C²），适合梯度优化
- 状态变量用**归一化肌腱力**（而非传统纤维长度），提升数值稳定性
- 支持**显式和隐式肌腱动力学**，隐式形式对 Moco 优化更鲁棒
- 是 OpenSim 中**唯一经过 Moco 测试的肌肉模型**

### 结构图

```
                    Hill型肌肉-肌腱单元
    ┌──────────────────────────────────────────────────┐
    │                                                  │
    │  ┌────────────── 肌纤维(Fiber) ──────────────┐   │
    │  │                                           │   │
    │  │  ┌─────────────┐  ┌──────────────────┐    │   │     ┌──────────┐
    │  │  │ 主动力 (CE)  │  │ 被动弹性力 (PE) │    │   │     │          │
    │  │  │             │  │                  │    │   │     │  肌腱    │
    │  │  │ a·f_L·f_V   │  │ f_PE(l̃_M)      │    │   ├─────┤ (Tendon) │──── 骨骼
    │  │  │             │  │                  │    │   │     │          │
    │  │  │  ┌───┐      │  └──────────────────┘    │   │     │ f_T(l̃_T)│
    │  │  │  │ ≈ │      │  ┌──────────────────┐    │   │     │          │
    │  │  │  └───┘      │  │ 阻尼力 (DE)      │    │   │     └──────────┘
    │  │  │  神经激活    │  │ β·ṽ_M·F_iso     │    │   │
    │  │  └─────────────┘  └──────────────────┘    │   │
    │  │                                     ╲     │   │
    │  └──────────────────────────────────── α(羽状角)──┘
    │                                              │
    └──────────────────────────────────────────────────┘
```

### 核心数据结构

| 结构体 | 用途 |
|--------|------|
| `MuscleLengthInfo (mli)` | 纤维长度、肌腱长度、羽状角、力-长度乘子等几何信息 |
| `FiberVelocityInfo (fvi)` | 纤维速度、力-速度乘子、沿肌腱的纤维速度 |
| `MuscleDynamicsInfo (mdi)` | 主动力、被动力、肌腱力、刚度、平衡残差 |
| `MusclePotentialEnergyInfo (mpei)` | 弹性势能（被动纤维 + 肌腱） |

**状态变量**：
- `activation`（激活度，范围 (0, ∞]）
- `normalized_tendon_force`（归一化肌腱力，范围 [0, 5]）

### 激活动力学

将神经兴奋信号 `e` 转换为肌肉激活度 `a`：`da/dt = f(e, a)`

使用 `tanh` 平滑切换激活/去激活时间常数：
- `activation_time_constant` = 0.015s（激活更快）
- `deactivation_time_constant` = 0.060s（去激活较慢）
- `activation_dynamics_smoothing` 控制切换平滑度（推荐值 10）

### 四条特征曲线

#### 曲线 1：主动力-长度曲线 `f_L(l̃_M)`

```python
calcActiveForceLengthMultiplier(normFiberLength)
```

三条类高斯曲线之和，峰值在 `l̃_M = 1.0` 处为 1.0：

```
f_L(l̃) = Σ(i=1..3) b1i * exp(-0.5 * (x̃ - b2i)² / (b3i + b4i·x̃)²)
其中 x̃ = (l̃_M - 1)/scale + 1，scale = active_force_width_scale
```

`active_force_width_scale` ≥ 1.0 可加宽曲线，增强力产生能力而不增大 F_iso。

常数（来自 simtk.org/projects/optcntrlmuscle，经修正使 f_L(1)=1）：
```
b11=0.815, b21=1.055, b31=0.162, b41=0.063
b12=0.433, b22=0.717, b32=-0.030, b42=0.200
b13=0.1,   b23=1.0,   b33=0.354,  b43=0.0
```

#### 曲线 2：力-速度曲线 `f_V(ṽ_M)`（静态方法）

```python
DeGrooteFregly2016Muscle.calcForceVelocityMultiplier(normFiberVelocity)
```

对数型曲线，定义域 [-1, 1]，值域 [0, 1.794]：

```
f_V(ṽ) = d1·ln(d2·ṽ + d3 + √((d2·ṽ + d3)² + 1)) + d4
```

关键点：f_V(-1)=0（最大缩短）, f_V(0)=1（等长）, f_V(1)≈1.794（最大拉长）

反函数：`calcForceVelocityInverseCurve(fvm)` = `(sinh((fvm - d4)/d1) - d3) / d2`

常数（经修正使曲线通过 (-1,0) 和 (0,1)）：
```
d1=-0.3211, d2=-8.149, d3=-0.374, d4=0.8825
```

#### 曲线 3：被动力-长度曲线 `f_PE(l̃_M)`

```python
calcPassiveForceMultiplier(normFiberLength)
```

指数曲线：

```
offset = exp(kPE * (l̃_min - 1) / e0)
f_PE(l̃) = (exp(kPE * (l̃ - 1) / e0) - offset) / (exp(kPE) - offset)
```

- `kPE = 4.0`（指数形状因子，固定）
- `e0 = passive_fiber_strain_at_one_norm_force`（默认 0.6）
- **注意**：在最优纤维长度（l̃=1）处被动力不为零，这与 Muscle 基类的描述不同

#### 曲线 4：肌腱力-长度曲线 `f_T(l̃_T)`

```python
calcTendonForceMultiplier(normTendonLength)
```

```
f_T(l̃_T) = c1 * exp(kT * (l̃_T - c2)) - c3
kT = ln((1 + c3) / c1) / (1 + ε₀ᵀ - c2)
```

- `c1=0.2, c2=1.0, c3=0.2`（OpenSim 修正值，原论文 c2=0.995, c3=0.250）
- `ε₀ᵀ = tendon_strain_at_one_norm_force`（默认 0.049）

反函数：`calcTendonForceLengthInverseCurve(F̃_T)` = `ln((F̃_T + c3)/c1) / kT + c2`

### 纤维力计算

`calcFiberForce()` 计算三个分量：

```
F_fiber = F_iso * [a · f_L(l̃_M) · f_V(ṽ_M)     ← 主动力（CE）
                 + f_PE(l̃_M)                      ← 保守被动力（弹性PE）
                 + β · ṽ_M]                        ← 非保守被动力（阻尼DE）
```

其中 `β = fiber_damping`（默认 0，推荐设非零值改善优化收敛）。

### 肌肉-肌腱平衡方程

几何约束（固定宽度羽状模型）：

```
l_MT = l_T + l_M · cos(α)
sin(α) = w / l_M，其中 w = l_opt · sin(α₀) 为常数（纤维宽度）
```

力平衡（沿肌腱方向）：

```
residual = F̃_T - F_fiber · cos(α) / F_iso = 0
```

`calcEquilibriumResidual()` 的计算链（隐式模式）：

```
给定: l_MT, v_MT, a, F̃_T, dF̃_T/dt
  ↓
calcMuscleLengthInfoHelper()     → 由 F̃_T 反推 l_T，再得 l_M, α, f_L, f_PE
  ↓
calcFiberVelocityInfoHelper()    → 由 dF̃_T/dt 算 v_T，再得 v_M, ṽ_M, f_V
  ↓
calcMuscleDynamicsInfoHelper()   → 组合得到 F_fiber, F_tendon, 残差
  ↓
residual = F̃_T - F_fiber·cosα / F_iso
```

### 显式 vs 隐式模式

| | 显式 (explicit) | 隐式 (implicit) |
|---|---|---|
| **状态变量** | `normalized_tendon_force` | 同左 |
| **导数计算** | ODE: dF̃_T/dt 由力平衡直接算出 | dF̃_T/dt 作为离散变量由优化器提供 |
| **平衡约束** | 自动满足 | 残差 = 0 作为路径约束 |
| **适用场景** | 正向仿真 (Manager) | Moco 最优控制 |
| **优势** | 简单 | 对初始猜测鲁棒 |

隐式模式是论文的 **Formulation 3**，推荐用于优化。

### 刚度计算

串联弹簧等效刚度：

```
K_muscle = (K_fiber_along_tendon × K_tendon) / (K_fiber_along_tendon + K_tendon)
```

其中：
- `K_fiber = F_iso * (a · df_L/dl̃ · f_V + df_PE/dl̃) / l_opt`
- `K_tendon = F_iso / l_slack * df_T/dl̃_T`

### 实用工具方法

```python
# 批量替换模型中的旧肌肉类型
osim.DeGrooteFregly2016Muscle.replaceMuscles(model)

# 导出曲线数据
muscle.printCurvesToSTOFiles("output_dir/")
table = muscle.exportFiberLengthCurvesToTable()
```

### Python 使用示例

```python
import opensim as osim

actu = osim.DeGrooteFregly2016Muscle()
actu.setName("muscle")
actu.set_max_isometric_force(30.0)
actu.set_optimal_fiber_length(0.10)
actu.set_tendon_slack_length(0.05)
actu.set_tendon_strain_at_one_norm_force(0.10)
actu.set_fiber_damping(0.01)
actu.set_tendon_compliance_dynamics_mode("implicit")
actu.set_max_contraction_velocity(10)
actu.set_pennation_angle_at_optimal(0.10)
actu.addNewPathPoint("origin", model.updGround(), osim.Vec3(0))
actu.addNewPathPoint("insertion", body, osim.Vec3(0))
model.addForce(actu)
```

完整示例：`Library/Resources/Code/Python/Moco/exampleHangingMuscle.py`

### 归一化纤维长度/肌腱力的允许范围

```
normalized_fiber_length:   [0.2, 1.8]
normalized_tendon_force:   [0.0, 5.0]
```