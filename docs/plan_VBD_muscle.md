# 研究计划：基于 VBD 的体积肌肉仿真，以 Hill-type 模型为配准基准

## Context

**问题**：当前体积肌肉仿真（如 Lee et al. SIGGRAPH 2018 "Dexterous Manipulation"）使用 Projective Dynamics (PD) 作为求解器。PD 的固定线性系统限制了收敛速度，且只能处理可投影分解的能量函数，不利于精确嵌入 Hill-type 非线性肌肉力学。

**方案**：用 Vertex Block Descent (VBD, Chen et al. SIGGRAPH 2024) 替代 PD。VBD 可处理任意能量函数、收敛性优于 PD、GPU 友好（顶点着色仅需 3-8 色 vs 单元着色 7-76 色）。

**最终目标**：全身肌骨系统（80+ 肌肉，Rajagopal 级别），配合 RL 控制。

**用户决策**：
- 求解器：VBD 优先
- F-V 处理：Rayleigh 阻尼近似（同 Dexterous 论文做法）
- 代码基础：部分已有（网格处理、可视化），需补全 FEM 弹性体管线
- 已有 OpenSim 配准脚本：`validation_level0/1/2_*.py`

---

## Phase 1：肌肉能量公式（数学推导 + 原型验证）

### 1.1 总能量分解

体积肌肉的应变能密度为横观各向同性超弹性模型：

```
W(C, a) = W_iso(I₁, J) + W_fiber(I₄, a)
```

- `I₁ = tr(C)`, `J = det(F)` — 各向同性不变量
- `I₄ = d₀ᵀ C d₀ = λ²` — 纤维方向拉伸（`d₀` 为参考构型纤维方向，`λ = √I₄` 为纤维拉伸比）
- `a ∈ [0, 1]` — 肌肉激活度

### 1.2 各向同性基质能量（Neo-Hookean）

```
W_iso = (μ/2)(I̅₁ - 3) + (κ/2)(J - 1)²
I̅₁ = J^(-2/3) · I₁
```

- `μ` ≈ 1-10 kPa（静息肌肉剪切模量）
- `κ` ≈ 100μ（近不可压）

### 1.3 纤维能量 ← Hill-type 映射（核心）

**关键思路**：不需要显式积分 f_L/f_PE 得到能量 W，VBD 只需要梯度和 Hessian，即只需 f_L(λ)、f_PE(λ) 及其导数。

**纤维应力（2nd Piola-Kirchhoff）**：

```
S_fiber = σ₀/λ · [a · f_L(λ) + f_PE(λ)] · (d₀ ⊗ d₀)
```

其中 `σ₀ = F₀ / PCSA`（峰值等长应力，文献 200-400 kPa，常用 300 kPa）。

**1st Piola-Kirchhoff 应力**（VBD 梯度所需）：

```
P_fiber = F · S_fiber = σ₀/λ · [a · f_L(λ) + f_PE(λ)] · F(d₀ ⊗ d₀)
```

**f_L 和 f_PE 直接使用 DeGrooteFregly2016 的解析公式**（见 `learn_opensim.md:236-310`）：
- f_L：三高斯之和，常数 b11=0.815, b21=1.055, b31=0.162, b41=0.063 等
- f_PE：指数形式，kPE=4.0, e0=0.6

### 1.4 VBD 所需的每顶点梯度和 Hessian

对线性四面体单元 `e`，形变梯度 `F_e = D_s · D_m⁻¹`。

**顶点 i 的梯度贡献**：
```
g_i^e = V_e · P_e : (∂F_e/∂x_i)
```

**顶点 i 的 3×3 Hessian 贡献**：
```
H_i^e = V_e · contract(∂²W/∂F∂F, ∂F_e/∂x_i, ∂F_e/∂x_i)
```

纤维项的弹性张量需要 `d²W_fiber/dI₄²`：
```
d²W/dI₄² = σ₀/(4λ³) · [a·(f_L'·λ - f_L) + (f_PE'·λ - f_PE)]
```

**实现要点**：Hessian 投影到半正定（钳制负特征值为零），确保 VBD 收敛。

### 1.5 参数映射表

| Hill 参数 | FEM 参数 | 映射关系 |
|---|---|---|
| `max_isometric_force` (F₀) | `σ₀` | `σ₀ = F₀ / PCSA` |
| `optimal_fiber_length` (l_opt) | 参考构型网格 | 网格在 l_opt 处建模 |
| `tendon_slack_length` (l_slack) | 肌腱单元静息长度 | 高刚度区域 |
| `pennation_angle_at_optimal` (α₀) | 纤维方向场 `d₀` | 每单元赋纤维方向 |
| `fiber_damping` (β) | Rayleigh 阻尼 k_d | `k_d = β·σ₀/(V_max·l_opt)` |
| `max_contraction_velocity` (V_max) | 阻尼参数 | 用于 k_d 推导 |

### 1.6 实现状态

- [x] DGF 曲线函数 + 解析导数 (`vbd_muscle/dgf_curves.py`)
- [x] Neo-Hookean + 纤维 PK1 应力 (`vbd_muscle/constitutive.py`)
- [x] FD 梯度验证：rel_err < 1e-10 (`tests/test_phase1.py`)
- [x] FD Hessian 一致性验证 (`tests/test_phase1.py`)

---

## Phase 2：VBD 求解器实现

### 2.1 变分隐式 Euler

```
x^{t+1} = argmin_x G(x) = 1/(2h²) ||x - y||²_M + E(x)
y = x^t + h·v^t + h²·g
```

### 2.2 逐顶点更新（Gauss-Seidel）

```
对每个顶点 i:
  f_i = -(m_i/h²)(x_i - y_i) - Σ_{e∈F_i} ∂E_e/∂x_i
  H_i = (m_i/h²)I₃ + Σ_{e∈F_i} ∂²E_e/∂x_i²
  H_i += (k_d/h) · H_elastic_i                    ← Rayleigh 阻尼
  f_i -= (k_d/h) · H_elastic_i · (x_i - x_i^t)   ← Rayleigh 阻尼
  H_i = project_SPD(H_i)                           ← 半正定投影
  Δx_i = H_i⁻¹ · f_i                              ← 3×3 直接求解
  x_i ← x_i + Δx_i
```

### 2.3 图着色与 GPU 并行

- 顶点邻接图着色（tet mesh 典型 3-8 色）
- 每色内顶点并行更新
- 每顶点一个 thread block，warp 处理相邻单元
- 共享内存做 3×3 系统规约

### 2.4 肌肉-骨骼耦合

**阶段 1-3（简化）**：附着点顶点用 Dirichlet 边界条件钉在骨骼上，骨骼运动由 OpenSim 运动学处方。

**阶段 4（双向）**：肌肉附着力作用于刚体骨骼，骨骼刚体动力学与 VBD 交替迭代。

### 2.5 激活动力学

独立 ODE，与 VBD 空间求解解耦：
```
a^{t+1} = a^t + h · f(e^{t+1}, a^{t+1})
```
使用 DGF 的 tanh 平滑切换（τ_a=0.015s, τ_d=0.060s）。

### 2.6 实现状态

- [x] 变分隐式 Euler (`vbd_muscle/solver.py: VBDSolver.step`)
- [x] 逐顶点 Gauss-Seidel (`vbd_muscle/solver.py`)
- [x] 图着色 (`vbd_muscle/coloring.py`)
- [x] SPD Hessian 投影 (`vbd_muscle/constitutive.py: project_spd`)
- [x] Rayleigh 阻尼 (`vbd_muscle/solver.py`)
- [x] Dirichlet BC + 逐自由度约束 (`vbd_muscle/solver.py: set_fixed_dof`)
- [x] 准静态求解 + 步长限制 (`vbd_muscle/solver.py: solve_static`)
- [x] 激活动力学 (`vbd_muscle/activation.py`)
- [ ] GPU 移植（Taichi / CUDA）
- [ ] 解析 Hessian（目前用 FD，GPU 版需解析）

---

## Phase 3：Hill-type 配准（复用已有脚本）

### 3.0 网格生成

单肌肉测试用圆柱网格：
- 长度 = l_opt + l_slack = 0.10 + 0.20 = 0.30 m
- 半径由 PCSA 决定：PCSA = F₀/σ₀ = 1000/300000 = 3.33 cm², r ≈ 1.03 cm
- 纤维方向：轴向
- 目标 1000-5000 四面体（TetGen/gmsh）

**实现状态**：
- [x] Box mesh: 6-tet Kuhn 分解 (`vbd_muscle/mesh.py: generate_box_mesh`)
- [x] Cylinder mesh: 中心棱柱 + 环形六面体 (`vbd_muscle/mesh.py: generate_cylinder_mesh`)
- [x] 纤维方向赋值 (`vbd_muscle/mesh.py: assign_fiber_directions`)

### 3.1 Level 0：单肌肉 F-L / F-V / 激活曲线

**复用** `validation_level0_single_muscle.py`（已有 OpenSim 基准曲线生成）

FEM 端需新建：
1. 构建圆柱 tet mesh，参数匹配 `MUSCLE_PARAMS`
2. F-L：扫描 λ ∈ [0.4, 1.8]，等长条件（v=0），比较归一化力
3. F-V：固定 λ=1.0，扫描归一化速度 ∈ [-1, 1]，比较力（此处将验证 Rayleigh 阻尼近似的偏差）
4. 激活动力学：阶跃 e: 0.01→1.0，比较力响应曲线

**验收标准**：F-L 的 RMSE < 5%；F-V 的 Rayleigh 线性近似与 Hill 对数曲线的偏差在 |ṽ| < 0.5 区域内 < 10%

**实现状态**：
- [x] 解析 F-L 对比：FEM 本构 vs DGF < 5%（λ ∈ [0.7, 1.5]）(`run_level0_fem.py` Part 2)
- [x] 激活动力学验证：50% 上升 11ms，90% 上升 38ms (`run_level0_fem.py` Part 4)
- [x] VBD 准静态 F-L：λ=1.0 时 -999N vs -1000N 期望 (`run_level0_fem.py` Part 3)
- [ ] F-V 曲线（需 Rayleigh 阻尼标定）
- [ ] 与 OpenSim Level 0 输出叠加对比

### 3.2 Level 1：关节力矩

**复用** `validation_level1_joint_torques.py` 的 `convert_fem_to_sto()` 和 `compare_joint_torques()`

1. 建简单关节模型（铰链 + 1 根肌肉）
2. 处方关节运动，VBD 正向仿真
3. 从附着点反力计算关节力矩
4. 与 OpenSim InverseDynamics 对比

**验收标准**：RMSE < 10% 峰值力矩，R² > 0.9

### 3.3 Level 2：多肌肉激活

**复用** `validation_level2_moco_inverse.py` 的 `run_moco_inverse()` 和 `compare_activations()`

1. 多肌肉 FEM 模型 + RL 策略输出激活值
2. 导出关节运动学 → MocoInverse 反解 OpenSim 激活
3. 逐肌肉对比

### 3.4 灵敏度分析

| 参数 | 扫描范围 | 影响 |
|---|---|---|
| σ₀ | 200-500 kPa | 峰值力幅度 |
| μ (基质模量) | 0.5-20 kPa | 被动刚度 |
| κ (体积模量) | 50μ-500μ | 体积保持 |
| k_d (阻尼) | 0.001-0.1 | F-V 斜率 |
| 网格密度 | 500-10000 tets | 精度/性能 |
| VBD 迭代数 | 5-50 | 求解器收敛 |
| 时间步长 h | 0.1ms-10ms | 稳定性/速度 |

---

## Phase 4：全身扩展 + RL 集成

### 4.1 全身网格生成

从 `Models/Rajagopal/Rajagopal2016.osim` 提取 80+ 肌肉路径点 → 生成 swept cylinder tet mesh → 赋纤维方向和 Hill 参数。

### 4.2 GPU 性能估计

- 500K 顶点 × 56 bytes ≈ 28 MB（顶点数据）
- 2M 单元 × 72 bytes ≈ 144 MB（单元数据）
- 20 次 VBD 迭代 × 8 色 ≈ 24 ms/步（h=1ms 时约 24× 慢于实时）
- h=10ms 时约 2.4× 慢于实时（RL 训练可接受）

### 4.3 RL 集成

```
RL Environment:
  观测: 关节角度、角速度、任务状态
  动作: 肌肉兴奋值 e (dim = n_muscles)
  Step:
    1. 激活动力学: a^{t+1} = f(e, a^t)
    2. VBD 求解: x^{t+1}, v^{t+1}
    3. 提取关节状态
    4. 计算奖励
```

框架选择：Isaac Lab（GPU RL + 自定义物理）或 stable-baselines3 + 自定义 env。

---

## 实施顺序与依赖

```
Phase 1 (能量公式, 数学推导)     ─── 1 周 ✅ 已完成
    │
Phase 2.1-2.3 (VBD CPU 原型)     ─── 3 周 ✅ 已完成
    │   ├── Phase 3.0 (网格生成)  ─── 1 周 ✅ 已完成
    │
Phase 2.4-2.5 (阻尼+着色)        ─── 1 周 ✅ 已完成
    │
Phase 3.1 (Level 0 单肌肉配准)    ─── 2 周 ⚠️ 部分完成（解析对比完成，VBD F-L 需优化）
    │
Phase 2.6 (肌骨耦合)              ─── 2 周 (待做)
    │
Phase 3.2 (Level 1 关节力矩)      ─── 2 周 (待做)
    │
Phase 2 GPU (GPU 移植)             ─── 3-4 周 (待做，CPU 验证后)
    │
Phase 3.3 (Level 2 多肌肉)        ─── 2 周 (待做)
    │
Phase 4.1-4.2 (全身网格+GPU优化)   ─── 4 周 (待做)
    │
Phase 4.3 (RL 集成)               ─── 4 周 (待做)
```

## 技术风险与缓解

| 风险 | 缓解策略 |
|---|---|
| Rayleigh 阻尼在高速度区与 Hill F-V 偏差大 | 添加非线性速度依赖阻尼项：计算每单元 v_fiber = (λ^{t+1}-λ^t)/h，评估 f_V(v_fiber/V_max) 作为额外力 |
| VBD 对硬肌腱收敛慢 | 肌腱单独建模为硬约束或极高刚度弹簧；考虑 XPBD 作为肌腱 fallback |
| 复杂肌肉路径的网格生成困难 | 从简单圆柱开始；绕包肌肉用碰撞几何处理 |
| Hessian 投影丢失负曲率信息 | 使用 VBD 论文推荐的"翻转最小特征值"策略 |
| 全身 GPU 性能不达实时 | RL 训练用粗网格；降采样；现代 GPU (A100/H100) |
| 纯 Python VBD 性能差 | 用 Numba JIT 加速内循环；或直接 Taichi 重写 |

## 推荐技术栈

- **网格生成**: gmsh (Python API) / TetGen
- **CPU 原型**: Python + NumPy（已实现）→ Numba 加速
- **GPU 实现**: Taichi（快速原型）→ CUDA（生产级）
- **RL 框架**: Isaac Lab / stable-baselines3
- **OpenSim 配准**: 已有脚本（`validation_level0/1/2_*.py`）
- **可视化**: Polyscope / Open3D

## 关键文件

| 用途 | 路径 |
|---|---|
| VBD 肌肉仿真包 | `vbd_muscle/` |
| Phase 1 验证测试 | `tests/test_phase1.py` |
| Level 0 FEM 验证 | `run_level0_fem.py` |
| F-L/F-V 基准曲线 | `validation_level0_single_muscle.py` |
| 关节力矩对比 | `validation_level1_joint_torques.py` |
| MocoInverse 激活对比 | `validation_level2_moco_inverse.py` |
| DGF 曲线公式常数 | `learn_opensim.md:236-310` |
| Hill 模型原理 | `muscle_force_generation_principles.md` |
| 全身模型 | `Models/Rajagopal/Rajagopal2016.osim` |

## Verification

1. **Phase 1 验证** ✅：单四面体 PK1 梯度 vs 有限差分，rel_err < 1e-10
2. **Phase 2 验证** ✅：VBD 求解 Neo-Hookean（无纤维），静息状态稳定（max_disp < 1e-19）
3. **Phase 3 验证** ⚠️：`run_level0_fem.py` 解析 F-L 对比 < 5%（λ ∈ [0.7, 1.5]），VBD F-L 收敛待优化
4. **Phase 4 验证**（待做）：全身模型跑步态周期，关节力矩 RMSE < 10%，reserve actuator < 5% 峰值力矩
