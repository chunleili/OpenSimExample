# 研究计划：使用 OpenSim 配准体积肌肉 + RL 仿真结果

## Context

用户正在做 RL + 体积肌肉（volumetric muscle, FEM 四面体网格）的研究。其仿真链条为：激活值 → 肌肉收缩（各向异性 FEM 变形）→ 关节力矩 → 骨骼运动。需要借助 OpenSim 作为**生理学基准**来验证其物理仿真结果的正确性，并在"逆问题"（视频→激活值）中提供对照。

---

## 1. 推荐使用的 Actuator / Muscle Model

### 主选：`DeGrooteFregly2016Muscle`

**原因：**
- OpenSim 中**唯一与 Moco 优化框架充分测试过**的肌肉模型（见 `exampleHangingMuscle.py:43-44`）
- 支持 implicit tendon compliance dynamics —— 可与 FEM 肌腱行为做对比
- 支持配置所有 Hill-type 参数：`max_isometric_force`, `optimal_fiber_length`, `tendon_slack_length`, `pennation_angle_at_optimal`, `fiber_damping`, `max_contraction_velocity`
- 可通过 `MocoInverse` 从关节角度反解肌肉激活值 —— 这正是配准所需的核心功能

### Hill-type 参数 ↔ FEM 体积肌肉参数映射

| OpenSim 参数 | FEM 体积肌肉对应量 |
|---|---|
| `max_isometric_force` | 峰值应力 × 生理横截面积 (PCSA) |
| `optimal_fiber_length` | 参考构型下沿纤维方向的静息长度 |
| `tendon_slack_length` | 肌腱单元静息长度 |
| `pennation_angle_at_optimal` | 纤维方向与肌腱作用线的夹角 |
| `fiber_damping` | 本构模型中的粘性阻尼系数 |
| `max_contraction_velocity` | 沿纤维方向的最大应变率 |
| Activation dynamics (excitation→activation 滤波) | RL 策略输出 → 激活值映射 |

### 不选 CoordinateActuator 的原因

CoordinateActuator 直接对关节自由度施加力矩，**绕过了肌肉力学**（无力-长度-速度关系、无激活动力学），无法验证体积肌肉行为。仅适合作为调试用的 reserve actuator。

---

## 2. 配准工作流（4 个层级，逐步递进）

### Level 0：单肌肉标定（隔离肌肉模型差异）

**目标**：验证 FEM 体积肌肉与 Hill-type 模型在基本力学特性上的一致性。

**做法**：
1. 用 `exampleHangingMuscle.py` 模式构建单自由度 + 单肌肉的 OpenSim 模型
2. 构建几何匹配的 FEM 模型
3. 对比以下曲线：
   - **力-长度关系**：固定激活值=1，扫描关节角度，测量等长力
   - **力-速度关系**：固定激活值=1，不同收缩速度下的力输出
   - **激活动力学**：阶跃输入激活值，测量力上升时间常数

**参考文件**：
- `Code/Python/Moco/exampleHangingMuscle.py` — 完整的 DeGrooteFregly2016Muscle 配置 + AnalyzeTool 提取纤维力/速度

### Level 1：关节力矩对比（模型无关的最稳健对比）

**目标**：给定相同的关节角度轨迹，对比两个系统计算出的关节力矩。

**做法**：
1. 从 FEM 仿真导出关节角度时间序列 → `.sto` 文件
2. 用 `InverseDynamicsTool` 在 OpenSim 中计算关节力矩
3. 与 FEM 直接计算的关节力矩做 RMSE、R² 对比

**数据转换代码**：
```python
import opensim as osim
import numpy as np

table = osim.TimeSeriesTable()
table.setColumnLabels(['hip_flexion_r', 'knee_angle_r', 'ankle_angle_r'])
for i in range(len(fem_times)):
    row = osim.RowVector.createFromMat(fem_angles[i])
    table.appendRow(fem_times[i], row)
table.addTableMetaDataString('inDegrees', 'no')
osim.STOFileAdapter.write(table, 'fem_coordinates.sto')
```

**参考文件**：
- `Pipelines/Rajagopal/ID/id_setup_walk.xml` — InverseDynamics 配置模板
- `tutorial_opensim_python.py` 第 4-5 章 — .sto 读写

### Level 2：肌肉激活值对比（核心配准）

**目标**：给定相同的关节运动学，对比 OpenSim MocoInverse 解出的激活值 vs RL 策略产生的激活值。

**做法**：
1. 将 FEM 关节角度导出为 `coordinates.sto`
2. 用 `MocoInverse` + `DeGrooteFregly2016Muscle` 反解激活值
3. 逐肌肉对比激活值时间曲线

**MocoInverse 代码**：
```python
inverse = osim.MocoInverse()
modelProcessor = osim.ModelProcessor('Models/Rajagopal/Rajagopal2016.osim')
modelProcessor.append(osim.ModOpReplaceMusclesWithDeGrooteFregly2016())
modelProcessor.append(osim.ModOpIgnoreTendonCompliance())
modelProcessor.append(osim.ModOpIgnorePassiveFiberForcesDGF())
modelProcessor.append(osim.ModOpScaleActiveFiberForceCurveWidthDGF(1.5))
modelProcessor.append(osim.ModOpAddReserves(1.0))
inverse.setModel(modelProcessor)
inverse.setKinematics(osim.TableProcessor('fem_coordinates.sto'))
inverse.set_initial_time(t0)
inverse.set_final_time(tf)
inverse.set_mesh_interval(0.02)
inverse.set_kinematics_allow_extra_columns(True)
solution = inverse.solve()
solution.getMocoSolution().write('opensim_inverse_solution.sto')
```

**参考文件**：
- `Code/Python/Moco/example3DWalking/exampleMocoInverse.py` — 完整 MocoInverse 流程含 EMG 追踪
- `Code/Python/Moco/exampleEMGTracking/` — EMG 对比可视化

### Level 3：灵敏度分析

对比不同 OpenSim 配置下的结果变化，理解 Hill-type 建模假设对配准的影响：

| 开关 | 对应 ModOp |
|---|---|
| 有/无肌腱柔度 | `ModOpIgnoreTendonCompliance()` |
| 有/无被动纤维力 | `ModOpIgnorePassiveFiberForcesDGF()` |
| 力-长度曲线宽度 | `ModOpScaleActiveFiberForceCurveWidthDGF(1.0~2.0)` |
| 纤维阻尼 | `muscle.set_fiber_damping(0.001~0.1)` |

---

## 3. 对比指标

| 层级 | 指标 | 计算方法 |
|---|---|---|
| 关节力矩 | RMSE, R², 峰值时序误差 | ID 输出 vs FEM 力矩 |
| 肌肉激活 | 互相关系数, RMSE, 峰值对齐 | MocoInverse vs RL 策略 |
| 纤维力 | 归一化力对比 | AnalyzeTool 提取 `ActiveFiberForce` |
| 力臂 | 各关节角度下的力臂曲线 | `force.computeMomentArm(state, coord)` |
| 残差力 | Reserve actuator 力 < 5% 峰值关节力矩 | MocoInverse 输出中的 reserve 列 |
| 代谢成本 | 总代谢率 | `Umberger2010MuscleMetabolicsProbe` |

---

## 4. 推荐模型

| 阶段 | 模型 | 路径 |
|---|---|---|
| 原型验证 | Arm26 (2 DOF, 6 muscles) | `Models/Arm26/` |
| 简化步态 | Gait10dof18musc | `Models/Gait10dof18musc/` |
| 完整步态 | Rajagopal 2016 (80+ muscles) | `Models/Rajagopal/Rajagopal2016.osim` |

---

## 5. 未来：视频反演流水线

```
视频 → 姿态估计 (MediaPipe/OpenPose) → 3D 关节位置 (.trc)
  → OpenSim IK → 关节角度 (.sto)
    ├→ MocoInverse → OpenSim 激活值 (基准)
    └→ FEM+RL 策略 → 体积肌肉激活值 (你的方法)
        → 对比两组激活值
```

---

## 6. 关键参考文件汇总

| 用途 | 文件 |
|---|---|
| MocoInverse 完整模板 | `Code/Python/Moco/example3DWalking/exampleMocoInverse.py` |
| DeGrooteFregly2016 全参数 | `Code/Python/Moco/exampleHangingMuscle.py` |
| 数据格式转换 | `tutorial_opensim_python.py` 第 4-5 章 |
| EMG 追踪对比 | `Code/Python/Moco/exampleEMGTracking/` |
| ID 配置模板 | `Pipelines/Rajagopal/ID/id_setup_walk.xml` |
| 完整步态流水线 | `Pipelines/Rajagopal/` |
| AnalyzeTool 后处理 | `Code/Python/Moco/exampleHangingMuscle.py:156-175` |

---

## Verification

1. **Level 0 验证**：修改 `exampleHangingMuscle.py`，扫描关节角度和收缩速度，导出力-长度和力-速度曲线，与 FEM 结果绘图对比
2. **Level 1 验证**：用 Rajagopal 管线的 `coordinates.sto` 作为测试输入运行 InverseDynamicsTool，与 `OutputReference` 比对
3. **Level 2 验证**：用 `example3DWalking/coordinates.sto` 运行 MocoInverse，与 repo 中提供的 EMG 数据对比，确认流程正确后替换为 FEM 数据
