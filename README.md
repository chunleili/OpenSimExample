# OpenSim Example + VBD Volumetric Muscle Simulation

OpenSim 生物力学示例集合 + 基于 Vertex Block Descent (VBD) 的体积肌肉仿真原型。

## Tutorials/Mine 示例脚本

以下脚本位于 `Tutorials/Mine/`，所有输出保存在 `output/` 文件夹下（`output_` 前缀）。

| 脚本 | 说明 | 依赖 |
|---|---|---|
| `tutorial_opensim_python.py` | OpenSim Python API 系统教程（9 章），涵盖模型构建、仿真、Moco 优化、数据 IO 等 | OpenSim |
| `run_sliding_mass.py` | Moco 滑块最短时间最优控制，最简入门示例 | OpenSim |
| `validation_level0_single_muscle.py` | Level 0：用 DeGrooteFregly2016Muscle 生成力-长度、力-速度、激活动力学基准曲线 | OpenSim |
| `run_level0_fem.py` | Level 0 FEM 端：DGF 解析曲线 + 本构应力对比 + VBD 准静态力-长度验证 | numpy, vbd_muscle |
| `validation_level1_joint_torques.py` | Level 1：FEM 关节角度 → OpenSim InverseDynamics → 关节力矩对比 | OpenSim |
| `validation_level2_moco_inverse.py` | Level 2：MocoInverse 反解肌肉激活值，与 RL 策略输出对比 | OpenSim |
| `vbd_energy.cl` | VBD 能量公式的 OpenCL 内核（GPU 实验） | — |

```bash
# 安装 conda 环境
conda env create -f environment.yml
conda activate opensim

# 运行示例（在 Tutorials/Mine/ 目录下）
cd Tutorials/Mine
python run_sliding_mass.py
python validation_level0_single_muscle.py
python run_level0_fem.py --skip-vbd   # 跳过耗时的 VBD 部分
```

## Quick Start

```bash
conda env create -f environment.yml
conda activate opensim
python main.py
python main.py --help
```

## 可用命令

| 命令 | 说明 |
|---|---|
| `python main.py` | 默认：Phase 1 测试 + DGF 曲线图 |
| `python main.py test` | Phase 1 验证测试（梯度 FD、Hessian、网格等） |
| `python main.py curves` | 绘制 DeGrooteFregly2016 解析力-长度曲线 |
| `python main.py stress` | 本构模型应力 vs DGF 参考对比 |
| `python main.py activation` | 激活动力学阶跃响应 |
| `python main.py vbd` | VBD 准静态力-长度验证（较慢） |
| `python main.py demo` | VBD 动态演示：重力 + 肌肉收缩 |
| `python main.py --all` | 运行全部验证 |

## 项目结构

```
.
├── Tutorials/Mine/                # 自研脚本（见上方表格）
│   ├── tutorial_opensim_python.py #   OpenSim Python API 教程
│   ├── run_sliding_mass.py        #   Moco 滑块最优控制
│   ├── run_level0_fem.py          #   Level 0 FEM 验证
│   ├── validation_level0_single_muscle.py  # Level 0 OpenSim 基准
│   ├── validation_level1_joint_torques.py  # Level 1 关节力矩对比
│   ├── validation_level2_moco_inverse.py   # Level 2 MocoInverse
│   └── vbd_energy.cl              #   OpenCL 内核（GPU 实验）
│
├── vbd_muscle/                    # VBD 体积肌肉仿真包
│   ├── dgf_curves.py              #   DeGrooteFregly2016 曲线 (f_L, f_PE, f_V)
│   ├── fem.py                     #   线性四面体 FEM 工具
│   ├── constitutive.py            #   Neo-Hookean + Hill-type 纤维本构
│   ├── mesh.py                    #   四面体网格生成 (box / cylinder)
│   ├── coloring.py                #   图着色 (VBD 并行)
│   ├── solver.py                  #   VBD 求解器
│   └── activation.py              #   肌肉激活动力学
│
├── tests/
│   └── test_phase1.py             # Phase 1 验证：梯度 FD、Hessian、网格、激活
│
├── docs/
│   ├── plan_VBD_muscle.md         #   VBD 研究计划（含进度）
│   ├── plan_RLMuscle.md           #   RL 肌肉控制计划
│   ├── learn_opensim.md           #   OpenSim/Moco 学习指南 + DGF 公式
│   └── muscle_force_generation_principles.md  # Hill 肌肉产力原理
│
├── main.py                        # VBD 验证主入口
├── environment.yml                # conda 环境配置
│
├── Models/                        # OpenSim 模型 (Rajagopal, Arm26, ...)
├── Geometry/                      # 可视化几何文件
├── Code/                          # OpenSim 官方示例脚本 (Python/C++/Matlab)
├── Pipelines/                     # 分析流程 (IK→ID→RRA→CMC)
└── Tutorials/                     # OpenSim 官方教程
```

## vbd_muscle 包

基于 [Vertex Block Descent (Chen et al., SIGGRAPH 2024)](https://graphics.cs.utah.edu/research/projects/vbd/) 的体积肌肉仿真 CPU 原型。

### 核心思路

将 Hill-type 肌肉力学嵌入横观各向同性超弹性 FEM 框架：

- **各向同性基质**：Modified Neo-Hookean（偏量-体积分解）
- **纤维项**：DeGrooteFregly2016 解析曲线直接映射为 PK1 应力
- **求解器**：VBD 逐顶点 Gauss-Seidel 迭代，支持 Rayleigh 阻尼
- **验证**：PK1 梯度 vs 能量有限差分，rel_err < 1e-10

### 当前状态

| Phase | 内容 | 状态 |
|---|---|---|
| Phase 1 | 能量公式 + 梯度/Hessian 验证 | 完成 |
| Phase 2 | VBD CPU 求解器 | 完成 |
| Phase 3.0 | 网格生成 | 完成 |
| Phase 3.1 | Level 0 单肌肉 F-L 对比 | 部分完成 |
| Phase 2 GPU | Warp 移植 | 待做 |
| Phase 3.2-3.3 | Level 1-2 关节力矩/多肌肉 | 待做 |
| Phase 4 | 全身 80+ 肌肉 + RL | 待做 |

详见 [docs/plan_VBD_muscle.md](docs/plan_VBD_muscle.md)。

## 依赖

- Python 3.11（conda 环境，见 `environment.yml`）
- OpenSim 4.5.2（含 Moco，通过 opensim-org channel 安装）
- numpy, matplotlib
