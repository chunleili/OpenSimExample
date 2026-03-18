"""
================================================================================
  Level 2：MocoInverse 肌肉激活值对比
================================================================================

目标：给定 FEM 仿真的关节运动学，用 MocoInverse 反解 OpenSim 的肌肉激活值，
      与 RL 策略产生的激活值进行对比。

用法：
    python validation_level2_moco_inverse.py

输出：
    output/output_level2_inverse_solution.sto —— MocoInverse 解（激活值 + 状态）
    output/output_level2_comparison.sto       —— 对比结果
================================================================================
"""

import opensim as osim
import numpy as np
import math
import os


# ============================================================================
# 配置区
# ============================================================================

# OpenSim 模型
# 简单模型（快，用于调试）：
MODEL_PATH = os.path.join("Models", "Gait10dof18musc", "gait10dof18musc.osim")
# 完整模型（慢，用于正式实验）：
# MODEL_PATH = os.path.join("Models", "Rajagopal", "Rajagopal2016.osim")

# FEM 输入数据
FEM_COORDINATES_FILE = "output/output_level1_fem_coordinates.sto"

# RL 策略输出的激活值（你需要准备这个文件）
RL_ACTIVATIONS_FILE = "output/output_level2_rl_activations.sto"

# MocoInverse 配置
MESH_INTERVAL = 0.02       # 网格间隔 (秒)
RESERVE_STRENGTH = 1.0     # Reserve actuator 强度

# 灵敏度分析开关
IGNORE_TENDON_COMPLIANCE = True
IGNORE_PASSIVE_FIBER_FORCE = True
ACTIVE_FORCE_WIDTH_SCALE = 1.5


# ============================================================================
# STEP 1: 运行 MocoInverse
# ============================================================================

def run_moco_inverse(coordinates_file, t0=None, tf=None):
    """
    用 MocoInverse 从关节角度反解肌肉激活值。

    参数：
        coordinates_file: .sto 关节角度文件
        t0, tf: 起止时间（None=自动从文件推断）
    """
    print("=" * 60)
    print("STEP 1: 运行 MocoInverse")
    print("=" * 60)

    if not os.path.exists(MODEL_PATH):
        print(f"  模型不存在: {MODEL_PATH}")
        return None
    if not os.path.exists(coordinates_file):
        print(f"  坐标文件不存在: {coordinates_file}")
        return None

    # 推断时间范围
    coords = osim.TimeSeriesTable(coordinates_file)
    times = coords.getIndependentColumn()
    if t0 is None:
        t0 = times[0]
    if tf is None:
        tf = times[-1]

    # --- 构建 ModelProcessor ---
    modelProcessor = osim.ModelProcessor(MODEL_PATH)

    # 替换肌肉为 DeGrooteFregly2016Muscle（Moco 专用）
    modelProcessor.append(osim.ModOpReplaceMusclesWithDeGrooteFregly2016())

    if IGNORE_TENDON_COMPLIANCE:
        modelProcessor.append(osim.ModOpIgnoreTendonCompliance())

    if IGNORE_PASSIVE_FIBER_FORCE:
        modelProcessor.append(osim.ModOpIgnorePassiveFiberForcesDGF())

    modelProcessor.append(
        osim.ModOpScaleActiveFiberForceCurveWidthDGF(ACTIVE_FORCE_WIDTH_SCALE)
    )

    # 添加 reserve actuators（残差执行器）
    modelProcessor.append(osim.ModOpAddReserves(RESERVE_STRENGTH))

    print(f"  模型: {MODEL_PATH}")
    print(f"  肌肉模型: DeGrooteFregly2016Muscle")
    print(f"  忽略肌腱柔度: {IGNORE_TENDON_COMPLIANCE}")
    print(f"  忽略被动纤维力: {IGNORE_PASSIVE_FIBER_FORCE}")
    print(f"  力-长度曲线宽度: {ACTIVE_FORCE_WIDTH_SCALE}")
    print(f"  Reserve 强度: {RESERVE_STRENGTH}")
    print(f"  时间范围: {t0:.3f} ~ {tf:.3f} s")
    print(f"  网格间隔: {MESH_INTERVAL} s")

    # --- 配置 MocoInverse ---
    inverse = osim.MocoInverse()
    inverse.setModel(modelProcessor)
    inverse.setKinematics(osim.TableProcessor(coordinates_file))
    inverse.set_initial_time(t0)
    inverse.set_final_time(tf)
    inverse.set_mesh_interval(MESH_INTERVAL)
    inverse.set_kinematics_allow_extra_columns(True)

    # --- 求解 ---
    print("\n  正在求解 MocoInverse（可能需要数分钟）...")
    solution = inverse.solve()
    moco_solution = solution.getMocoSolution()

    status = moco_solution.getStatus()
    print(f"  求解状态: {status}")

    # 保存
    output_path = "output/output_level2_inverse_solution.sto"
    moco_solution.write(output_path)
    print(f"  解已保存到 {output_path}")

    # 生成 PDF 报告（如果 matplotlib 可用）
    try:
        model = modelProcessor.process()
        report = osim.report.Report(model, output_path)
        report.generate()
        print("  PDF 报告已生成")
    except Exception as e:
        print(f"  PDF 报告生成跳过: {e}")

    return moco_solution


# ============================================================================
# STEP 2: 生成 demo RL 激活值
# ============================================================================

def demo_generate_rl_activations(moco_solution):
    """
    生成模拟的 RL 激活值数据作为示例。
    实际使用时替换为你的 RL 策略输出。
    """
    print("\n" + "=" * 60)
    print("STEP 2: 生成示例 RL 激活值")
    print("=" * 60)

    if moco_solution is None:
        print("  无 MocoInverse 解，跳过。")
        return

    # 从 MocoInverse 解中提取控制列名
    controls_table = moco_solution.exportToControlsTable()
    col_labels = list(controls_table.getColumnLabels())
    times = np.array(controls_table.getIndependentColumn())

    # 获取 OpenSim 的激活值
    n_cols = len(col_labels)
    n_rows = len(times)
    opensim_activations = np.zeros((n_rows, n_cols))
    for j, label in enumerate(col_labels):
        opensim_activations[:, j] = controls_table.getDependentColumn(label).to_numpy()

    # 生成模拟的 RL 激活值（加噪声的 OpenSim 值）
    np.random.seed(42)
    rl_activations = opensim_activations + 0.1 * np.random.randn(*opensim_activations.shape)
    rl_activations = np.clip(rl_activations, 0.01, 1.0)

    # 保存
    table = osim.TimeSeriesTable()
    labels_vec = osim.StdVectorString()
    for label in col_labels:
        labels_vec.append(label)
    table.setColumnLabels(labels_vec)

    for i in range(n_rows):
        row = osim.RowVector.createFromMat(rl_activations[i])
        table.appendRow(float(times[i]), row)

    osim.STOFileAdapter.write(table, RL_ACTIVATIONS_FILE)
    print(f"  RL 激活值已保存到 {RL_ACTIVATIONS_FILE}")
    print(f"  控制通道: {n_cols}")
    print(f"  肌肉列: {[c for c in col_labels if 'reserve' not in c.lower()][:5]}...")


# ============================================================================
# STEP 3: 对比激活值
# ============================================================================

def compare_activations(moco_solution):
    """
    逐肌肉对比 MocoInverse 解与 RL 策略的激活值。
    """
    print("\n" + "=" * 60)
    print("STEP 3: 对比肌肉激活值")
    print("=" * 60)

    if moco_solution is None:
        print("  无 MocoInverse 解，跳过。")
        return

    if not os.path.exists(RL_ACTIVATIONS_FILE):
        print(f"  RL 激活值文件不存在: {RL_ACTIVATIONS_FILE}")
        return

    # 读取两组激活值
    opensim_controls = moco_solution.exportToControlsTable()
    rl_table = osim.TimeSeriesTable(RL_ACTIVATIONS_FILE)

    opensim_labels = list(opensim_controls.getColumnLabels())
    rl_labels = list(rl_table.getColumnLabels())

    # 找到共同的肌肉（排除 reserve actuators）
    common = [c for c in opensim_labels
              if c in rl_labels and "reserve" not in c.lower()]

    if not common:
        print("  没有找到共同的肌肉名，请检查列名一致性。")
        print(f"  OpenSim 列: {opensim_labels[:5]}...")
        print(f"  RL 列: {rl_labels[:5]}...")
        return

    print(f"  共同肌肉数: {len(common)}")

    # 检查 reserve actuators 的大小
    reserves = [c for c in opensim_labels if "reserve" in c.lower()]
    if reserves:
        print(f"\n  Reserve actuators ({len(reserves)} 个):")
        for r in reserves:
            data = opensim_controls.getDependentColumn(r).to_numpy()
            peak = np.max(np.abs(data))
            print(f"    {r}: 峰值 = {peak:.4f}")

    # 逐肌肉对比
    print(f"\n  {'肌肉':<35} {'RMSE':>8} {'R2':>8} {'相关系数':>8} {'OS峰值':>8} {'RL峰值':>8}")
    print("  " + "-" * 75)

    for muscle in common:
        os_data = opensim_controls.getDependentColumn(muscle).to_numpy()
        rl_data = rl_table.getDependentColumn(muscle).to_numpy()

        n = min(len(os_data), len(rl_data))
        os_data = os_data[:n]
        rl_data = rl_data[:n]

        # RMSE
        rmse = np.sqrt(np.mean((os_data - rl_data) ** 2))

        # R2
        ss_res = np.sum((os_data - rl_data) ** 2)
        ss_tot = np.sum((rl_data - np.mean(rl_data)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 1e-10 else float('nan')

        # Pearson 相关系数
        if np.std(os_data) > 1e-10 and np.std(rl_data) > 1e-10:
            corr = np.corrcoef(os_data, rl_data)[0, 1]
        else:
            corr = float('nan')

        print(f"  {muscle:<35} {rmse:>8.4f} {r2:>8.4f} {corr:>8.4f} "
              f"{np.max(os_data):>8.3f} {np.max(rl_data):>8.3f}")

    # 尝试绘图
    try:
        import matplotlib.pyplot as plt

        n_muscles = min(6, len(common))
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        axes = axes.flatten()

        os_times = np.array(opensim_controls.getIndependentColumn())
        rl_times = np.array(rl_table.getIndependentColumn())

        for i in range(n_muscles):
            muscle = common[i]
            os_data = opensim_controls.getDependentColumn(muscle).to_numpy()
            rl_data = rl_table.getDependentColumn(muscle).to_numpy()

            ax = axes[i]
            ax.plot(os_times[:len(os_data)], os_data, 'b-', linewidth=2,
                    label='OpenSim MocoInverse')
            ax.plot(rl_times[:len(rl_data)], rl_data, 'r--', linewidth=2,
                    label='RL Policy')
            ax.set_title(muscle.split('/')[-1], fontsize=10)
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Activation')
            ax.set_ylim(-0.05, 1.05)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        for i in range(n_muscles, len(axes)):
            axes[i].set_visible(False)

        plt.suptitle('Muscle Activation Comparison: OpenSim vs RL', fontsize=14)
        plt.tight_layout()
        plt.savefig('output/output_level2_comparison.png', dpi=150)
        print(f"\n  对比图已保存到 output/output_level2_comparison.png")
    except ImportError:
        print("\n  matplotlib 未安装，跳过绘图。")


# ============================================================================
# STEP 4: 灵敏度分析辅助函数
# ============================================================================

def run_sensitivity_analysis(coordinates_file, t0, tf):
    """
    切换不同配置运行 MocoInverse，对比结果差异。
    """
    print("\n" + "=" * 60)
    print("STEP 4: 灵敏度分析（可选）")
    print("=" * 60)

    global IGNORE_TENDON_COMPLIANCE, IGNORE_PASSIVE_FIBER_FORCE, ACTIVE_FORCE_WIDTH_SCALE

    configs = [
        ("baseline",             True,  True,  1.5),
        ("with_tendon_compliance", False, True,  1.5),
        ("with_passive_force",   True,  False, 1.5),
        ("narrow_fl_curve",      True,  True,  1.0),
    ]

    print(f"  将运行 {len(configs)} 种配置...")
    print(f"  注意：每种配置可能需要数分钟，总计可能较长。")
    print(f"  如需跳过，请注释掉 run_sensitivity_analysis() 调用。\n")

    for name, itc, ipf, afw in configs:
        print(f"\n  --- 配置: {name} ---")
        IGNORE_TENDON_COMPLIANCE = itc
        IGNORE_PASSIVE_FIBER_FORCE = ipf
        ACTIVE_FORCE_WIDTH_SCALE = afw

        try:
            sol = run_moco_inverse(coordinates_file, t0, tf)
            if sol is not None:
                sol.write(f"output/output_level2_sensitivity_{name}.sto")
                print(f"  → 已保存 output/output_level2_sensitivity_{name}.sto")
        except Exception as e:
            print(f"  → 配置 {name} 失败: {e}")


# ============================================================================
# 主程序
# ============================================================================

if __name__ == "__main__":
    osim.ModelVisualizer.addDirToGeometrySearchPaths(
        os.path.join(os.getcwd(), "Geometry")
    )

    os.makedirs("output", exist_ok=True)

    print("=" * 60)
    print("  Level 2: MocoInverse 肌肉激活值对比")
    print("=" * 60)
    print(f"  OpenSim 版本: {osim.GetVersion()}")
    print(f"  模型: {MODEL_PATH}")

    # 确保有坐标文件（如果没有，先运行 Level 1）
    if not os.path.exists(FEM_COORDINATES_FILE):
        print(f"\n  坐标文件 {FEM_COORDINATES_FILE} 不存在。")
        print(f"  请先运行 validation_level1_joint_torques.py 或准备自己的数据。")
        exit(1)

    # STEP 1: MocoInverse
    solution = run_moco_inverse(FEM_COORDINATES_FILE)

    # STEP 2: 生成/加载 RL 激活值
    if not os.path.exists(RL_ACTIVATIONS_FILE):
        demo_generate_rl_activations(solution)

    # STEP 3: 对比
    compare_activations(solution)

    # STEP 4: 灵敏度分析（取消注释以运行）
    # run_sensitivity_analysis(FEM_COORDINATES_FILE, t0=0.0, tf=2.0)

    print("\n" + "=" * 60)
    print("Level 2 完成！")
    print("下一步：")
    print("  1. 替换 demo 数据为真实 FEM 关节角度和 RL 激活值")
    print("  2. 取消注释 run_sensitivity_analysis() 进行灵敏度分析")
    print("  3. 如使用完整 Rajagopal 模型，修改 MODEL_PATH")
    print("=" * 60)
