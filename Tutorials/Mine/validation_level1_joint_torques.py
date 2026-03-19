"""
================================================================================
  Level 1：关节力矩对比 —— FEM 数据导入 + InverseDynamics
================================================================================

目标：将 FEM 仿真的关节角度导入 OpenSim，用 InverseDynamics 计算关节力矩，
      与 FEM 直接计算的关节力矩进行对比。

用法：
    1. 先准备 FEM 数据（见 STEP 1 的 demo_generate_fem_data）
    2. python validation_level1_joint_torques.py

输出：
    output/output_level1_fem_coordinates.sto  —— FEM 关节角度（OpenSim 格式）
    output/output_level1_id_results/          —— InverseDynamics 输出（关节力矩）
================================================================================
"""

import opensim as osim
import numpy as np
import math
import os


# ============================================================================
# 配置区
# ============================================================================

# 脚本目录和项目根目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

# 使用的 OpenSim 模型（可改为 Rajagopal 等）
MODEL_PATH = os.path.join(PROJECT_ROOT, "Models", "Gait10dof18musc", "gait10dof18musc.osim")

# FEM 数据文件路径（如果已有，设为你的文件路径；否则用 demo 生成）
FEM_COORDINATES_FILE = "output/output_level1_fem_coordinates.sto"
FEM_TORQUES_FILE = "output/output_level1_fem_torques.sto"

# InverseDynamics 结果目录
ID_RESULTS_DIR = "output/output_level1_id_results"


# ============================================================================
# STEP 1: FEM 数据 → OpenSim .sto 格式
# ============================================================================

def convert_fem_to_sto(fem_times, fem_angles, coord_names, output_path,
                       in_degrees=False):
    """
    将 FEM 仿真输出的关节角度转换为 OpenSim .sto 文件。

    参数：
        fem_times:    np.array, shape (N,)        —— 时间序列 (秒)
        fem_angles:   np.array, shape (N, n_dof)  —— 关节角度
        coord_names:  list[str]                   —— 自由度名称，需与 .osim 模型一致
        output_path:  str                         —— 输出 .sto 文件路径
        in_degrees:   bool                        —— 角度是否为度（False=弧度）
    """
    table = osim.TimeSeriesTable()
    labels = osim.StdVectorString()
    for name in coord_names:
        labels.append(name)
    table.setColumnLabels(labels)

    for i in range(len(fem_times)):
        row = osim.RowVector.createFromMat(fem_angles[i].astype(float))
        table.appendRow(float(fem_times[i]), row)

    table.addTableMetaDataString("inDegrees", "yes" if in_degrees else "no")
    table.addTableMetaDataString("nRows", str(len(fem_times)))
    table.addTableMetaDataString("nColumns", str(len(coord_names)))

    osim.STOFileAdapter.write(table, output_path)
    print(f"  FEM 关节角度已保存到 {output_path}")
    print(f"  时间范围: {fem_times[0]:.3f} ~ {fem_times[-1]:.3f} s")
    print(f"  自由度: {coord_names}")


def demo_generate_fem_data():
    """
    生成模拟的 FEM 数据作为示例。
    用 InverseDynamics 计算物理一致的力矩，再加小噪声模拟 FEM 偏差。
    实际使用时替换为你的 FEM 仿真输出。
    """
    print("=" * 60)
    print("STEP 1: 生成示例 FEM 数据")
    print("=" * 60)

    # 加载模型以获取自由度名称
    if not os.path.exists(MODEL_PATH):
        print(f"  模型不存在: {MODEL_PATH}")
        return None, None, None

    model = osim.Model(MODEL_PATH)
    state = model.initSystem()

    coord_set = model.getCoordinateSet()
    coord_names = []
    for i in range(coord_set.getSize()):
        coord_names.append(coord_set.get(i).getName())

    n_dof = len(coord_names)
    print(f"  模型自由度: {n_dof}")
    print(f"  坐标名: {coord_names}")

    # 生成简单正弦运动 (2 秒, 100Hz)
    dt = 0.01
    times = np.arange(0, 2.0, dt)
    n_frames = len(times)
    angles = np.zeros((n_frames, n_dof))

    # 只对前几个自由度添加小幅度正弦运动
    for j in range(min(3, n_dof)):
        coord = coord_set.get(j)
        mid = (coord.getRangeMin() + coord.getRangeMax()) / 2
        amp = (coord.getRangeMax() - coord.getRangeMin()) * 0.05  # 5% 幅度
        angles[:, j] = mid + amp * np.sin(2 * math.pi * 0.5 * times)

    # 保存关节角度
    convert_fem_to_sto(times, angles, coord_names, FEM_COORDINATES_FILE)

    # 用 InverseDynamics 计算物理一致的力矩作为基准
    print("  运行 InverseDynamics 生成物理一致的基准力矩...")
    temp_id_dir = os.path.join("output", "_temp_demo_id")
    os.makedirs(temp_id_dir, exist_ok=True)

    id_tool = osim.InverseDynamicsTool()
    id_tool.setModelFileName(MODEL_PATH)
    id_tool.setCoordinatesFileName(FEM_COORDINATES_FILE)
    id_tool.setLowpassCutoffFrequency(6.0)
    id_tool.setStartTime(float(times[0]))
    id_tool.setEndTime(float(times[-1]))
    id_tool.setResultsDir(temp_id_dir)
    id_tool.setOutputGenForceFileName("temp_id.sto")
    id_tool.run()

    # 读取 ID 力矩，映射回坐标名
    id_output_path = os.path.join(temp_id_dir, "temp_id.sto")
    id_table = osim.TimeSeriesTable(id_output_path)
    id_times = np.array(id_table.getIndependentColumn())
    id_labels = list(id_table.getColumnLabels())

    n_id = len(id_times)
    torques = np.zeros((n_id, n_dof))
    for j, coord_name in enumerate(coord_names):
        for suffix in ["_moment", "_force", ""]:
            id_col = coord_name + suffix
            if id_col in id_labels:
                torques[:, j] = id_table.getDependentColumn(id_col).to_numpy()
                break

    # 加 5% 高斯噪声模拟 FEM 与 OpenSim 的差异
    np.random.seed(42)
    for j in range(n_dof):
        scale = max(np.std(torques[:, j]), 1e-6)
        torques[:, j] += 0.05 * scale * np.random.randn(n_id)

    convert_fem_to_sto(id_times, torques, coord_names, FEM_TORQUES_FILE)
    print(f"  FEM 力矩已保存到 {FEM_TORQUES_FILE}")
    print(f"  （基于 ID 基准 + 5% 噪声）")

    # 清理临时文件
    import shutil
    shutil.rmtree(temp_id_dir, ignore_errors=True)

    return id_times, torques, coord_names


# ============================================================================
# STEP 2: 运行 InverseDynamics
# ============================================================================

def run_inverse_dynamics():
    """
    用 OpenSim InverseDynamicsTool 从关节角度计算关节力矩。
    """
    print("\n" + "=" * 60)
    print("STEP 2: 运行 InverseDynamics")
    print("=" * 60)

    if not os.path.exists(MODEL_PATH):
        print(f"  模型不存在: {MODEL_PATH}")
        return None

    if not os.path.exists(FEM_COORDINATES_FILE):
        print(f"  坐标文件不存在: {FEM_COORDINATES_FILE}")
        return None

    os.makedirs(ID_RESULTS_DIR, exist_ok=True)

    # 读取坐标文件确定时间范围
    coords_table = osim.TimeSeriesTable(FEM_COORDINATES_FILE)
    times = coords_table.getIndependentColumn()
    t0 = times[0]
    tf = times[-1]

    # 配置 InverseDynamicsTool
    id_tool = osim.InverseDynamicsTool()
    id_tool.setModelFileName(MODEL_PATH)
    id_tool.setCoordinatesFileName(FEM_COORDINATES_FILE)
    id_tool.setLowpassCutoffFrequency(6.0)  # 6 Hz 低通滤波
    id_tool.setStartTime(t0)
    id_tool.setEndTime(tf)
    id_tool.setResultsDir(ID_RESULTS_DIR)
    id_tool.setOutputGenForceFileName("inverse_dynamics.sto")

    print(f"  模型: {MODEL_PATH}")
    print(f"  坐标: {FEM_COORDINATES_FILE}")
    print(f"  时间范围: {t0:.3f} ~ {tf:.3f} s")
    print(f"  低通滤波: 6 Hz")
    print(f"  输出目录: {ID_RESULTS_DIR}")

    # 保存配置（可复现）
    id_tool.printToXML(os.path.join(ID_RESULTS_DIR, "id_setup.xml"))

    # 运行
    print("  正在运行 InverseDynamics...")
    id_tool.run()

    id_output = os.path.join(ID_RESULTS_DIR, "inverse_dynamics.sto")
    if os.path.exists(id_output):
        print(f"  InverseDynamics 完成，结果: {id_output}")
        return id_output
    else:
        print("  InverseDynamics 运行失败")
        return None


# ============================================================================
# STEP 3: 对比关节力矩
# ============================================================================

def compare_joint_torques(id_output_path):
    """
    对比 OpenSim ID 计算的关节力矩 vs FEM 关节力矩。
    计算 RMSE 和 R2 指标。
    """
    print("\n" + "=" * 60)
    print("STEP 3: 对比关节力矩")
    print("=" * 60)

    if id_output_path is None or not os.path.exists(id_output_path):
        print("  ID 输出不存在，跳过对比。")
        return

    if not os.path.exists(FEM_TORQUES_FILE):
        print(f"  FEM 力矩文件不存在: {FEM_TORQUES_FILE}")
        return

    # 读取 OpenSim ID 结果
    id_table = osim.TimeSeriesTable(id_output_path)
    id_times = np.array(id_table.getIndependentColumn())
    id_labels = list(id_table.getColumnLabels())

    # 读取 FEM 力矩
    fem_table = osim.TimeSeriesTable(FEM_TORQUES_FILE)
    fem_times = np.array(fem_table.getIndependentColumn())
    fem_labels = list(fem_table.getColumnLabels())

    print(f"  OpenSim ID 列: {id_labels[:5]}...")
    print(f"  FEM 力矩列: {fem_labels[:5]}...")

    # 找到共同的列（自由度名）
    # OpenSim ID 输出列名格式为 "coord_name_moment" 或 "coord_name_force"
    # FEM 数据列名为 "coord_name"，需要做模糊匹配
    common_cols = []  # (fem_col, id_col) 对
    for fem_col in fem_labels:
        # 直接匹配
        if fem_col in id_labels:
            common_cols.append((fem_col, fem_col))
            continue
        # 尝试 _moment / _force 后缀
        for suffix in ["_moment", "_force"]:
            id_col = fem_col + suffix
            if id_col in id_labels:
                common_cols.append((fem_col, id_col))
                break

    print(f"  匹配的自由度: {len(common_cols)}")
    for fem_col, id_col in common_cols:
        print(f"    FEM: {fem_col}  <->  ID: {id_col}")

    if not common_cols:
        print("  没有找到共同的列名，请检查坐标命名一致性。")
        return

    # 对比每个共同自由度
    print(f"\n  {'自由度':<25} {'RMSE (Nm)':>12} {'R2':>8} {'峰值比':>8}")
    print("  " + "-" * 55)

    for fem_col, id_col in common_cols:
        id_data = id_table.getDependentColumn(id_col).to_numpy()
        fem_data = fem_table.getDependentColumn(fem_col).to_numpy()
        col = fem_col

        # 简单对齐：取较短的长度
        n = min(len(id_data), len(fem_data))
        id_data = id_data[:n]
        fem_data = fem_data[:n]

        # RMSE
        rmse = np.sqrt(np.mean((id_data - fem_data) ** 2))

        # R2
        ss_res = np.sum((id_data - fem_data) ** 2)
        ss_tot = np.sum((fem_data - np.mean(fem_data)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 1e-10 else float('nan')

        # 峰值比
        peak_id = np.max(np.abs(id_data))
        peak_fem = np.max(np.abs(fem_data))
        peak_ratio = peak_id / peak_fem if peak_fem > 1e-10 else float('nan')

        print(f"  {col:<25} {rmse:>12.4f} {r2:>8.4f} {peak_ratio:>8.3f}")


# ============================================================================
# 主程序
# ============================================================================

if __name__ == "__main__":
    osim.ModelVisualizer.addDirToGeometrySearchPaths(
        os.path.join(PROJECT_ROOT, "Geometry")
    )

    os.makedirs("output", exist_ok=True)

    print("=" * 60)
    print("  Level 1: 关节力矩对比 (InverseDynamics)")
    print("=" * 60)
    print(f"  OpenSim 版本: {osim.GetVersion()}")

    # copy model to ouput dir
    import shutil
    shutil.copy(MODEL_PATH, os.path.join("output", os.path.basename(MODEL_PATH)))

    # STEP 1: 生成/导入 FEM 数据
    # 实际使用时：替换 demo_generate_fem_data() 为你的 FEM 数据加载代码
    demo_generate_fem_data()

    # STEP 2: 运行 InverseDynamics
    id_output = run_inverse_dynamics()

    # STEP 3: 对比
    compare_joint_torques(id_output)

    print("\n" + "=" * 60)
    print("Level 1 完成！")
    print("下一步：将 demo 数据替换为真实 FEM 仿真数据，运行对比。")
    print("=" * 60)
