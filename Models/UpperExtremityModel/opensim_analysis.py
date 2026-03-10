"""
使用 OpenSim Python API 分析上肢模型

安装 opensim:
    conda install -c opensim-org opensim
    (或参考 https://opensim-gui.github.io/opensim-documentation/md_doc_developer_PythonSetupGuide.html)

.osim 文件获取方式:
    1. 安装 OpenSim GUI 后，在 Models 目录下找到 UpperExtremity.osim
    2. 或从 SimTK 下载: https://simtk.org/projects/up-ext-model
    3. 或通过 OpenSim GUI 将 .jnt + .msl 导入并另存为 .osim
"""

import opensim as osim
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# ============================================================
# 修改此路径指向你的 .osim 文件
# ============================================================
MODEL_PATH = "upper_extremity.osim"


def load_model(path: str) -> osim.Model:
    """加载 OpenSim 模型。"""
    model = osim.Model(path)
    model.initSystem()
    print(f"模型加载成功: {model.getName()}")
    print(f"  自由度: {model.getNumCoordinates()}")
    print(f"  刚体: {model.getNumBodies()}")
    print(f"  肌肉: {model.getMuscles().getSize()}")
    print(f"  关节: {model.getNumJoints()}")
    return model


def list_coordinates(model: osim.Model):
    """列出所有自由度（广义坐标）及其范围。"""
    print("\n" + "=" * 60)
    print("自由度 (Coordinates)")
    print("=" * 60)
    coords = []
    for i in range(model.getNumCoordinates()):
        coord = model.getCoordinateSet().get(i)
        name = coord.getName()
        range_min = np.degrees(coord.getRangeMin())
        range_max = np.degrees(coord.getRangeMax())
        default = np.degrees(coord.getDefaultValue())
        coords.append({
            "name": name,
            "default_deg": round(default, 1),
            "min_deg": round(range_min, 1),
            "max_deg": round(range_max, 1),
        })
        print(f"  {name}: [{range_min:.1f}, {range_max:.1f}] deg, default={default:.1f}")
    return pd.DataFrame(coords)


def list_muscles(model: osim.Model) -> pd.DataFrame:
    """提取所有肌肉的力学参数。"""
    print("\n" + "=" * 60)
    print("肌肉参数")
    print("=" * 60)
    muscles = []
    for i in range(model.getMuscles().getSize()):
        m = model.getMuscles().get(i)
        info = {
            "name": m.getName(),
            "max_force_N": m.getMaxIsometricForce(),
            "optimal_fiber_length_m": m.getOptimalFiberLength(),
            "tendon_slack_length_m": m.getTendonSlackLength(),
            "pennation_angle_deg": np.degrees(m.getPennationAngleAtOptimalFiberLength()),
        }
        muscles.append(info)
    df = pd.DataFrame(muscles)
    print(df.to_string(index=False))
    return df


def list_bodies(model: osim.Model):
    """列出所有刚体及其质量。"""
    print("\n" + "=" * 60)
    print("刚体 (Bodies)")
    print("=" * 60)
    bodies = []
    for i in range(model.getNumBodies()):
        body = model.getBodySet().get(i)
        bodies.append({
            "name": body.getName(),
            "mass_kg": body.getMass(),
        })
        print(f"  {body.getName()}: {body.getMass():.4f} kg")
    return pd.DataFrame(bodies)


def compute_moment_arms(model: osim.Model, coord_name: str):
    """
    计算指定自由度下所有肌肉的力臂。
    在自由度的活动范围内等间距采样。
    """
    state = model.initSystem()
    coord = model.getCoordinateSet().get(coord_name)

    angles = np.linspace(coord.getRangeMin(), coord.getRangeMax(), 50)
    muscle_names = [model.getMuscles().get(i).getName()
                    for i in range(model.getMuscles().getSize())]

    moment_arms = {name: [] for name in muscle_names}

    for angle in angles:
        coord.setValue(state, angle)
        model.realizeVelocity(state)
        for i, name in enumerate(muscle_names):
            muscle = model.getMuscles().get(i)
            ma = muscle.computeMomentArm(state, coord)
            moment_arms[name].append(ma * 100)  # 转换为 cm

    # 筛选有显著力臂的肌肉
    angles_deg = np.degrees(angles)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_title(f"Moment Arms about '{coord_name}'")
    ax.set_xlabel(f"{coord_name} (deg)")
    ax.set_ylabel("Moment Arm (cm)")

    for name in muscle_names:
        ma_arr = np.array(moment_arms[name])
        if np.max(np.abs(ma_arr)) > 0.2:  # 只显示力臂 > 2mm 的肌肉
            ax.plot(angles_deg, ma_arr, label=name)

    ax.legend(fontsize=7, ncol=3, loc="best")
    ax.axhline(y=0, color="gray", linewidth=0.5)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"moment_arms_{coord_name}.png", dpi=150)
    print(f"\n力臂图已保存: moment_arms_{coord_name}.png")
    plt.show()


def plot_muscle_analysis(df: pd.DataFrame):
    """肌肉参数可视化。"""
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    fig.suptitle("Upper Extremity Model - Muscle Analysis")

    # 最大力
    ax = axes[0]
    sorted_df = df.sort_values("max_force_N", ascending=True)
    ax.barh(sorted_df["name"], sorted_df["max_force_N"], color="steelblue")
    ax.set_xlabel("Max Isometric Force (N)")
    ax.tick_params(axis="y", labelsize=5)

    # 纤维长度 vs 肌腱长度
    ax = axes[1]
    sc = ax.scatter(
        df["optimal_fiber_length_m"] * 100,
        df["tendon_slack_length_m"] * 100,
        c=df["max_force_N"], cmap="plasma", s=40,
    )
    ax.set_xlabel("Optimal Fiber Length (cm)")
    ax.set_ylabel("Tendon Slack Length (cm)")
    plt.colorbar(sc, ax=ax, label="Max Force (N)")

    # 羽状角
    ax = axes[2]
    ax.hist(df["pennation_angle_deg"], bins=12, color="coral", edgecolor="white")
    ax.set_xlabel("Pennation Angle (deg)")
    ax.set_ylabel("Count")

    plt.tight_layout()
    plt.savefig("muscle_analysis_opensim.png", dpi=150)
    print("图表已保存: muscle_analysis_opensim.png")
    plt.show()


def run_forward_simulation(model: osim.Model, duration: float = 0.5):
    """运行简单的前向动力学仿真（自由落体/重力驱动）。"""
    state = model.initSystem()

    # 设置初始姿态（例如肘关节屈曲 90°）
    coord = model.getCoordinateSet()
    for i in range(coord.getSize()):
        c = coord.get(i)
        if "elbow_flexion" in c.getName():
            c.setValue(state, np.radians(90))

    # 创建仿真器
    manager = osim.Manager(model)
    manager.setIntegratorAccuracy(1e-5)
    state.setTime(0)
    manager.initialize(state)
    state = manager.integrate(duration)

    print(f"\n前向仿真完成: {duration}s")
    print("最终关节角度:")
    for i in range(coord.getSize()):
        c = coord.get(i)
        val = np.degrees(c.getValue(state))
        print(f"  {c.getName()}: {val:.2f} deg")


# ============================================================
# 主程序
# ============================================================
def main():
    print("=" * 60)
    print("Stanford VA Upper Extremity Model - OpenSim 分析")
    print("=" * 60)

    model = load_model(MODEL_PATH)

    # 基本信息
    coord_df = list_coordinates(model)
    muscle_df = list_muscles(model)
    body_df = list_bodies(model)

    # 保存参数到 CSV
    muscle_df.to_csv("muscle_params_opensim.csv", index=False, encoding="utf-8-sig")
    coord_df.to_csv("coordinates_opensim.csv", index=False, encoding="utf-8-sig")
    print("\n参数已保存为 CSV 文件")

    # 可视化
    plot_muscle_analysis(muscle_df)

    # 力臂分析
    print("\n计算肘关节力臂...")
    compute_moment_arms(model, "elbow_flexion")

    # 可选: 前向仿真
    # run_forward_simulation(model, duration=0.5)

    print("\n分析完成！")


if __name__ == "__main__":
    main()
