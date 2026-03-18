"""
================================================================================
  Level 0：单肌肉标定 —— 力-长度 & 力-速度曲线 & 激活动力学
================================================================================

目标：用 DeGrooteFregly2016Muscle 生成力-长度、力-速度、激活动力学曲线，
      作为 FEM 体积肌肉的标定基准。

用法：
    python validation_level0_single_muscle.py

输出：
    output/output_level0_force_length.sto          —— 归一化纤维长度 vs 归一化力
    output/output_level0_force_length_states.sto   —— 坐标状态（配合 .osim 在 GUI 中播放）
    output/output_level0_force_velocity.sto        —— 归一化纤维速度 vs 归一化力
    output/output_level0_activation_dynamics.sto   —— 阶跃激活下的力响应
    output/output_level0.osim                      —— 带小球的模型
    output/output_level0_force_length.png          —— 力-长度曲线图
    output/output_level0_force_velocity.png        —— 力-速度曲线图
    output/output_level0_activation_dynamics.png   —— 激活动力学曲线图
================================================================================
"""

import opensim as osim
import numpy as np
import os

# ============================================================================
# 配置区：修改以下参数以匹配你的 FEM 体积肌肉
# ============================================================================

MUSCLE_PARAMS = {
    "max_isometric_force": 1000.0,       # 峰值应力 × PCSA (N)
    "optimal_fiber_length": 0.10,        # 沿纤维方向的静息长度 (m)
    "tendon_slack_length": 0.20,         # 肌腱松弛长度 (m)
    "pennation_angle_at_optimal": 0.0,   # 羽状角 (rad)
    "fiber_damping": 0.01,               # 纤维阻尼
    "max_contraction_velocity": 10.0,    # 最大收缩速度 (Lopt/s)
    "tendon_strain_at_one_norm_force": 0.049,  # 肌腱应变
}


# ============================================================================
# 模型构建
# ============================================================================

def build_model(*, ignore_activation_dynamics=True, ignore_tendon_compliance=True,
                tendon_compliance_dynamics_mode="explicit"):
    """构建滑块关节 + 单根 DeGrooteFregly2016Muscle 模型。"""
    model = osim.Model()
    model.setName("single_muscle_test")
    model.set_gravity(osim.Vec3(0, 0, 0))

    body = osim.Body("body", 1.0, osim.Vec3(0), osim.Inertia(1))
    body.attachGeometry(osim.Sphere(0.05))
    model.addBody(body)

    joint = osim.SliderJoint("slider", model.getGround(), body)
    coord = joint.updCoordinate()
    coord.setName("displacement")
    model.addJoint(joint)

    muscle = osim.DeGrooteFregly2016Muscle()
    muscle.setName("muscle")
    muscle.set_max_isometric_force(MUSCLE_PARAMS["max_isometric_force"])
    muscle.set_optimal_fiber_length(MUSCLE_PARAMS["optimal_fiber_length"])
    muscle.set_tendon_slack_length(MUSCLE_PARAMS["tendon_slack_length"])
    muscle.set_pennation_angle_at_optimal(MUSCLE_PARAMS["pennation_angle_at_optimal"])
    muscle.set_fiber_damping(MUSCLE_PARAMS["fiber_damping"])
    muscle.set_max_contraction_velocity(MUSCLE_PARAMS["max_contraction_velocity"])
    muscle.set_tendon_strain_at_one_norm_force(MUSCLE_PARAMS["tendon_strain_at_one_norm_force"])
    muscle.set_ignore_activation_dynamics(ignore_activation_dynamics)
    muscle.set_ignore_tendon_compliance(ignore_tendon_compliance)
    if not ignore_tendon_compliance:
        muscle.set_tendon_compliance_dynamics_mode(tendon_compliance_dynamics_mode)
    muscle.addNewPathPoint("origin", model.updGround(), osim.Vec3(0))
    muscle.addNewPathPoint("insertion", body, osim.Vec3(0))
    model.addForce(muscle)

    return model, coord, muscle


# ============================================================================
# 1. 力-长度关系
# ============================================================================

def compute_force_length_curve():
    """固定激活值=1，扫描纤维长度，记录等长力。"""
    print("=" * 60)
    print("Level 0.1: 计算力-长度关系")
    print("=" * 60)

    model, coord, muscle = build_model(
        ignore_activation_dynamics=True, ignore_tendon_compliance=True)

    model.finalizeConnections()
    model.printToXML("output/output_level0.osim")
    state = model.initSystem()

    Lopt = MUSCLE_PARAMS["optimal_fiber_length"]
    Ltendon = MUSCLE_PARAMS["tendon_slack_length"]

    norm_lengths = np.linspace(0.4, 1.8, 100)

    # 数据 .sto
    table = osim.TimeSeriesTable()
    table.setColumnLabels(["normalized_fiber_length", "normalized_force"])

    # 状态 .sto（驱动 GUI 动画）
    states_table = osim.TimeSeriesTable()
    states_table.setColumnLabels(["/slider/displacement/value"])
    states_table.addTableMetaDataString("inDegrees", "no")

    for i, nL in enumerate(norm_lengths):
        total_length = nL * Lopt + Ltendon
        coord.setValue(state, total_length)
        coord.setSpeedValue(state, 0.0)
        muscle.setActivation(state, 1.0)
        model.equilibrateMuscles(state)
        model.realizeDynamics(state)

        norm_force = abs(muscle.getActuation(state)) / MUSCLE_PARAMS["max_isometric_force"]

        table.appendRow(float(i),
                        osim.RowVector.createFromMat(np.array([nL, norm_force])))
        states_table.appendRow(float(i) / len(norm_lengths),
                               osim.RowVector.createFromMat(np.array([total_length])))

    osim.STOFileAdapter.write(table, "output/output_level0_force_length.sto")
    osim.STOFileAdapter.write(states_table, "output/output_level0_force_length_states.sto")
    print(f"  已保存到 output/output_level0_force_length.sto / _states.sto / output/output_level0.osim")


# ============================================================================
# 2. 力-速度关系
# ============================================================================

def compute_force_velocity_curve():
    """固定激活值=1、纤维长度=最优长度，扫描收缩速度，记录力。"""
    print("\n" + "=" * 60)
    print("Level 0.2: 计算力-速度关系")
    print("=" * 60)

    model, coord, muscle = build_model(
        ignore_activation_dynamics=True, ignore_tendon_compliance=True)

    model.finalizeConnections()
    state = model.initSystem()

    Lopt = MUSCLE_PARAMS["optimal_fiber_length"]
    Ltendon = MUSCLE_PARAMS["tendon_slack_length"]
    Vmax = MUSCLE_PARAMS["max_contraction_velocity"]

    norm_velocities = np.linspace(-1.0, 1.0, 100)
    coord.setValue(state, Lopt + Ltendon)

    table = osim.TimeSeriesTable()
    table.setColumnLabels(["normalized_velocity", "normalized_force"])

    for i, nV in enumerate(norm_velocities):
        coord.setSpeedValue(state, nV * Vmax * Lopt)
        muscle.setActivation(state, 1.0)
        model.realizeDynamics(state)

        norm_force = abs(muscle.getActuation(state)) / MUSCLE_PARAMS["max_isometric_force"]
        table.appendRow(float(i),
                        osim.RowVector.createFromMat(np.array([nV, norm_force])))

    osim.STOFileAdapter.write(table, "output/output_level0_force_velocity.sto")
    print(f"  已保存到 output/output_level0_force_velocity.sto")


# ============================================================================
# 3. 激活动力学：阶跃输入，观察力响应时间常数
# ============================================================================

def compute_activation_dynamics():
    """从激活=0.01 阶跃到激活=1.0，前向仿真并记录力随时间变化。"""
    print("\n" + "=" * 60)
    print("Level 0.3: 激活动力学响应")
    print("=" * 60)

    model, coord, muscle = build_model(
        ignore_activation_dynamics=False, ignore_tendon_compliance=False,
        tendon_compliance_dynamics_mode="explicit")

    # 控制器：阶跃激励 0.01 -> 1.0
    controller = osim.PrescribedController()
    controller.addActuator(muscle)
    controller.prescribeControlForActuator(
        "muscle", osim.StepFunction(0.05, 0.15, 0.01, 1.0))
    model.addController(controller)

    # Reporter
    reporter = osim.TableReporter()
    reporter.setName("reporter")
    reporter.set_report_time_interval(0.001)
    reporter.addToReport(muscle.getOutput("actuation"), "muscle_force")
    reporter.addToReport(muscle.getOutput("fiber_length"), "fiber_length")
    model.addComponent(reporter)

    model.finalizeConnections()
    state = model.initSystem()

    Lopt = MUSCLE_PARAMS["optimal_fiber_length"]
    Ltendon = MUSCLE_PARAMS["tendon_slack_length"]
    coord.setValue(state, Lopt + Ltendon)
    coord.setLocked(state, True)
    muscle.setActivation(state, 0.01)
    model.equilibrateMuscles(state)

    # 前向仿真
    manager = osim.Manager(model)
    state.setTime(0)
    manager.initialize(state)
    state = manager.integrate(0.5)

    table = reporter.getTable()
    osim.STOFileAdapter.write(table, "output/output_level0_activation_dynamics.sto")
    print(f"  已保存 {table.getNumRows()} 行到 output/output_level0_activation_dynamics.sto")


# ============================================================================
# 4. 绘图（从 .sto 文件读取）
# ============================================================================

def plot_results():
    """从 .sto 文件读取数据并绘图。"""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n  matplotlib 未安装，跳过绘图。")
        return

    # 力-长度曲线
    fl_table = osim.TimeSeriesTable("output/output_level0_force_length.sto")
    lengths = np.array(fl_table.getDependentColumn("normalized_fiber_length").to_numpy())
    forces = np.array(fl_table.getDependentColumn("normalized_force").to_numpy())

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.plot(lengths, forces, 'b-', linewidth=2, label='OpenSim DeGrooteFregly2016')
    ax.set_xlabel('Normalized Fiber Length (L/Lopt)')
    ax.set_ylabel('Normalized Force (F/Fmax)')
    ax.set_title('Force-Length Relationship')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.4, 1.8)
    ax.set_ylim(0, 1.2)
    ax.text(0.5, 1.1, '<-- Overlay FEM force-length curve here', fontsize=10, color='gray')
    plt.tight_layout()
    plt.savefig('output/output_level0_force_length.png', dpi=150)
    print("  已保存 output/output_level0_force_length.png")

    # 力-速度曲线
    fv_table = osim.TimeSeriesTable("output/output_level0_force_velocity.sto")
    velocities = np.array(fv_table.getDependentColumn("normalized_velocity").to_numpy())
    forces = np.array(fv_table.getDependentColumn("normalized_force").to_numpy())

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.plot(velocities, forces, 'r-', linewidth=2, label='OpenSim DeGrooteFregly2016')
    ax.set_xlabel('Normalized Fiber Velocity (V/Vmax)')
    ax.set_ylabel('Normalized Force (F/Fmax)')
    ax.set_title('Force-Velocity Relationship')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.text(-0.9, 1.5, '<-- Overlay FEM force-velocity curve here', fontsize=10, color='gray')
    plt.tight_layout()
    plt.savefig('output/output_level0_force_velocity.png', dpi=150)
    print("  已保存 output/output_level0_force_velocity.png")

    # 激活动力学响应
    ad_table = osim.TimeSeriesTable("output/output_level0_activation_dynamics.sto")
    times = np.array([ad_table.getIndependentColumn()[i]
                       for i in range(ad_table.getNumRows())])
    muscle_force = np.array(ad_table.getDependentColumn("muscle_force").to_numpy())
    fiber_length = np.array(ad_table.getDependentColumn("fiber_length").to_numpy())

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    ax1.plot(times * 1000, abs(muscle_force), 'b-', linewidth=2)
    ax1.set_ylabel('Muscle Force (N)')
    ax1.set_title('Activation Dynamics Step Response')
    ax1.grid(True, alpha=0.3)

    ax2.plot(times * 1000, fiber_length * 1000, 'g-', linewidth=2)
    ax2.set_xlabel('Time (ms)')
    ax2.set_ylabel('Fiber Length (mm)')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('output/output_level0_activation_dynamics.png', dpi=150)
    print("  已保存 output/output_level0_activation_dynamics.png")

    plt.show()


# ============================================================================
# 主程序
# ============================================================================

if __name__ == "__main__":
    os.makedirs("output", exist_ok=True)

    print("=" * 60)
    print("  Level 0: 单肌肉标定 (DeGrooteFregly2016Muscle)")
    print("=" * 60)
    print(f"  OpenSim 版本: {osim.GetVersion()}")
    print(f"  肌肉参数: {MUSCLE_PARAMS}")

    compute_force_length_curve()
    compute_force_velocity_curve()
    compute_activation_dynamics()
    plot_results()

    print("\n" + "=" * 60)
    print("Level 0 完成！")
    print("下一步：将 FEM 体积肌肉在相同条件下的曲线叠加到图上进行对比。")
    print("=" * 60)
