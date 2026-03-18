"""
================================================================================
  Level 0：单肌肉标定 —— 力-长度 & 力-速度曲线
================================================================================

目标：用 DeGrooteFregly2016Muscle 生成力-长度、力-速度曲线，
      作为 FEM 体积肌肉的标定基准。

用法：
    python validation_level0_single_muscle.py

输出：
    output_level0_force_length.sto   —— 归一化纤维长度 vs 归一化力
    output_level0_force_velocity.sto —— 归一化纤维速度 vs 归一化力
    output_level0_activation_dynamics.sto —— 阶跃激活下的力响应
    output_level0_force_length.png   —— 力-长度曲线图
    output_level0_force_velocity.png —— 力-速度曲线图
================================================================================
"""

import opensim as osim
import numpy as np
import math
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
# 1. 力-长度关系：扫描归一化纤维长度，测量等长力
# ============================================================================

def compute_force_length_curve():
    """
    固定激活值=1，扫描不同关节角度（从而改变纤维长度），
    记录等长力。
    """
    print("=" * 60)
    print("Level 0.1: 计算力-长度关系")
    print("=" * 60)

    # 构建模型：滑块关节 + 单根肌肉
    model = osim.Model()
    model.setName("force_length_test")
    model.set_gravity(osim.Vec3(0, 0, 0))

    body = osim.Body("body", 1.0, osim.Vec3(0), osim.Inertia(1))
    model.addComponent(body)

    joint = osim.SliderJoint("slider", model.getGround(), body)
    coord = joint.updCoordinate()
    coord.setName("displacement")
    model.addComponent(joint)

    muscle = osim.DeGrooteFregly2016Muscle()
    muscle.setName("muscle")
    muscle.set_max_isometric_force(MUSCLE_PARAMS["max_isometric_force"])
    muscle.set_optimal_fiber_length(MUSCLE_PARAMS["optimal_fiber_length"])
    muscle.set_tendon_slack_length(MUSCLE_PARAMS["tendon_slack_length"])
    muscle.set_pennation_angle_at_optimal(MUSCLE_PARAMS["pennation_angle_at_optimal"])
    muscle.set_fiber_damping(MUSCLE_PARAMS["fiber_damping"])
    muscle.set_max_contraction_velocity(MUSCLE_PARAMS["max_contraction_velocity"])
    muscle.set_tendon_strain_at_one_norm_force(MUSCLE_PARAMS["tendon_strain_at_one_norm_force"])
    muscle.set_ignore_activation_dynamics(True)
    muscle.set_ignore_tendon_compliance(True)  # 刚性肌腱，隔离纤维力-长度关系
    muscle.addNewPathPoint("origin", model.updGround(), osim.Vec3(0))
    muscle.addNewPathPoint("insertion", body, osim.Vec3(0))
    model.addForce(muscle)

    model.finalizeConnections()
    state = model.initSystem()

    Lopt = MUSCLE_PARAMS["optimal_fiber_length"]
    Ltendon = MUSCLE_PARAMS["tendon_slack_length"]

    # 扫描归一化纤维长度 0.4 ~ 1.8
    norm_lengths = np.linspace(0.4, 1.8, 100)
    results = []

    for nL in norm_lengths:
        # 肌肉总长度 = 纤维长度 + 肌腱松弛长度（刚性肌腱）
        total_length = nL * Lopt + Ltendon
        coord.setValue(state, total_length)
        coord.setSpeedValue(state, 0.0)

        # 设置最大激活
        muscle.setActivation(state, 1.0)
        model.equilibrateMuscles(state)
        model.realizeDynamics(state)

        # 读取力
        force = muscle.getActuation(state)
        norm_force = abs(force) / MUSCLE_PARAMS["max_isometric_force"]
        results.append((nL, norm_force))

    # 保存为 .sto
    table = osim.TimeSeriesTable()
    table.setColumnLabels(["normalized_fiber_length", "normalized_force"])
    for i, (nL, nF) in enumerate(results):
        row = osim.RowVector.createFromMat(np.array([nL, nF]))
        table.appendRow(float(i), row)
    osim.STOFileAdapter.write(table, "output_level0_force_length.sto")

    print(f"  已保存 {len(results)} 个数据点到 output_level0_force_length.sto")
    return results


# ============================================================================
# 2. 力-速度关系：在最优纤维长度下，扫描收缩速度
# ============================================================================

def compute_force_velocity_curve():
    """
    固定激活值=1、纤维长度=最优长度，扫描不同收缩速度，记录力。
    使用 DeGrooteFregly2016Muscle 的解析公式直接计算。
    """
    print("\n" + "=" * 60)
    print("Level 0.2: 计算力-速度关系")
    print("=" * 60)

    model = osim.Model()
    model.setName("force_velocity_test")
    model.set_gravity(osim.Vec3(0, 0, 0))

    body = osim.Body("body", 1.0, osim.Vec3(0), osim.Inertia(1))
    model.addComponent(body)

    joint = osim.SliderJoint("slider", model.getGround(), body)
    coord = joint.updCoordinate()
    coord.setName("displacement")
    model.addComponent(joint)

    muscle = osim.DeGrooteFregly2016Muscle()
    muscle.setName("muscle")
    muscle.set_max_isometric_force(MUSCLE_PARAMS["max_isometric_force"])
    muscle.set_optimal_fiber_length(MUSCLE_PARAMS["optimal_fiber_length"])
    muscle.set_tendon_slack_length(MUSCLE_PARAMS["tendon_slack_length"])
    muscle.set_pennation_angle_at_optimal(MUSCLE_PARAMS["pennation_angle_at_optimal"])
    muscle.set_fiber_damping(MUSCLE_PARAMS["fiber_damping"])
    muscle.set_max_contraction_velocity(MUSCLE_PARAMS["max_contraction_velocity"])
    muscle.set_tendon_strain_at_one_norm_force(MUSCLE_PARAMS["tendon_strain_at_one_norm_force"])
    muscle.set_ignore_activation_dynamics(True)
    muscle.set_ignore_tendon_compliance(True)
    muscle.addNewPathPoint("origin", model.updGround(), osim.Vec3(0))
    muscle.addNewPathPoint("insertion", body, osim.Vec3(0))
    model.addForce(muscle)

    model.finalizeConnections()
    state = model.initSystem()

    Lopt = MUSCLE_PARAMS["optimal_fiber_length"]
    Ltendon = MUSCLE_PARAMS["tendon_slack_length"]
    Vmax = MUSCLE_PARAMS["max_contraction_velocity"]

    # 在最优纤维长度下，扫描归一化速度 -1 ~ +1
    norm_velocities = np.linspace(-1.0, 1.0, 100)
    results = []

    total_length = 1.0 * Lopt + Ltendon
    coord.setValue(state, total_length)

    for nV in norm_velocities:
        speed = nV * Vmax * Lopt  # 实际速度 = 归一化速度 × Vmax × Lopt
        coord.setSpeedValue(state, speed)
        muscle.setActivation(state, 1.0)

        model.realizeDynamics(state)
        force = muscle.getActuation(state)
        norm_force = abs(force) / MUSCLE_PARAMS["max_isometric_force"]
        results.append((nV, norm_force))

    # 保存
    table = osim.TimeSeriesTable()
    table.setColumnLabels(["normalized_velocity", "normalized_force"])
    for i, (nV, nF) in enumerate(results):
        row = osim.RowVector.createFromMat(np.array([nV, nF]))
        table.appendRow(float(i), row)
    osim.STOFileAdapter.write(table, "output_level0_force_velocity.sto")

    print(f"  已保存 {len(results)} 个数据点到 output_level0_force_velocity.sto")
    return results


# ============================================================================
# 3. 激活动力学：阶跃输入，观察力响应时间常数
# ============================================================================

def compute_activation_dynamics():
    """
    从激活=0.01 阶跃到激活=1.0，前向仿真并记录力随时间变化。
    """
    print("\n" + "=" * 60)
    print("Level 0.3: 激活动力学响应")
    print("=" * 60)

    model = osim.Model()
    model.setName("activation_dynamics_test")
    model.set_gravity(osim.Vec3(9.81, 0, 0))

    body = osim.Body("body", 0.5, osim.Vec3(0), osim.Inertia(1))
    model.addComponent(body)

    joint = osim.SliderJoint("joint", model.getGround(), body)
    coord = joint.updCoordinate()
    coord.setName("height")
    model.addComponent(joint)

    muscle = osim.DeGrooteFregly2016Muscle()
    muscle.setName("muscle")
    muscle.set_max_isometric_force(MUSCLE_PARAMS["max_isometric_force"])
    muscle.set_optimal_fiber_length(MUSCLE_PARAMS["optimal_fiber_length"])
    muscle.set_tendon_slack_length(MUSCLE_PARAMS["tendon_slack_length"])
    muscle.set_pennation_angle_at_optimal(MUSCLE_PARAMS["pennation_angle_at_optimal"])
    muscle.set_fiber_damping(MUSCLE_PARAMS["fiber_damping"])
    muscle.set_max_contraction_velocity(MUSCLE_PARAMS["max_contraction_velocity"])
    muscle.set_tendon_strain_at_one_norm_force(MUSCLE_PARAMS["tendon_strain_at_one_norm_force"])
    muscle.set_ignore_activation_dynamics(False)  # 开启激活动力学
    muscle.set_ignore_tendon_compliance(False)
    muscle.set_tendon_compliance_dynamics_mode("implicit")
    muscle.addNewPathPoint("origin", model.updGround(), osim.Vec3(0))
    muscle.addNewPathPoint("insertion", body, osim.Vec3(0))
    model.addForce(muscle)

    # 控制器：在 t=0.1 时阶跃到 1.0
    controller = osim.PrescribedController()
    controller.addActuator(muscle)
    controller.prescribeControlForActuator(
        "muscle", osim.StepFunction(0.05, 0.15, 0.01, 1.0)
    )
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

    # 仿真
    manager = osim.Manager(model)
    state.setTime(0)
    manager.initialize(state)
    state = manager.integrate(0.5)

    # 保存
    table = reporter.getTable()
    osim.STOFileAdapter.write(table, "output_level0_activation_dynamics.sto")
    print(f"  已保存 {table.getNumRows()} 行到 output_level0_activation_dynamics.sto")


# ============================================================================
# 4. 绘图（可选）
# ============================================================================

def plot_results(fl_results, fv_results):
    """尝试绘图，如果 matplotlib 不可用则跳过。"""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n  matplotlib 未安装，跳过绘图。")
        return

    # 力-长度曲线
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    lengths = [r[0] for r in fl_results]
    forces = [r[1] for r in fl_results]
    ax.plot(lengths, forces, 'b-', linewidth=2, label='OpenSim DeGrooteFregly2016')
    ax.set_xlabel('Normalized Fiber Length (L/Lopt)')
    ax.set_ylabel('Normalized Force (F/Fmax)')
    ax.set_title('Force-Length Relationship')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.4, 1.8)
    ax.set_ylim(0, 1.2)
    # Overlay your FEM curve here
    ax.text(0.5, 1.1, '<-- Overlay FEM force-length curve here', fontsize=10, color='gray')
    plt.tight_layout()
    plt.savefig('output_level0_force_length.png', dpi=150)
    print("  已保存 output_level0_force_length.png")

    # 力-速度曲线
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    velocities = [r[0] for r in fv_results]
    forces = [r[1] for r in fv_results]
    ax.plot(velocities, forces, 'r-', linewidth=2, label='OpenSim DeGrooteFregly2016')
    ax.set_xlabel('Normalized Fiber Velocity (V/Vmax)')
    ax.set_ylabel('Normalized Force (F/Fmax)')
    ax.set_title('Force-Velocity Relationship')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.text(-0.9, 1.5, '<-- Overlay FEM force-velocity curve here', fontsize=10, color='gray')
    plt.tight_layout()
    plt.savefig('output_level0_force_velocity.png', dpi=150)
    print("  已保存 output_level0_force_velocity.png")


# ============================================================================
# 主程序
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  Level 0: 单肌肉标定 (DeGrooteFregly2016Muscle)")
    print("=" * 60)
    print(f"  OpenSim 版本: {osim.GetVersion()}")
    print(f"  肌肉参数: {MUSCLE_PARAMS}")

    fl = compute_force_length_curve()
    fv = compute_force_velocity_curve()
    compute_activation_dynamics()
    plot_results(fl, fv)

    print("\n" + "=" * 60)
    print("Level 0 完成！")
    print("下一步：将 FEM 体积肌肉在相同条件下的曲线叠加到图上进行对比。")
    print("=" * 60)
