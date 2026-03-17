"""
================================================================================
  OpenSim Python API 教程
================================================================================

本教程基于 OpenSim 4.5.2，系统性地讲解如何用 Python 进行生物力学建模与仿真。
教程分为 9 个章节，每个章节都是可独立运行的代码段。

环境准备：
    conda env create -f environment.yml
    conda activate opensim

运行本教程：
    python tutorial_opensim_python.py

================================================================================
"""

import opensim as osim
import numpy as np
import math
import os

# 设置为 False 可跳过可视化（无 GUI 环境下使用）
USE_VISUALIZER = os.getenv('OPENSIM_USE_VISUALIZER', '0') != '0'


# ============================================================================
# 第 1 章：核心概念
# ============================================================================
#
# OpenSim 的核心对象层次结构：
#
#   Model（模型）
#   ├── Body（刚体）         —— 有质量、惯性的物理实体
#   ├── Joint（关节）        —— 连接刚体，定义自由度
#   ├── Force（力）          —— 包括肌肉、弹簧、执行器等
#   ├── Controller（控制器） —— 驱动执行器的控制信号
#   ├── Constraint（约束）   —— 运动学约束
#   ├── Marker（标记点）     —— 用于运动捕捉对齐
#   └── Component（组件）    —— Reporter 等通用组件
#
# 仿真流程：
#   1. 构建/加载 Model
#   2. initSystem() → 获得 State
#   3. 设置初始条件
#   4. Manager.integrate() → 前向仿真
#   5. 分析结果
#


# ============================================================================
# 第 2 章：从零构建一个手臂模型
# ============================================================================

def chapter2_build_arm_model():
    """构建一个包含上臂、前臂、肘关节和二头肌的简单手臂模型。"""
    print("\n" + "=" * 60)
    print("第 2 章：从零构建手臂模型")
    print("=" * 60)

    # --- 2.1 创建空模型 ---
    arm = osim.Model()
    arm.setName("simple_arm")
    # 可视化器（可选，需要 GUI 环境）
    if USE_VISUALIZER:
        arm.setUseVisualizer(True)

    # --- 2.2 创建刚体 ---
    # Body(名称, 质量kg, 质心位置, 惯性)
    humerus = osim.Body("humerus", 1.0, osim.Vec3(0), osim.Inertia(0))
    radius = osim.Body("radius", 1.0, osim.Vec3(0), osim.Inertia(0))

    # --- 2.3 创建关节 ---
    # PinJoint: 单自由度旋转关节
    # 参数：名称, 父体, 父体偏移位置, 父体偏移旋转, 子体, 子体偏移位置, 子体偏移旋转
    shoulder = osim.PinJoint(
        "shoulder",
        arm.getGround(),        # 父体 = 地面
        osim.Vec3(0, 0, 0),     # 关节在父体上的位置
        osim.Vec3(0, 0, 0),     # 关节在父体上的旋转
        humerus,                # 子体
        osim.Vec3(0, 1, 0),     # 关节在子体上的位置（上臂长 1m）
        osim.Vec3(0, 0, 0),
    )

    elbow = osim.PinJoint(
        "elbow",
        humerus,                # 父体 = 上臂
        osim.Vec3(0, 0, 0),
        osim.Vec3(0, 0, 0),
        radius,                 # 子体 = 前臂
        osim.Vec3(0, 1, 0),     # 前臂长 1m
        osim.Vec3(0, 0, 0),
    )

    # --- 2.4 创建肌肉 ---
    # Millard2012EquilibriumMuscle: 常用的肌肉模型
    # 参数：名称, 最大等长力(N), 最优纤维长度(m), 肌腱松弛长度(m), 羽状角(rad)
    biceps = osim.Millard2012EquilibriumMuscle(
        "biceps", 200.0, 0.6, 0.55, 0.0
    )
    # 添加肌肉路径点：起点在上臂，止点在前臂
    biceps.addNewPathPoint("origin", humerus, osim.Vec3(0, 0.8, 0))
    biceps.addNewPathPoint("insertion", radius, osim.Vec3(0, 0.7, 0))

    # --- 2.5 创建控制器 ---
    # PrescribedController: 按预设函数控制肌肉激活
    brain = osim.PrescribedController()
    brain.addActuator(biceps)
    # StepFunction(起始时间, 结束时间, 起始值, 结束值)
    # 在 0.5~3.0 秒之间，激活从 0.3 升到 1.0
    brain.prescribeControlForActuator(
        "biceps", osim.StepFunction(0.5, 3.0, 0.3, 1.0)
    )

    # --- 2.6 组装模型 ---
    arm.addBody(humerus)
    arm.addBody(radius)
    arm.addJoint(shoulder)
    arm.addJoint(elbow)
    arm.addForce(biceps)
    arm.addController(brain)

    # --- 2.7 添加显示几何体（可选，用于可视化） ---
    body_geom = osim.Ellipsoid(0.1, 0.5, 0.1)
    body_geom.setColor(osim.Gray)

    humerus_center = osim.PhysicalOffsetFrame()
    humerus_center.setName("humerusCenter")
    humerus_center.setParentFrame(humerus)
    humerus_center.setOffsetTransform(osim.Transform(osim.Vec3(0, 0.5, 0)))
    humerus.addComponent(humerus_center)
    humerus_center.attachGeometry(body_geom.clone())

    radius_center = osim.PhysicalOffsetFrame()
    radius_center.setName("radiusCenter")
    radius_center.setParentFrame(radius)
    radius_center.setOffsetTransform(osim.Transform(osim.Vec3(0, 0.5, 0)))
    radius.addComponent(radius_center)
    radius_center.attachGeometry(body_geom.clone())

    # --- 2.8 添加控制台报告器 ---
    reporter = osim.ConsoleReporter()
    reporter.set_report_time_interval(1.0)
    reporter.addToReport(biceps.getOutput("fiber_force"))
    reporter.addToReport(elbow.getCoordinate().getOutput("value"), "elbow_angle")
    arm.addComponent(reporter)

    # --- 2.9 初始化并配置初始状态 ---
    state = arm.initSystem()
    shoulder.getCoordinate().setLocked(state, True)        # 锁定肩关节
    elbow.getCoordinate().setValue(state, 0.5 * math.pi)   # 肘关节初始角度 90°
    arm.equilibrateMuscles(state)                          # 平衡肌肉状态

    # --- 2.10 运行仿真 ---
    manager = osim.Manager(arm)
    state.setTime(0)
    manager.initialize(state)
    state = manager.integrate(10.0)  # 仿真 10 秒

    # --- 2.11 保存运动数据 ---
    states_table = manager.getStatesTable()
    sto_adapter = osim.STOFileAdapter()
    sto_adapter.write(states_table, "output_SimpleArm_states.sto")
    print("运动数据已保存到 output_SimpleArm_states.sto")

    # --- 2.12 保存模型文件 ---
    arm.printToXML("output_SimpleArm.osim")
    print("模型已保存到 output_SimpleArm.osim")

    return arm


# ============================================================================
# 第 3 章：加载和检查现有模型
# ============================================================================

def chapter3_load_and_inspect():
    """加载 .osim 模型文件，提取模型信息。"""
    print("\n" + "=" * 60)
    print("第 3 章：加载和检查现有模型")
    print("=" * 60)

    # 使用项目中自带的步态模型
    model_path = os.path.join("Models", "Gait10dof18musc", "gait10dof18musc.osim")

    if not os.path.exists(model_path):
        print(f"模型文件不存在: {model_path}，跳过本章。")
        return

    model = osim.Model(model_path)
    state = model.initSystem()

    # --- 3.1 基本信息 ---
    print(f"模型名称: {model.getName()}")
    print(f"重力: {model.get_gravity()}")

    # --- 3.2 遍历刚体 ---
    print(f"\n刚体数量: {model.getBodySet().getSize()}")
    for i in range(model.getBodySet().getSize()):
        body = model.getBodySet().get(i)
        print(f"  [{i}] {body.getName()}, 质量: {body.getMass():.3f} kg")

    # --- 3.3 遍历关节和坐标（自由度） ---
    print(f"\n关节数量: {model.getJointSet().getSize()}")
    coord_set = model.getCoordinateSet()
    print(f"自由度数量: {coord_set.getSize()}")
    for i in range(coord_set.getSize()):
        coord = coord_set.get(i)
        print(f"  [{i}] {coord.getName()}, "
              f"范围: [{math.degrees(coord.getRangeMin()):.1f}°, "
              f"{math.degrees(coord.getRangeMax()):.1f}°]")

    # --- 3.4 遍历肌肉 ---
    muscles = model.getMuscles()
    print(f"\n肌肉数量: {muscles.getSize()}")
    for i in range(muscles.getSize()):
        muscle = muscles.get(i)
        print(f"  [{i}] {muscle.getName()}, "
              f"最大力: {muscle.getMaxIsometricForce():.1f} N, "
              f"最优纤维长: {muscle.getOptimalFiberLength():.4f} m")


# ============================================================================
# 第 4 章：NumPy 与 OpenSim 数据互转
# ============================================================================

def chapter4_numpy_conversions():
    """NumPy 数组与 OpenSim 向量/矩阵之间的转换。"""
    print("\n" + "=" * 60)
    print("第 4 章：NumPy 数据互转")
    print("=" * 60)

    # --- 4.1 Vec3 互转 ---
    np_vec = np.array([1.0, 2.0, 3.0])
    osim_vec3 = osim.Vec3.createFromMat(np_vec)
    back_to_np = osim_vec3.to_numpy()
    print(f"Vec3: {np_vec} → OpenSim → {back_to_np}")

    # --- 4.2 Vector 互转 ---
    np_arr = np.array([5, 3, 6, 2, 9], dtype=float)
    osim_vector = osim.Vector.createFromMat(np_arr)
    back_to_np = osim_vector.to_numpy()
    print(f"Vector: {np_arr} → OpenSim → {back_to_np}")

    # --- 4.3 RowVector 互转 ---
    osim_row = osim.RowVector.createFromMat(np_arr)
    print(f"RowVector: {osim_row.to_numpy()}")

    # --- 4.4 Matrix 互转 ---
    np_mat = np.array([[5.0, 3.0], [3.0, 6.0], [8.0, 1.0]])
    osim_mat = osim.Matrix.createFromMat(np_mat)
    back_to_np = osim_mat.to_numpy()
    print(f"Matrix:\n{np_mat}\n→ OpenSim → \n{back_to_np}")

    # --- 4.5 从 TimeSeriesTable 提取列到 NumPy ---
    # 创建一个简单的 TimeSeriesTable
    table = osim.TimeSeriesTable()
    table.setColumnLabels(["col_a", "col_b"])
    row = osim.RowVector.createFromMat(np.array([1.0, 2.0]))
    table.appendRow(0.0, row)
    table.appendRow(0.1, osim.RowVector.createFromMat(np.array([3.0, 4.0])))
    table.appendRow(0.2, osim.RowVector.createFromMat(np.array([5.0, 6.0])))

    # 提取单列为 NumPy 数组
    col_a = table.getDependentColumn("col_a").to_numpy()
    times = np.array(table.getIndependentColumn())
    print(f"\n时间: {times}")
    print(f"col_a: {col_a}")


# ============================================================================
# 第 5 章：数据文件读写（STO/MOT/CSV）
# ============================================================================

def chapter5_data_io():
    """读写 OpenSim 的数据文件格式。"""
    print("\n" + "=" * 60)
    print("第 5 章：数据文件读写")
    print("=" * 60)

    # --- 5.1 创建并写入 STO 文件 ---
    table = osim.TimeSeriesTable()
    table.setColumnLabels(["angle", "velocity"])
    for i in range(50):
        t = i * 0.02
        row = osim.RowVector.createFromMat(
            np.array([math.sin(t * 2 * math.pi), math.cos(t * 2 * math.pi)])
        )
        table.appendRow(t, row)

    osim.STOFileAdapter.write(table, "output_example_data.sto")
    print("已写入 output_example_data.sto")

    # --- 5.2 读取 STO 文件 ---
    loaded = osim.TimeSeriesTable("output_example_data.sto")
    print(f"读取到 {loaded.getNumRows()} 行, {loaded.getNumColumns()} 列")
    print(f"列名: {loaded.getColumnLabels()}")
    print(f"时间范围: {loaded.getIndependentColumn()[0]:.3f} ~ "
          f"{loaded.getIndependentColumn()[-1]:.3f} s")

    # 提取数据到 NumPy
    angle = loaded.getDependentColumn("angle").to_numpy()
    velocity = loaded.getDependentColumn("velocity").to_numpy()
    times = np.array(loaded.getIndependentColumn())
    print(f"angle 前5个值: {angle[:5]}")

    # --- 5.3 读取 Vec3 类型的 STO ---
    # Vec3 STO 文件常见于标记点、力等三维数据
    # 使用 TimeSeriesTableVec3 读取
    # table_vec3 = osim.TimeSeriesTableVec3("markers.sto")

    # --- 5.4 写入 Vec3 类型的 STO ---
    table_vec3 = osim.TimeSeriesTableVec3()
    table_vec3.setColumnLabels(["marker_1", "marker_2"])
    for i in range(10):
        t = i * 0.1
        row = osim.RowVectorVec3([
            osim.Vec3(t, 0, 0),
            osim.Vec3(0, t, 0),
        ])
        table_vec3.appendRow(t, row)

    osim.STOFileAdapterVec3.write(table_vec3, "output_markers.sto")
    print("已写入 output_markers.sto (Vec3 格式)")


# ============================================================================
# 第 6 章：仿真后分析（Post-hoc Analysis）
# ============================================================================

def chapter6_posthoc_analysis():
    """用 StatesTrajectory 进行仿真后分析，计算关节反力。"""
    print("\n" + "=" * 60)
    print("第 6 章：仿真后分析")
    print("=" * 60)

    # --- 6.1 创建单摆模型 ---
    model = osim.Model()
    model.setName("pendulum")

    body = osim.Body("body", 1.0, osim.Vec3(0), osim.Inertia(1))
    model.addComponent(body)

    joint = osim.PinJoint(
        "joint",
        model.getGround(), osim.Vec3(0), osim.Vec3(0),
        body, osim.Vec3(0, 1.0, 0), osim.Vec3(0),
    )
    joint.updCoordinate().setName("q")
    model.addComponent(joint)

    # 使用 StatesTrajectoryReporter 收集仿真过程中的状态
    reporter = osim.StatesTrajectoryReporter()
    reporter.setName("reporter")
    reporter.set_report_time_interval(0.05)
    model.addComponent(reporter)

    # --- 6.2 运行仿真 ---
    state = model.initSystem()
    model.setStateVariableValue(state, "joint/q/value", 0.25 * math.pi)

    manager = osim.Manager(model, state)
    manager.integrate(1.0)

    # --- 6.3 分析关节反力 ---
    states_traj = reporter.getStates()
    print(f"收集到 {states_traj.getSize()} 个状态帧\n")

    print(f"{'时间(s)':>8}  {'角度(rad)':>10}  {'反力(N)':>10}")
    print("-" * 32)
    for i in range(states_traj.getSize()):
        s = states_traj[i]
        t = s.getTime()
        q = model.getStateVariableValue(s, "joint/q/value")

        # 计算关节反力需要 realize 到 Acceleration 级别
        model.realizeAcceleration(s)
        reaction = joint.calcReactionOnParentExpressedInGround(s)
        force = reaction.get(1)  # SpatialVec: [0]=力矩, [1]=力
        force_mag = math.sqrt(force[0] ** 2 + force[1] ** 2 + force[2] ** 2)

        if i % 4 == 0:  # 每隔几帧打印一次
            print(f"{t:8.3f}  {q:10.4f}  {force_mag:10.4f}")

    # --- 6.4 也可以从 StatesTable 创建 StatesTrajectory ---
    states_table = manager.getStatesTable()
    traj2 = osim.StatesTrajectory.createFromStatesTable(model, states_table)
    print(f"\n从 StatesTable 创建: {traj2.getSize()} 帧")


# ============================================================================
# 第 7 章：输入输出连接与 TableReporter
# ============================================================================

def chapter7_wiring_and_reporter():
    """展示 OpenSim 的 Input/Output 连接系统和 TableReporter。"""
    print("\n" + "=" * 60)
    print("第 7 章：Input/Output 连接与 TableReporter")
    print("=" * 60)

    model = osim.Model()
    model.setName("reporter_demo")

    # 创建一个自由体
    body = osim.Body("body", 1.0, osim.Vec3(0), osim.Inertia(1))
    joint = osim.FreeJoint("joint", model.getGround(), body)
    model.addComponent(body)
    model.addComponent(joint)

    # --- 7.1 创建 TableReporter 并连接输出 ---
    # TableReporterVec3 用于记录三维向量输出
    reporter = osim.TableReporterVec3()
    reporter.setName("reporter")
    reporter.set_report_time_interval(0.1)

    # 连接 body 的位置输出
    reporter.addToReport(body.getOutput("position"), "body_pos")
    # 连接模型的质心位置输出
    reporter.addToReport(model.getOutput("com_position"), "com_pos")
    model.addComponent(reporter)

    # 必须调用 finalizeConnections() 来完成输入输出连接
    model.finalizeConnections()

    # --- 7.2 仿真 ---
    state = model.initSystem()
    manager = osim.Manager(model)
    state.setTime(0)
    manager.initialize(state)
    state = manager.integrate(1.0)

    # --- 7.3 获取并保存结果 ---
    table = reporter.getTable()
    print(f"Reporter 记录了 {table.getNumRows()} 行数据")
    print(f"列名: {table.getColumnLabels()}")

    osim.STOFileAdapterVec3.write(table, "output_reporter_demo.sto")
    print("结果已保存到 output_reporter_demo.sto")

    # --- 7.4 保存模型（连接信息会序列化到 XML 中） ---
    model.printToXML("output_reporter_demo.osim")
    print("模型已保存到 output_reporter_demo.osim")


# ============================================================================
# 第 8 章：Moco 轨迹优化入门
# ============================================================================

def chapter8_moco_basics():
    """
    用 Moco 求解最优控制问题：让一个滑块在最短时间内从位置 0 移动到位置 1。

    Moco 工作流：
        1. 创建 MocoStudy
        2. 定义 MocoProblem（模型 + 约束 + 目标）
        3. 配置 Solver（CasADi 或 Tropter）
        4. solve() → MocoTrajectory
    """
    print("\n" + "=" * 60)
    print("第 8 章：Moco 轨迹优化入门")
    print("=" * 60)

    # --- 8.1 创建滑块模型 ---
    model = osim.Model()
    model.setName("sliding_mass")
    model.set_gravity(osim.Vec3(0, 0, 0))  # 无重力

    body = osim.Body("body", 2.0, osim.Vec3(0), osim.Inertia(0))
    model.addComponent(body)

    joint = osim.SliderJoint("slider", model.getGround(), body)
    coord = joint.updCoordinate()
    coord.setName("position")
    model.addComponent(joint)

    # CoordinateActuator: 直接对某个自由度施力
    actu = osim.CoordinateActuator()
    actu.setCoordinate(coord)
    actu.setName("actuator")
    actu.setOptimalForce(1)
    model.addComponent(actu)

    body.attachGeometry(osim.Sphere(0.05))
    model.finalizeConnections()

    # --- 8.2 创建 MocoStudy ---
    study = osim.MocoStudy()
    study.setName("sliding_mass")

    # --- 8.3 定义优化问题 ---
    problem = study.updProblem()
    problem.setModel(model)

    # 时间边界：初始时间 = 0，终止时间在 [0, 5] 之间
    problem.setTimeBounds(
        osim.MocoInitialBounds(0.0),
        osim.MocoFinalBounds(0.0, 5.0),
    )

    # 状态边界
    # 位置：全程 [-5, 5]，初始 = 0，终止 = 1
    problem.setStateInfo(
        "/slider/position/value",
        osim.MocoBounds(-5, 5),
        osim.MocoInitialBounds(0),
        osim.MocoFinalBounds(1),
    )
    # 速度：全程 [-50, 50]，初始 = 0，终止 = 0
    problem.setStateInfo("/slider/position/speed", [-50, 50], [0], [0])

    # 控制边界
    problem.setControlInfo("/actuator", osim.MocoBounds(-50, 50))

    # 目标函数：最小化终止时间
    problem.addGoal(osim.MocoFinalTimeGoal())

    # --- 8.4 配置求解器 ---
    solver = study.initCasADiSolver()
    solver.set_num_mesh_intervals(50)

    # --- 8.5 求解 ---
    print("正在求解最优控制问题...")
    solution = study.solve()

    # --- 8.6 查看结果 ---
    print(f"求解状态: {solution.getStatus()}")
    solution.write("output_sliding_mass_solution.sto")
    print("解已保存到 output_sliding_mass_solution.sto")

    # 提取状态数据
    states_table = solution.exportToStatesTable()
    times = np.array(states_table.getIndependentColumn())
    position = states_table.getDependentColumn("/slider/position/value").to_numpy()
    speed = states_table.getDependentColumn("/slider/position/speed").to_numpy()
    print(f"终止时间: {times[-1]:.4f} s")
    print(f"终止位置: {position[-1]:.4f}")
    print(f"终止速度: {speed[-1]:.6f}")

    # 保存问题配置文件（可复现）
    study.printToXML("output_sliding_mass.omoco")

    return solution


# ============================================================================
# 第 9 章：带肌肉的 Moco 优化
# ============================================================================

def chapter9_moco_with_muscle():
    """
    用 Moco 优化肌肉控制：一个质量挂在肌肉上，用最短时间抬起一小段距离。
    这个例子展示了 DeGrooteFregly2016Muscle —— Moco 专用的肌肉模型。
    """
    print("\n" + "=" * 60)
    print("第 9 章：带肌肉的 Moco 优化")
    print("=" * 60)

    # --- 9.1 创建肌肉悬挂模型 ---
    model = osim.Model()
    model.setName("hanging_muscle")
    model.set_gravity(osim.Vec3(9.81, 0, 0))  # +x 方向为重力方向

    body = osim.Body("body", 0.5, osim.Vec3(0), osim.Inertia(1))
    model.addComponent(body)

    joint = osim.SliderJoint("joint", model.getGround(), body)
    coord = joint.updCoordinate()
    coord.setName("height")
    model.addComponent(joint)

    # DeGrooteFregly2016Muscle: 唯一与 Moco 充分测试过的肌肉模型
    muscle = osim.DeGrooteFregly2016Muscle()
    muscle.setName("muscle")
    muscle.set_max_isometric_force(30.0)
    muscle.set_optimal_fiber_length(0.10)
    muscle.set_tendon_slack_length(0.05)
    muscle.set_tendon_strain_at_one_norm_force(0.10)
    muscle.set_ignore_activation_dynamics(False)
    muscle.set_ignore_tendon_compliance(False)
    muscle.set_fiber_damping(0.01)
    muscle.set_tendon_compliance_dynamics_mode("implicit")
    muscle.set_max_contraction_velocity(10)
    muscle.set_pennation_angle_at_optimal(0.10)
    muscle.addNewPathPoint("origin", model.updGround(), osim.Vec3(0))
    muscle.addNewPathPoint("insertion", body, osim.Vec3(0))
    model.addForce(muscle)

    body.attachGeometry(osim.Sphere(0.05))
    model.finalizeConnections()
    model.printToXML("output_hanging_muscle.osim")

    # --- 9.2 设置 Moco 问题 ---
    study = osim.MocoStudy()
    problem = study.updProblem()
    problem.setModelAsCopy(model)

    # 时间边界
    problem.setTimeBounds(0, [0.05, 1.0])

    # 位置：从 0.15m 抬到 0.14m（下降 1cm 即抬起负重）
    problem.setStateInfo("/joint/height/value", [0.14, 0.16], 0.15, 0.14)
    problem.setStateInfo("/joint/height/speed", [-1, 1], 0, 0)
    problem.setControlInfo("/forceset/muscle", [0.01, 1])

    # 初始激活约束
    initial_activation = osim.MocoInitialActivationGoal()
    initial_activation.setName("initial_activation")
    problem.addGoal(initial_activation)

    # 初始肌腱平衡约束
    initial_eq = osim.MocoInitialVelocityEquilibriumDGFGoal()
    initial_eq.setName("initial_velocity_equilibrium")
    initial_eq.setMode("cost")
    initial_eq.setWeight(0.001)
    problem.addGoal(initial_eq)

    # 目标：最小化时间
    problem.addGoal(osim.MocoFinalTimeGoal())

    # --- 9.3 配置求解器 ---
    solver = study.initCasADiSolver()
    solver.set_num_mesh_intervals(25)
    solver.set_multibody_dynamics_mode("implicit")
    solver.set_optim_convergence_tolerance(1e-4)
    solver.set_optim_constraint_tolerance(1e-4)

    # --- 9.4 求解 ---
    print("正在求解肌肉优化问题（可能需要几十秒）...")
    solution = study.solve()
    print(f"求解状态: {solution.getStatus()}")

    # --- 9.5 导出结果 ---
    osim.STOFileAdapter.write(
        solution.exportToStatesTable(), "output_hanging_muscle_states.sto"
    )
    osim.STOFileAdapter.write(
        solution.exportToControlsTable(), "output_hanging_muscle_controls.sto"
    )
    print("状态和控制信号已保存。")

    return solution


# ============================================================================
# 第 10 章：常用关节类型速查
# ============================================================================

def chapter10_joint_types():
    """展示 OpenSim 中常用的关节类型。"""
    print("\n" + "=" * 60)
    print("第 10 章：关节类型速查")
    print("=" * 60)

    joint_info = """
    ┌─────────────────────┬──────┬───────────────────────────────────┐
    │ 关节类型             │ DOF  │ 说明                              │
    ├─────────────────────┼──────┼───────────────────────────────────┤
    │ PinJoint            │  1   │ 单轴旋转（铰链），如肘关节          │
    │ SliderJoint         │  1   │ 单轴平移，如滑轨                   │
    │ PlanarJoint         │  3   │ 平面运动（2 平移 + 1 旋转）        │
    │ BallJoint           │  3   │ 球窝关节（3 旋转），如肩关节         │
    │ UniversalJoint      │  2   │ 万向节（2 旋转）                   │
    │ FreeJoint           │  6   │ 完全自由（3 旋转 + 3 平移）        │
    │ WeldJoint           │  0   │ 刚性焊接，无自由度                 │
    │ CustomJoint         │ 自定 │ 自定义坐标变换                     │
    │ GimbalJoint         │  3   │ 万向架（3 旋转，不同于 Ball）      │
    │ EllipsoidJoint      │  3   │ 椭球面约束的 3 旋转                │
    └─────────────────────┴──────┴───────────────────────────────────┘
    """
    print(joint_info)

    # 示例：用 CustomJoint 定义耦合坐标
    print("提示：CustomJoint 可定义坐标之间的耦合关系，")
    print("例如膝关节屈曲时的胫骨前移（tibial translation）。")


# ============================================================================
# 主程序：选择要运行的章节
# ============================================================================

if __name__ == "__main__":
    osim.ModelVisualizer.addDirToGeometrySearchPaths(
        os.path.join(os.getcwd(), "Geometry")
    )

    print("=" * 60)
    print("  OpenSim Python API 教程")
    print("  OpenSim 版本:", osim.GetVersion())
    print("=" * 60)

    # 可以注释掉不需要的章节
    # 第 2 章需要可视化器，在无 GUI 环境下设 USE_VISUALIZER = False
    # 第 8、9 章需要 CasADi 求解器，首次运行可能较慢
    chapter2_build_arm_model()     # 构建手臂模型
    chapter3_load_and_inspect()      # 加载和检查模型

    chapter4_numpy_conversions()     # NumPy 互转（最快，无需仿真）
    chapter5_data_io()               # 数据文件读写
    chapter10_joint_types()          # 关节类型速查
    chapter3_load_and_inspect()      # 加载和检查模型
    chapter6_posthoc_analysis()      # 仿真后分析
    chapter7_wiring_and_reporter()   # Input/Output 和 Reporter

    # 以下章节涉及仿真和优化，运行时间较长：
    # chapter2_build_arm_model()     # 构建手臂并仿真 10 秒（需要可视化器或设 USE_VISUALIZER=False）
    # chapter8_moco_basics()         # Moco 滑块优化
    # chapter9_moco_with_muscle()    # Moco 肌肉优化

    print("\n" + "=" * 60)
    print("教程运行完毕！")
    print("=" * 60)
    print("""
后续学习建议：
  1. 取消注释 chapter2 运行手臂仿真
  2. 取消注释 chapter8/9 体验 Moco 轨迹优化
  3. 阅读 Code/Python/Moco/ 下的进阶示例：
     - examplePredictAndTrack.py    双摆预测与追踪
     - example3DWalking/            80 块肌肉的 3D 步行
     - exampleSquatToStand/         深蹲起立全流程
  4. 阅读 Code/Python/OpenSenseExample/ 学习 IMU 追踪
  5. 用 Models/ 中的预建模型进行实验
""")
