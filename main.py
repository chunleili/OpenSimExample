"""VBD 体积肌肉仿真 - 主入口

用法:
    uv run python main.py                  # 运行全部（跳过慢速VBD）
    uv run python main.py --all            # 运行全部（含VBD准静态）
    uv run python main.py test             # 仅运行Phase 1验证测试
    uv run python main.py curves           # 仅绘制DGF解析曲线
    uv run python main.py activation       # 仅运行激活动力学
    uv run python main.py vbd              # 仅运行VBD准静态F-L
    uv run python main.py demo             # 运行VBD动态演示（重力下落）
"""

import sys
import numpy as np



def run_tests():
    """Phase 1 验证测试"""
    from tests.test_phase1 import main as test_main
    return test_main()


def run_curves():
    """绘制DGF解析曲线"""
    from run_level0_fem import plot_analytical_curves
    plot_analytical_curves()
    import matplotlib.pyplot as plt
    plt.show()


def run_activation():
    """激活动力学验证"""
    from run_level0_fem import run_activation_dynamics_validation
    run_activation_dynamics_validation()
    import matplotlib.pyplot as plt
    plt.show()


def run_vbd():
    """VBD准静态F-L验证"""
    from run_level0_fem import run_vbd_fl_validation
    run_vbd_fl_validation()
    import matplotlib.pyplot as plt
    plt.show()


def run_demo():
    """VBD动态演示：肌肉块在重力下自由下落并激活收缩"""
    import matplotlib.pyplot as plt
    from vbd_muscle.mesh import generate_box_mesh, assign_fiber_directions
    from vbd_muscle.solver import VBDSolver
    from vbd_muscle.activation import activation_dynamics

    print("=== VBD 动态演示：重力 + 激活收缩 ===")

    # 创建小肌肉块
    l_opt = 0.10
    side = 0.02
    nodes, tets = generate_box_mesh(side, side, l_opt, 2, 2, 5)
    fiber_dirs = assign_fiber_directions(nodes, tets)

    solver = VBDSolver(
        nodes, tets, fiber_dirs,
        mu=5000.0, kappa=500000.0, sigma0=300000.0,
        density=1060.0, damping=0.05, dt=0.002,
        n_iterations=10,
        gravity=np.array([0.0, 0.0, -9.81]),
    )

    # 固定z=0面
    z_min = np.where(np.abs(nodes[:, 2]) < 1e-10)[0]
    solver.set_fixed_vertices(z_min)
    solver.mesh_info()

    import os
    from vbd_muscle.mesh import extract_surface_triangles, save_ply

    output_dir = "output/ply"
    os.makedirs(output_dir, exist_ok=True)
    surface_indices = extract_surface_triangles(tets)
    save_ply(f'{output_dir}/0.ply', nodes, surface_indices)


    def activation_ramp(t: float) -> float:
        """Piecewise activation over normalized time [0,1]: 0→0.5→1.0→0.7→0.3→0."""
        if t < 0.0:
            return 0.0
        elif t <= 0.3:
            return 0.5
        elif t <= 0.5:
            return 1.0
        elif t <= 0.7:
            return 0.7
        elif t <= 0.8:
            return 0.3
        else:
            return 0.0
    

    # 模拟
    n_steps = 200
    dt = 0.002
    excitation = 0.0
    activation = 0.01
    times, lengths, activations = [], [], []

    print(f"\n模拟 {n_steps} 步 (dt={dt*1000:.0f}ms, 总时长 {n_steps*dt*1000:.0f}ms)")
    print("  t=0-100ms: 无激活（自由悬挂）")
    print("  t=100-400ms: 激活=1.0（收缩）\n")

    for step in range(n_steps):
        t = step * dt

        # TODO: for now, turn off the activation dynamics and directly use the ramped activation.
        activation = activation_ramp(t / (n_steps * dt))
        # excitation = activation_ramp(t / (n_steps * dt))
        # activation = float(activation_dynamics(excitation, activation, dt))

        solver.step(activation=activation)

        # 测量自由端z坐标（平均）
        z_max_verts = np.where(nodes[:, 2] > l_opt - 1e-10)[0]
        avg_z = solver.x[z_max_verts, 2].mean()

        times.append(t * 1000)
        lengths.append(avg_z / l_opt)
        activations.append(activation)

        save_ply(f'{output_dir}/{step + 1}.ply', solver.x, surface_indices)

        if step % 50 == 0:
            print(f"  t={t*1000:6.0f}ms  a={activation:.3f}  "
                  f"free_end_z={avg_z*100:.2f}cm  stretch={avg_z/l_opt:.3f}")

    # 绘图
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    ax1.plot(times, activations, 'r-', linewidth=2)
    ax1.set_ylabel('Activation')
    ax1.set_title('VBD Dynamic Demo: Gravity + Activation')
    ax1.grid(True, alpha=0.3)

    ax2.plot(times, lengths, 'b-', linewidth=2)
    ax2.axhline(1.0, color='gray', linestyle='--', alpha=0.5, label='rest length')
    ax2.set_xlabel('Time (ms)')
    ax2.set_ylabel('Normalized length')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('output_demo.png', dpi=150)
    print(f"\nSaved output_demo.png")
    plt.show()


def main():
    commands = {
        'test': ('Phase 1 验证测试', run_tests),
        'curves': ('DGF 解析曲线', run_curves),
        'activation': ('激活动力学', run_activation),
        'vbd': ('VBD 准静态 F-L', run_vbd),
        'demo': ('VBD 动态演示', run_demo),
    }
    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        if cmd == '--all':
            run_tests()
            from run_level0_fem import main as level0_main
            level0_main()
        elif cmd in commands:
            name, func = commands[cmd]
            print(f"--- {name} ---")
            func()
        else:
            print(f"未知命令: {cmd}")
            print(__doc__)
            sys.exit(1)
    else:
        # 默认：运行 demo
        print("=" * 50)
        print("VBD 体积肌肉仿真")
        print("=" * 50)
        print()
        print("可用命令:")
        for cmd, (name, _) in commands.items():
            print(f"  uv run python main.py {cmd:12s} # {name}")
        print(f"  uv run python main.py {'--all':12s} # 全部运行")
        print()

        # 运行 demo
        print("--- 运行动态演示 ---\n")
        run_demo()


if __name__ == "__main__":
    main()
