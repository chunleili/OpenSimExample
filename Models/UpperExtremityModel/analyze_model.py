"""
Stanford VA Upper Extremity Model 解析与可视化工具
解析 SIMM 格式的 .jnt（关节）和 .msl（肌肉）文件

依赖: pip install numpy matplotlib pandas
可选: pip install pyvista  (用于三维骨骼可视化)
"""

import re
import struct
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).parent
JNT_FILE = BASE_DIR / "Stanford VA upper limb model.jnt"
MSL_FILE = BASE_DIR / "Stanford VA upper limb model.msl"
BONES_DIR = BASE_DIR / "bones"


# ============================================================
# 1. 解析 .msl 文件 — 提取所有肌肉参数
# ============================================================
def parse_muscles(filepath: Path) -> list[dict]:
    """解析 .msl 文件，提取每条肌肉的力学参数和附着点。"""
    text = filepath.read_text(encoding="utf-8", errors="ignore")
    muscles = []

    # 匹配每个 beginmuscle ... endmuscle 块
    pattern = re.compile(
        r"beginmuscle\s+(\S+)\s*\n(.*?)endmuscle", re.DOTALL
    )
    for match in pattern.finditer(text):
        name = match.group(1)
        if name == "defaultmuscle":
            continue
        body = match.group(2)

        muscle = {"name": name}

        # 提取数值参数
        for param in [
            "max_force",
            "optimal_fiber_length",
            "tendon_slack_length",
            "pennation_angle",
        ]:
            m = re.search(rf"{param}\s+([\d.]+)", body)
            if m:
                muscle[param] = float(m.group(1))

        # 提取肌肉分组
        gm = re.search(r"begingroups\s*\n(.*?)endgroups", body, re.DOTALL)
        if gm:
            muscle["group"] = gm.group(1).strip()

        # 提取附着点
        pm = re.search(r"beginpoints\s*\n(.*?)endpoints", body, re.DOTALL)
        if pm:
            points = []
            for line in pm.group(1).strip().split("\n"):
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) >= 5:
                    x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                    segment = parts[4]
                    points.append({"x": x, "y": y, "z": z, "segment": segment})
            muscle["points"] = points

        muscles.append(muscle)

    return muscles


# ============================================================
# 2. 解析 .jnt 文件 — 提取骨骼段和关节信息
# ============================================================
def parse_segments(filepath: Path) -> list[dict]:
    """解析 .jnt 文件中的 segment 定义。"""
    text = filepath.read_text(encoding="utf-8", errors="ignore")
    segments = []
    pattern = re.compile(
        r"beginsegment\s+(\S+)\s*\n(.*?)endsegment", re.DOTALL
    )
    for match in pattern.finditer(text):
        name = match.group(1)
        body = match.group(2)
        bone_match = re.search(r"bone\s+(\S+)", body)
        bone_file = bone_match.group(1) if bone_match else None
        segments.append({"name": name, "bone_file": bone_file})
    return segments


def parse_joints(filepath: Path) -> list[dict]:
    """解析 .jnt 文件中的关节定义。"""
    text = filepath.read_text(encoding="utf-8", errors="ignore")
    joints = []
    pattern = re.compile(
        r"beginjoint\s+(\S+)\s*\n(.*?)endjoint", re.DOTALL
    )
    for match in pattern.finditer(text):
        name = match.group(1)
        body = match.group(2)

        joint = {"name": name}

        sm = re.search(r"segments\s+(\S+)\s+(\S+)", body)
        if sm:
            joint["parent"] = sm.group(1)
            joint["child"] = sm.group(2)

        om = re.search(r"order\s+(.+)", body)
        if om:
            joint["order"] = om.group(1).strip()

        joints.append(joint)
    return joints


# ============================================================
# 3. 读取 .asc 骨骼几何文件
# ============================================================
def read_bone_asc(filepath: Path) -> tuple[np.ndarray, np.ndarray] | None:
    """
    读取 SIMM .asc 骨骼文件（二进制格式）。
    返回 (vertices, faces) 或 None。
    """
    try:
        data = filepath.read_bytes()
    except FileNotFoundError:
        return None

    # .asc 文件头部包含顶点数和面数（小端 int32）
    # 格式: num_vertices, num_faces, 然后是顶点和面数据
    if len(data) < 8:
        return None

    offset = 0
    num_vertices = struct.unpack_from("<i", data, offset)[0]
    offset += 4
    num_faces = struct.unpack_from("<i", data, offset)[0]
    offset += 4

    if num_vertices <= 0 or num_vertices > 100000:
        return None
    if num_faces <= 0 or num_faces > 200000:
        return None

    # 每个顶点: x, y, z (float64) + 法线 nx, ny, nz (float64) = 48 bytes
    vertex_size = 6 * 8  # 6 doubles
    expected = offset + num_vertices * vertex_size + num_faces * 3 * 4
    if len(data) < offset + num_vertices * vertex_size:
        return None

    vertices = np.zeros((num_vertices, 3))
    for i in range(num_vertices):
        vx, vy, vz = struct.unpack_from("<3d", data, offset)
        vertices[i] = [vx, vy, vz]
        offset += vertex_size

    faces = np.zeros((num_faces, 3), dtype=int)
    for i in range(num_faces):
        if offset + 12 > len(data):
            break
        f1, f2, f3 = struct.unpack_from("<3i", data, offset)
        faces[i] = [f1, f2, f3]
        offset += 12

    return vertices, faces


# ============================================================
# 4. 肌肉参数分析与可视化
# ============================================================
def analyze_muscles(muscles: list[dict]):
    """生成肌肉参数汇总表和可视化图表。"""
    df = pd.DataFrame(muscles)

    # 打印汇总表
    display_cols = [
        "name", "group", "max_force",
        "optimal_fiber_length", "tendon_slack_length", "pennation_angle",
    ]
    cols = [c for c in display_cols if c in df.columns]
    print("\n" + "=" * 70)
    print("肌肉参数汇总表")
    print("=" * 70)
    print(df[cols].to_string(index=False))
    print(f"\n共 {len(df)} 条肌肉")

    # 按肌群统计
    if "group" in df.columns:
        print("\n按肌群统计:")
        print(df.groupby("group")["max_force"].agg(["count", "sum", "mean"]).rename(
            columns={"count": "数量", "sum": "总力(N)", "mean": "平均力(N)"}
        ))

    # 保存 CSV
    csv_path = BASE_DIR / "muscle_parameters.csv"
    df[cols].to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\n参数已保存到: {csv_path}")

    # --- 图1: 最大等长力 ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Stanford VA Upper Extremity Model - Muscle Analysis", fontsize=14)

    ax = axes[0, 0]
    sorted_df = df.sort_values("max_force", ascending=True)
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(sorted_df)))
    ax.barh(sorted_df["name"], sorted_df["max_force"], color=colors)
    ax.set_xlabel("Max Isometric Force (N)")
    ax.set_title("Maximum Isometric Force")
    ax.tick_params(axis="y", labelsize=6)

    # --- 图2: 最优纤维长度 vs 肌腱松弛长度 ---
    ax = axes[0, 1]
    if "optimal_fiber_length" in df.columns and "tendon_slack_length" in df.columns:
        ax.scatter(
            df["optimal_fiber_length"] * 100,
            df["tendon_slack_length"] * 100,
            c=df["max_force"],
            cmap="plasma",
            s=50,
            alpha=0.7,
        )
        for _, row in df.iterrows():
            ax.annotate(
                row["name"], (row["optimal_fiber_length"] * 100, row["tendon_slack_length"] * 100),
                fontsize=4, alpha=0.7,
            )
        ax.set_xlabel("Optimal Fiber Length (cm)")
        ax.set_ylabel("Tendon Slack Length (cm)")
        ax.set_title("Fiber Length vs Tendon Length")
        cbar = plt.colorbar(ax.collections[0], ax=ax)
        cbar.set_label("Max Force (N)")

    # --- 图3: 羽状角分布 ---
    ax = axes[1, 0]
    if "pennation_angle" in df.columns:
        ax.hist(df["pennation_angle"], bins=15, color="steelblue", edgecolor="white")
        ax.set_xlabel("Pennation Angle (deg)")
        ax.set_ylabel("Count")
        ax.set_title("Pennation Angle Distribution")

    # --- 图4: 按肌群的力量对比 ---
    ax = axes[1, 1]
    if "group" in df.columns:
        group_force = df.groupby("group")["max_force"].sum().sort_values()
        group_force.plot(kind="barh", ax=ax, color="coral")
        ax.set_xlabel("Total Max Force (N)")
        ax.set_title("Force by Muscle Group")

    plt.tight_layout()
    fig_path = BASE_DIR / "muscle_analysis.png"
    plt.savefig(fig_path, dpi=150)
    print(f"图表已保存到: {fig_path}")
    plt.show()

    return df


# ============================================================
# 5. 三维骨骼可视化
# ============================================================
def visualize_bones_matplotlib(bones_dir: Path, segments: list[dict]):
    """使用 matplotlib 进行简单的三维骨骼可视化。"""
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title("Upper Extremity Bones (3D)")

    colors = plt.cm.tab20(np.linspace(0, 1, 20))
    color_idx = 0

    for seg in segments:
        if seg["bone_file"] is None:
            continue
        bone_path = bones_dir / seg["bone_file"]
        result = read_bone_asc(bone_path)
        if result is None:
            continue

        vertices, faces = result
        color = colors[color_idx % len(colors)]
        color_idx += 1

        # 绘制三角面片
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        valid_faces = faces[
            (faces[:, 0] < len(vertices))
            & (faces[:, 1] < len(vertices))
            & (faces[:, 2] < len(vertices))
        ]
        if len(valid_faces) > 0:
            triangles = vertices[valid_faces]
            poly = Poly3DCollection(triangles, alpha=0.3)
            poly.set_facecolor(color)
            poly.set_edgecolor("gray")
            ax.add_collection3d(poly)

            # 更新坐标范围
            ax.auto_scale_xyz(
                vertices[:, 0], vertices[:, 1], vertices[:, 2]
            )

        ax.text(
            vertices[:, 0].mean(),
            vertices[:, 1].mean(),
            vertices[:, 2].mean(),
            seg["name"],
            fontsize=5,
            alpha=0.7,
        )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    fig_path = BASE_DIR / "bones_3d.png"
    plt.savefig(fig_path, dpi=150)
    print(f"\n骨骼可视化已保存到: {fig_path}")
    plt.show()


def visualize_bones_pyvista(bones_dir: Path, segments: list[dict]):
    """使用 pyvista 进行交互式三维骨骼可视化（效果更好）。"""
    try:
        import pyvista as pv
    except ImportError:
        print("pyvista 未安装，使用 matplotlib 替代")
        print("安装: pip install pyvista")
        visualize_bones_matplotlib(bones_dir, segments)
        return

    plotter = pv.Plotter()
    plotter.set_background("white")

    colors = [
        "lightblue", "salmon", "lightgreen", "khaki", "plum",
        "lightyellow", "lightcoral", "paleturquoise", "peachpuff", "thistle",
    ]

    for i, seg in enumerate(segments):
        if seg["bone_file"] is None:
            continue
        bone_path = bones_dir / seg["bone_file"]
        result = read_bone_asc(bone_path)
        if result is None:
            continue

        vertices, faces = result
        # pyvista 面格式: [n, v0, v1, v2, ...]
        pv_faces = np.column_stack([
            np.full(len(faces), 3), faces
        ]).ravel()

        mesh = pv.PolyData(vertices, pv_faces)
        color = colors[i % len(colors)]
        plotter.add_mesh(mesh, color=color, opacity=0.7, label=seg["name"])

    plotter.add_legend()
    plotter.show_axes()
    plotter.show()


# ============================================================
# 6. 关节链可视化
# ============================================================
def visualize_joint_chain(joints: list[dict]):
    """可视化关节连接关系（树状图）。"""
    print("\n" + "=" * 50)
    print("关节连接链")
    print("=" * 50)
    for j in joints:
        parent = j.get("parent", "?")
        child = j.get("child", "?")
        print(f"  {parent} --[{j['name']}]--> {child}")


# ============================================================
# 主程序
# ============================================================
def main():
    print("Stanford VA Upper Extremity Model 分析工具")
    print("=" * 50)

    # 解析数据
    print("\n[1/4] 解析肌肉数据...")
    muscles = parse_muscles(MSL_FILE)
    print(f"  找到 {len(muscles)} 条肌肉")

    print("[2/4] 解析骨骼段...")
    segments = parse_segments(JNT_FILE)
    bone_segments = [s for s in segments if s["bone_file"]]
    print(f"  找到 {len(segments)} 个段, 其中 {len(bone_segments)} 个有骨骼几何")

    print("[3/4] 解析关节...")
    joints = parse_joints(JNT_FILE)
    print(f"  找到 {len(joints)} 个关节")

    # 关节链
    visualize_joint_chain(joints)

    # 肌肉分析
    print("\n[4/4] 肌肉参数分析与可视化...")
    df = analyze_muscles(muscles)

    # 骨骼可视化
    print("\n是否进行三维骨骼可视化？")
    choice = input("输入 1=matplotlib, 2=pyvista, 其他=跳过: ").strip()
    if choice == "1":
        visualize_bones_matplotlib(BONES_DIR, bone_segments)
    elif choice == "2":
        visualize_bones_pyvista(BONES_DIR, bone_segments)
    else:
        print("跳过骨骼可视化")

    print("\n分析完成！")


if __name__ == "__main__":
    main()
