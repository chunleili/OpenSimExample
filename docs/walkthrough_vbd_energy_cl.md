# Walkthrough: `vbd_energy.cl` — Houdini VBD 肌肉仿真 OpenCL 内核

> 源文件: `Tutorials/Mine/vbd_energy.cl`
> Side Effects Software (Houdini) 的 VBD (Vertex Block Descent) 可变形体求解器核心能量计算内核。

---

## 1. 总体架构

VBD 是一种**逐顶点的块坐标下降**求解器，用于隐式时间积分。其核心思想：

1. 对网格顶点进行**图着色 (Graph Coloring)**，保证同色顶点互不相邻
2. 按颜色分批并行处理，每批内各顶点独立求解一个 **3x3 局部牛顿步**
3. 每个顶点汇总所有关联单元的**能量梯度 (force)** 和 **Hessian**，求解位移增量 `dx = H^{-1} f`

整个 kernel 处理以下能量项：
- **Stable Neo-Hookean (SNH)** 超弹性体积材料
- **各向异性纤维 (Fiber)** 肌肉方向约束
- **弹簧 (Spring / Distance)** 约束
- **软钉 (Soft Pin)** 约束
- **碰撞 (Collision)** 惩罚力 + 摩擦力
- **地面 (Ground)** 惩罚力
- **惯性 (Inertial)** 项（动态）或 **重力**（准静态）
- **Chebyshev 加速器**

---

## 2. 材料参数：Lame 常数

```c
mu = E / (2 * (1 + v))
lambda = E * v / ((1 + v) * (1 - 2*v))
```

- `E` = 杨氏模量 (stiffness)
- `v` = 泊松比
- `mu` = 剪切模量，控制**形状保持**
- `lambda` = 体积模量，控制**体积保持**

---

## 3. 变形梯度 F 的计算

对于四面体单元，由四个顶点 `p0, p1, p2, p3` 计算：

```
Ds = [p0-p3, p1-p3, p2-p3]   (当前构型的边矩阵, 3x3)
F  = Ds * Dm^{-1}             (变形梯度)
```

- `Dm^{-1}` 是**静止构型边矩阵的逆**，预计算存储在 `restmatrix` 中
- `J = det(F)` 是体积变化比

---

## 4. Stable Neo-Hookean (SNH) 能量

### 4.1 能量密度

基于 [Smith et al. 2018, "Stable Neo-Hookean Flesh Simulation"](https://graphics.pixar.com/library/StableElasticity/)：

重新参数化（Section 3.4），令 `alpha = 1 + mu / (lambda + mu)`：

$$\Psi_{\text{SNH}} = \frac{\mu}{2} \|F\|_F^2 + \frac{\lambda+\mu}{2} (J - \alpha)^2$$

其中 $\|F\|_F^2 = \text{tr}(F^T F)$ 是 Frobenius 范数平方。

### 4.2 第一 Piola-Kirchhoff 应力 (PK1)

$$P = \mu \cdot F + (\lambda+\mu)(J - \alpha) \cdot \frac{\partial J}{\partial F}$$

其中 $\frac{\partial J}{\partial F}$ 的每一列是 F 另外两列的叉积（Eqn. 19）：

```
pJpF = [F1 x F2,  F2 x F0,  F0 x F1]
```

### 4.3 Hessian (d^2 Psi / dF^2)

9x9 矩阵，由三部分组成：

$$H_F = \mu \cdot I_9 + (\lambda+\mu) \cdot \text{vec}(\frac{\partial J}{\partial F}) \cdot \text{vec}(\frac{\partial J}{\partial F})^T + \text{HessianJ}$$

其中 `HessianJ` 是 J 对 F 的二阶导数，通过 `crossProduct` 函数构造（Eqn. 29, Section 4.5），
本质上是一个**反对称叉积矩阵**的分块组合。

---

## 5. 各向异性纤维 (Fiber) 约束 — 肌肉方向

### 5.1 核心思想

在肌肉仿真中，纤维方向 `w`（材料空间中的单位向量）表示**肌纤维走向**。
纤维约束是一个**各向异性 ARAP (As-Rigid-As-Possible)** 能量，沿纤维方向允许可控的伸缩。

### 5.2 不变量

- $I_5 = \|F^T w\|^2 = w^T (F^T F) w$：纤维方向的拉伸平方
- $I_4 = w^T S w$：纤维方向的有符号拉伸（通过 SVD 得到拉伸矩阵 $S = V \Sigma V^T$）
- $\text{sign}(I_4)$：检测纤维方向是否**翻转**（inversion）

### 5.3 能量与导数

$$\frac{\partial \Psi_{\text{fiber}}}{\partial I_5} = \mu \left(1 - \text{sign}(I_4) \cdot \text{fiberscale} \cdot \frac{1}{\sqrt{I_5}}\right)$$

- `fiberscale = 0` → 标准各向异性 ARAP（纤维方向和其他方向等同）
- `fiberscale > 0` → 纤维方向可自由伸缩（模拟肌肉沿纤维方向的柔软性）
- `fiberscale = 1` → 纤维方向完全不提供刚度（静止长度处零力）

PK1 贡献：
$$P_{\text{fiber}} = \frac{\partial \Psi}{\partial I_5} \cdot F \cdot A, \quad A = w w^T$$

### 5.4 Hessian 贡献

$$H_{\text{fiber}} = \frac{\partial \Psi}{\partial I_5} \cdot \text{kron}(A, I_3) + \frac{\partial^2 \Psi}{\partial I_5^2} \cdot \text{vec}(FA) \cdot \text{vec}(FA)^T$$

- 第一项：Kronecker 积 $A \otimes I_3$，对应 $I_5$ 的 Hessian
- 第二项：仅当 `fiberscale > 0` 时存在，$\frac{\partial^2 \Psi}{\partial I_5^2} = \mu \cdot \text{sign}(I_4) \cdot \text{fiberscale} \cdot I_5^{-3/2}$

### 5.5 约束类型

- `TETARAPNORMVOL`：SNH 体积材料 **+** 可选的纤维约束（`kfiber > 0` 时叠加）
- `TETFIBERNORM`：纯纤维约束（无体积项）

---

## 6. 弹簧约束 (Spring / Distance)

### 6.1 能量

$$E = \frac{1}{2} k_s (l - l_0)^2$$

### 6.2 力与 Hessian

$$f = -k_s (l - l_0) \hat{n}, \quad \hat{n} = \frac{p_0 - p_1}{\|p_0 - p_1\|}$$

$$H = k_s \hat{n} \hat{n}^T + k_s \frac{l - l_0}{l} (I - \hat{n}\hat{n}^T)$$

**注意**：第二项仅在**拉伸** ($l > l_0$) 时添加。压缩时省略以保证 Hessian 半正定
（Section 13.4, Dynamic Deformables; Section 3.1, Choi "Stable but Responsive Cloth"）。

---

## 7. 从 F 空间到 x 空间的转换

能量的梯度和 Hessian 首先在 **F 空间** (9-dim) 计算，然后通过链式法则转换到**位置空间** (3-dim)：

### 7.1 力（梯度）

$$f_i = -V \cdot P^T \cdot d_i$$

其中 $d_i$ 是 $D_m^{-1}$ 对应当前顶点的行（第4个顶点的行为 $-d_0 - d_1 - d_2$），$V$ 是静止体积。

### 7.2 Hessian

$$J_i = V \cdot \left(\frac{\partial F}{\partial x_i}\right)^T \cdot \frac{\partial^2 \Psi}{\partial F^2} \cdot \frac{\partial F}{\partial x_i}$$

这是一个 9x9 到 3x3 的张量"链式法则"收缩，代码中直接展开乘法以利用对称性。

---

## 8. SPD 投影（正定性保证）

高体积刚度可能导致 Hessian 出现负特征值。代码通过**绝对值特征值过滤**投影回 SPD：

```
H → L * diag(|eig_i|) * L^T
```

基于 ["Stabler Neo-Hookean Simulation: Absolute Eigenvalue Filtering for Projected Newton"]。

这保证了每个局部 Hessian 块是正定的，从而整体 Hessian 也是正定的。

---

## 9. 阻尼 (Damping)

Rayleigh 型刚度阻尼，按时间步预缩放 `kdamp = dampingratio / dt`：

$$f_{\text{damp}} = -k_{\text{damp}} \cdot H \cdot (x - x_{\text{prev}})$$

$$H_{\text{damped}} = (1 + k_{\text{damp}}) \cdot H$$

---

## 10. 碰撞处理

### 10.1 Point-Triangle 碰撞

- 使用预计算的碰撞 UV 和法线（`hituv`, `hitnml`），或实时计算最近点（`closestpttriangle`）
- 碰撞深度：`d = hitrad - dot(diff, n)`
- 碰撞力：`f_col = k_col * d * n`（惩罚力）
- 碰撞 Hessian：`H_col = k_col * n * n^T`
- 力按重心坐标分配到各顶点

### 10.2 Edge-Edge 碰撞

- 通过 `closestptedges` 计算两边最近点参数 `(u, v)`
- 类似的惩罚力公式，按插值权重分配

### 10.3 地面碰撞

$$d = r - (p - \text{origin}) \cdot n$$

当 `d > 0` 时施加惩罚力 `f = k_col * d * n`。

### 10.4 摩擦力 (IPC Friction)

基于 [IPC Toolkit](https://github.com/ipc-sim/ipc-toolkit) 的光滑摩擦模型：

1. 构建切平面基 $T = [b_0, b_1]$ (3x2 矩阵)
2. 投影位移到切平面：$u = T^T \Delta x$ (2D 切向位移)
3. Mollified friction：

$$f_1(u) / \|u\| = \begin{cases} 1/\|u\| & \|u\| > \epsilon_U \\ (-\|u\|/\epsilon_U + 2) / \epsilon_U & \|u\| \leq \epsilon_U \end{cases}$$

4. 摩擦力：$f_{\text{fric}} = -\mu_s \lambda \cdot f_1/\|u\| \cdot T u$
5. 摩擦 Hessian：$H_{\text{fric}} = \mu_s \lambda \cdot f_1/\|u\| \cdot T T^T$

---

## 11. 惯性项与时间积分

### 11.1 动态模式

$$f_{\text{inertia}} = \frac{m}{\Delta t^2} (\tilde{x} - x)$$

$$H_{\text{inertia}} = \frac{m}{\Delta t^2} I_3$$

其中 $\tilde{x}$ 是惯性预测位置（`inertial`），包含了速度外推和外力。

### 11.2 准静态模式 (`QUASISTATIC`)

不含惯性项，仅加外力（重力）：

$$f_{\text{ext}} = m \cdot g$$

---

## 12. 逐顶点求解（主 kernel `solveVBD`）

### 12.1 并行策略

- 每个工作组 (workgroup, BLOCKSIZE=16 个线程) 处理**一个顶点**
- 组内线程**分摊**该顶点关联的多个 prim 和碰撞的计算
- 通过 **local memory + binary reduction** 汇总 `f` (3D) 和 `H` (3x3)

### 12.2 求解

汇总所有能量贡献后，thread 0 执行：

```
dx = H^{-1} * f     (通过 LDLT 分解求解 3x3 系统)
x_new = x + dx
```

由于 H 已经保证 SPD，LDLT 分解是安全的。

---

## 13. Chebyshev 加速

VBD 收敛较慢（类似 Gauss-Seidel），通过 **Chebyshev 半迭代加速** 提速：

$$\omega_1 = 1, \quad \omega_2 = \frac{2}{2 - \rho^2}, \quad \omega_{k+1} = \frac{4}{4 - \rho^2 \omega_k}$$

$$x_{\text{acc}} = \omega (x_{\text{new}} - x_{\text{last}}) + x_{\text{last}}$$

- `rho` 是谱半径估计
- 碰撞点 (`fallback`) 和零质量点**不加速**，避免振荡

相关 kernels：`updateOmega`, `copyToPrevIter`, `applyOmega`。

---

## 14. 自适应碰撞刚度

### 14.1 `initAdaptiveStiffness`

初始化时估算碰撞刚度缩放因子 $\kappa$：

$$\kappa = \max\left(\kappa_{\text{prev}},\; -\frac{g_c \cdot g_E}{\|g_c\|^2}\right)$$

- $g_c$ = 碰撞能量梯度
- $g_E$ = 材料能量梯度 + 惯性梯度
- 物理含义：使碰撞力至少能平衡材料力在碰撞方向的分量

### 14.2 `updateAdaptiveCollisionStiffness`

迭代中如果碰撞深度增大（碰撞恶化），则逐步增大碰撞刚度：

```
if maxdepth > prevmaxdepth:
    scale *= stiffmult      (乘法增长)
    scale = min(scale, maxstiffscale)
```

### 14.3 `updateAdaptiveStiffness`（体积刚度）

对体积约束误差 $C = J - 1$ 进行自适应增强：

```
curstiff += beta * |C|
scale = min(1, curstiff / maxstiff)
```

---

## 15. 关键数据结构与约定

| 属性 | 存储 | 说明 |
|------|------|------|
| `restlength` | per-prim | 弹簧静止长度 / 四面体静止体积 |
| `restmatrix` | per-prim (9 floats) | $D_m^{-1}$，静止构型边矩阵的逆 |
| `restvector` | per-prim (4 floats) | 纤维方向 `w` / Soft Pin 目标位置 |
| `stiffness` | per-prim | 形状刚度 ($\mu$) |
| `volumestiffness` | per-prim | 体积刚度 ($\lambda$) |
| `fiberstiffness` | per-prim | 纤维额外刚度 |
| `fiberscale` | per-prim | 纤维伸缩自由度 (0=各向同性, 1=完全自由) |
| `dampingratio` | per-prim | 阻尼系数 |
| `mass` | per-point | 质量 (0=固定点) |
| `pscale` | per-point | 碰撞半径 |
| `type_hash` | per-prim | 约束类型 (TETARAPNORMVOL, TETFIBERNORM, PIN, DISTANCE) |

---

## 16. 参考文献

- Smith, Schaefer, Kim. **"Stable Neo-Hookean Flesh Simulation"**, SIGGRAPH 2018
- Kim. **"Stabler Neo-Hookean Simulation: Absolute Eigenvalue Filtering for Projected Newton"**
- Kim, Eberle. **"Dynamic Deformables"**, SIGGRAPH Course
- Choi, Ko. **"Stable but Responsive Cloth"**, SIGGRAPH 2002
- Li et al. **"IPC: Incremental Potential Contact"**, SIGGRAPH 2020
- Ericson. **"Real-Time Collision Detection"**
- Pixar **Dynamic Deformables** 课程中的 `ComputePFPx`
