# Stanford VA Upper Extremity Model — 数据格式说明

> 模型来源: Holzbaur, Murray, and Delp. *Annals of Biomedical Engineering*, 2005.
> 格式: SIMM (Software for Interactive Musculoskeletal Modeling)
> 下载: https://simtk.org/projects/up-ext-model

---

## 文件总览

| 文件 | 格式 | 内容 |
|------|------|------|
| `Stanford VA upper limb model.jnt` | 文本 | 骨骼段 (Segments)、关节 (Joints)、自由度 (DOFs) |
| `Stanford VA upper limb model.msl` | 文本 | 肌肉路径、力学参数、力-长度曲线 |
| `bones/*.asc` | 二进制 | 骨骼三维网格几何（仅用于可视化） |

---

## 1. `.jnt` 文件 — 关节与骨骼定义

### 1.1 Segments（骨骼段 / 刚体）

```
beginsegment humerus        ← 段名称（肱骨）
bone humerus.asc            ← 关联的三维几何文件
endsegment
```

- 每个 segment 代表一个**刚体**
- 有些段没有 `bone`（如 `humphant`），是纯数学辅助段，用于分解复杂的多轴旋转
- 模型共 61 个段，其中 33 个有骨骼几何

### 1.2 Joints（关节）

```
beginjoint elbow                         ← 关节名
segments humerus ulna                    ← 父段 → 子段
order t r1 r2 r3                         ← 变换顺序：平移 t → 旋转 r1 → r2 → r3
axis1 1.0  0.0  0.0                      ← 第1旋转轴方向向量
axis2 0.0  1.0  0.0                      ← 第2旋转轴方向向量
axis3 0.049 0.037 0.998                  ← 第3旋转轴方向向量
tx constant  0.0061                      ← X方向平移（固定值，单位：米）
ty constant -0.2904                      ← Y方向平移
tz constant -0.0123                      ← Z方向平移
r1 constant  0.0                         ← 绕轴1旋转角（固定为0）
r2 constant  0.0                         ← 绕轴2旋转角（固定为0）
r3 function f14(elbow_flexion)           ← 绕轴3旋转 = 自由度 elbow_flexion 的函数
endjoint
```

**关键字段：**

| 字段 | 含义 |
|------|------|
| `segments A B` | 父段 A 连接到子段 B |
| `order t r1 r2 r3` | 齐次变换的应用顺序（平移和旋转的组合） |
| `axis1/2/3` | 三个旋转轴的方向向量（在父段坐标系中） |
| `tx/ty/tz` | 平移分量（米） |
| `r1/r2/r3` | 旋转分量（弧度） |
| `constant X` | 该自由度被锁定为固定值 X |
| `function fN(dof)` | 该分量由广义坐标 `dof` 通过样条函数 fN 驱动 |

### 1.3 自由度（广义坐标，共 15 个）

| 自由度名称 | 含义 |
|------------|------|
| `shoulder_elv` | 肩关节抬高面角度 |
| `elv_angle` | 肩关节抬高角 |
| `shoulder_rot` | 肩关节内/外旋 |
| `elbow_flexion` | 肘关节屈/伸 |
| `pro_sup` | 前臂旋前/旋后 |
| `wrist_flex` | 腕关节屈/伸 |
| `wrist_dev` | 腕关节尺偏/桡偏 |
| 拇指 × 4 | 腕掌关节屈伸、外展，掌指关节屈伸，指间关节屈伸 |
| 食指 × 4 | 掌指关节外展、屈伸，近-中指间屈伸，中-远指间屈伸 |

### 1.4 关节连接链（主要路径）

```
ground → thorax → clavicle → scapula → humerus → ulna → radius
                                                          ↓
                                            腕骨 (lunate, scaphoid, ...)
                                                          ↓
                                            掌骨 (1mc ~ 5mc)
                                                          ↓
                                            指骨 (proxph → midph → distph)
```

> 注：肩关节使用多个辅助段 (clavphant, scapphant, humphant 等) 将三维旋转分解为依次绕不同轴的旋转。

---

## 2. `.msl` 文件 — 肌肉定义

### 2.1 默认肌肉模型（Hill 型肌肉）

```
beginmuscle defaultmuscle

beginactiveforcelengthcurve           ← 主动力-长度曲线（归一化）
(归一化纤维长度, 归一化力)
( 0.401,  0.000)                      ← 纤维过短时无法产力
( 1.045,  0.993)                      ← 最优长度附近产生最大力
( 1.619,  0.000)                      ← 纤维过长时无法产力
endactiveforcelengthcurve

beginpassiveforcelengthcurve          ← 被动力-长度曲线
(归一化纤维长度, 归一化力)
( 0.998,  0.000)                      ← 被动力在纤维未被拉长时为零
( 1.750,  2.000)                      ← 拉长到 1.75 倍最优长度时被动力 = 2 倍最大力
endpassiveforcelengthcurve

begintendonforcelengthcurve           ← 肌腱力-应变曲线
(肌腱应变, 归一化力)
( 0.000,  0.000)                      ← 无应变时无力
( 0.012,  0.227)                      ← 应变 1.2% 时力 = 22.7% 最大力
endtendonforcelengthcurve

endmuscle
```

这三条曲线定义了 Hill 型肌肉模型的核心力学行为：
- **主动力-长度曲线**: 肌纤维的收缩力随长度变化的关系
- **被动力-长度曲线**: 肌肉被动拉伸时的弹性力
- **肌腱力-应变曲线**: 肌腱的弹性特性

### 2.2 具体肌肉定义

```
beginmuscle DELT1                            ← 肌肉名称（三角肌前束）
beginpoints
 0.00896 -0.11883  0.00585 segment humerus   ← 附着点 (x, y, z) 在肱骨坐标系，单位：米
 0.01623 -0.11033  0.00412 segment humerus   ← 途经点
 0.04347 -0.03252  0.00099 segment scapula   ← 途经点（肩胛骨）
-0.01400  0.01106  0.08021 segment clavicle  ← 起点（锁骨）
endpoints
begingroups
shoulder                                     ← 所属肌群
endgroups
max_force           1142.6                   ← 最大等长收缩力 (N)
optimal_fiber_length  0.0976                 ← 最优肌纤维长度 (m)
tendon_slack_length   0.0930                 ← 肌腱松弛长度 (m)
pennation_angle      22.0                    ← 羽状角 (度)
wrapobject delt2hum                          ← 绕行对象（防止穿透骨骼）
endmuscle
```

### 2.3 肌肉参数含义

| 参数 | 含义 | 说明 |
|------|------|------|
| `max_force` | 最大等长收缩力 (N) | = 比张力 × PCSA（肌肉横截面积）。肩/肘肌肉使用 140 N/cm²，前臂/手使用 45 N/cm² |
| `optimal_fiber_length` | 最优肌纤维长度 (m) | 肌纤维在此长度时主动力-长度曲线达到峰值 |
| `tendon_slack_length` | 肌腱松弛长度 (m) | 肌腱在此长度以下不产生张力，决定肌肉的工作范围 |
| `pennation_angle` | 羽状角 (°) | 肌纤维与肌腱方向的夹角，影响力的传递效率（有效力 = 肌纤维力 × cos(角度)） |
| `beginpoints...endpoints` | 肌肉路径经过点 | 定义肌肉的几何走向，用于计算力臂 |
| `wrapobject` | 绕行约束对象 | 让肌肉沿骨骼/关节表面滑动，模拟真实解剖路径 |

### 2.4 肌群分类

| 肌群 | 包含肌肉 |
|------|----------|
| `shoulder` | DELT1/2/3, SUPSP, INFSP, SUBSC, TMIN, TMAJ, PECM1/2/3, LAT1/2/3, CORB |
| `elbow` | TRIlong/lat/med, BIClong/short, BRD, ANC, SUP |
| `forearm` | PT, PQ, FCR, FCU, ECRB/L, ECU, EDC 等 |
| `hand` | FDS, FDP, EDC, EIP, FPL, EPL, APL 等 |

模型共 **50 条肌肉**（含多头/多束分支）。

---

## 3. `bones/*.asc` — 骨骼三维网格

大端序 (big-endian) 二进制文件，共 33 个骨骼文件。

### 3.1 文件头 (20 字节)

| 偏移 | 类型 | 含义 | 示例 (humerus.asc) |
|------|------|------|---------------------|
| 0 | int32 | 顶点数 | 133 |
| 4 | int32 | 三角面片数 | 309 |
| 8 | int32 | 法线向量数 | 588 |
| 12 | int32 | 法线分量总数 (= 法线数 × 3) | 1764 |
| 16 | int32 | 每面顶点数 | 3 (三角形) |

### 3.2 数据区

| 区段 | 格式 | 说明 |
|------|------|------|
| 顶点坐标 | 每个顶点 3 个 float64 (大端序) | x, y, z 坐标，单位：米 |
| 法线向量 | 每个法线 3 个 float64 (大端序) | 用于光照渲染 |
| 面索引 | 整数数组 | 顶点索引和法线索引 |

### 3.3 骨骼文件列表

```
thorax.asc      胸廓          scapula.asc     肩胛骨
clavicle.asc    锁骨          humerus.asc     肱骨
ulna.asc        尺骨          radius.asc      桡骨
lunate.asc      月骨          scaphoid.asc    舟骨
pisiform.asc    豌豆骨        triquetrum.asc  三角骨
capitate.asc    头状骨        trapezium.asc   大多角骨
trapezoid.asc   小多角骨      hamate.asc      钩骨
1mc~5mc.asc     第1~5掌骨
thumbprox.asc   拇指近节指骨  thumbdist.asc   拇指远节指骨
2~5proxph.asc   第2~5近节指骨
2~5midph.asc    第2~5中节指骨
2~5distph.asc   第2~5远节指骨
```

> 注：这些几何数据**仅用于可视化**，不参与力学计算。

---

## 4. 整体模型工作原理

```
┌─────────────────────────────────────────────┐
│  骨骼段 (Segments)    ← 刚体               │
│      ↕                                      │
│  关节 (Joints)        ← 运动学约束          │
│      ↕                                      │
│  自由度 (DOFs)        ← 广义坐标 (15个)     │
│      ↕                                      │
│  肌肉 (Muscles)       ← Hill型力产生器      │
│      │                                      │
│      ├─ 附着在骨骼段上                       │
│      ├─ 跨越一个或多个关节                   │
│      ├─ 通过 Hill 模型计算可产生的力         │
│      └─ 力 × 力臂 = 关节力矩               │
└─────────────────────────────────────────────┘
```

**典型使用流程：**

1. 给定一组关节角度（如肘屈 90°）
2. 由运动学计算每条肌肉的当前长度和力臂
3. 由 Hill 模型和肌肉参数计算每条肌肉的最大产力能力
4. 通过优化算法（如静态优化）分配肌肉激活水平
5. 得到关节力矩、关节反力等结果

---

## 5. 参考文献

- Holzbaur KRS, Murray WM, Delp SL. *A model of the upper extremity for simulating musculoskeletal surgery and analyzing neuromuscular control.* Annals of Biomedical Engineering, 2005.
- 肘部肌肉数据: Murray WM. PhD dissertation, Appendix A & B.
- 肩部力臂: Hughes 1998, Liu 1997, Otis 1994.
- 肩部肌肉参数: Langenderfer 2004.
- 前臂/手肌肉: Loren 1996, Brand & Hollister 1993, Lieber 1990/2000.

---

## 6. 单位约定

| 量 | 单位 |
|----|------|
| 长度（坐标、肌纤维长度、肌腱长度） | 米 (m) |
| 力 | 牛顿 (N) |
| 角度（关节文件中） | 弧度 (rad) |
| 羽状角（肌肉文件中） | 度 (°) |
| 比张力 | N/cm² (肩/肘: 140, 前臂/手: 45) |
