# N体轨道积分——太阳系简化模型与 Leapfrog 守恒性分析

## 项目概述

本项目实现了基于 Leapfrog（速度 Verlet）和 Euler 方法的 N 体轨道积分，
用于模拟太阳系简化模型并分析辛积分器的长期能量守恒优势。

## 文件结构

```
2_Code/
├── physics.py       # 物理模块：引力加速度、能量、角动量、轨道根数、初始化
├── solvers.py       # 数值积分模块：Euler (一阶非辛) 和 Leapfrog (二阶辛)
├── analysis.py      # 分析与可视化：能量/角动量历史、Kozai-Lidov 追踪、收敛性
├── main.py          # 主程序：参数设置、三类模拟运行、结果输出
├── requirements.txt # 依赖清单
└── README.md        # 本文件
```

## 运行步骤

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 运行主程序

```bash
python main.py
```

### 3. 输出文件

**图像文件** (`../1_论文/assets/`):

| 文件 | 内容 |
|------|------|
| `orbit_2body_euler.png` | Euler 两体轨道 (螺旋外扩 -> 能量漂移) |
| `orbit_2body_leapfrog.png` | Leapfrog 两体轨道 (闭合椭圆 -> 能量守恒) |
| `energy_comparison_2body.png` | 能量时间序列对比 (Leapfrog vs Euler) |
| `energy_error_2body.png` | 相对能量误差半对数图 |
| `angular_momentum_2body.png` | 角动量守恒对比 |
| `comprehensive_2body.png` | 六合一综合对比面板 (推荐用于论文) |
| `orbit_3body_leapfrog_2d.png` | 三体系统 x-y 轨道投影 |
| `orbit_3body_leapfrog_3d.png` | 三体系统 3D 轨道 |
| `energy_3body_leapfrog.png` | 三体系统能量守恒 |
| `kozai_lidov_evolution.png` | Kozai-Lidov 机制四面板分析 |
| `convergence_test.png` | 收敛性与精度阶数验证 (含理论斜率) |

**数据文件** (`../3_Data/`):
- `simulation_2body_leapfrog.csv`
- `simulation_2body_euler.csv`
- `convergence_test.csv`

## 核心算法

### 物理方程

牛顿万有引力（含软化参数 ε）：
$$ \mathbf{a}_i = \sum_{j 
eq i} G rac{m_j}{(|r_{ij}|^2 + arepsilon^2)^{3/2}} \mathbf{r}_{ij} $$

系统总能量：
$$ E = \sum_i rac{1}{2} m_i v_i^2 - \sum_{i<j} G rac{m_i m_j}{r_{ij}} $$

### 数值方法

**Euler (一阶精度, 非辛)**：
$$ \mathbf{v}_{n+1} = \mathbf{v}_n + \mathbf{a}_n \Delta t $$
$$ \mathbf{r}_{n+1} = \mathbf{r}_n + \mathbf{v}_n \Delta t $$

**Leapfrog / 速度 Verlet (二阶精度, 辛)**：
$$ \mathbf{v}_{n+1/2} = \mathbf{v}_n + 	frac{1}{2} \mathbf{a}_n \Delta t $$
$$ \mathbf{r}_{n+1} = \mathbf{r}_n + \mathbf{v}_{n+1/2} \Delta t $$
$$ \mathbf{a}_{n+1} = \mathbf{f}(\mathbf{r}_{n+1}) $$
$$ \mathbf{v}_{n+1} = \mathbf{v}_{n+1/2} + 	frac{1}{2} \mathbf{a}_{n+1} \Delta t $$

### 单位系统

- 长度: AU, 质量: M_sun, 时间: yr -> G = 4pi^2

## 模拟参数

| 模拟 | 天体 | dt (yr) | 时长 (yr) | 方法 |
|------|------|---------|-----------|------|
| 两体系统 | 太阳-地球 | 0.001 | 100 | Euler + Leapfrog |
| 三体系统 | 太阳-木星-小行星 | 0.01 | 500 | Leapfrog |
| 收敛性测试 | 太阳-地球 | 0.001-0.1 | 10 | Euler + Leapfrog |

## 关键发现

1. Leapfrog 能量误差保持在 10^-10 以下（100 年模拟），Euler 能量漂移超过 65%
2. 误差比 (Euler/Leapfrog) 超过 10^13
3. 收敛性测试揭示 Euler 的非辛性质导致系统性能量漂移
4. 三体系统成功观测到 Kozai-Lidov 机制的 e-i 耦合振荡

## AI 使用声明

本项目代码使用 AI 辅助编写，包括：
- 算法框架设计与模块划分
- 引力加速度的半向量化优化
- 可视化模块开发
