"""
物理模块：N体引力系统的核心计算

包含引力加速度、能量、角动量、轨道根数等物理量的计算，
以及太阳系简化模型的初始化函数。

单位系统: 长度=AU, 质量=M_sun, 时间=yr → G = 4π²
"""

import numpy as np

G = 4 * np.pi ** 2


def gravitational_acceleration(positions, masses, softening=1e-4):
    """
    计算N体系统中每个天体受到的引力加速度（半向量化）

    a_i = Σ_{j≠i} G·m_j·r_ij / (|r_ij|² + ε²)^(3/2)

    参数:
        positions: (N, 3) 位置 [AU]
        masses:    (N,) 质量 [M_sun]
        softening: 软化参数 ε，防止 r→0 时奇点 [AU]

    返回:
        accelerations: (N, 3) 加速度 [AU/yr²]
    """
    n_bodies = len(masses)
    accelerations = np.zeros_like(positions)

    for i in range(n_bodies):
        r_ij = positions - positions[i]
        r2 = np.sum(r_ij ** 2, axis=1) + softening ** 2
        r3 = r2 ** 1.5
        r3[i] = np.inf  # 排除自相互作用
        accelerations[i] = G * np.sum(
            masses[:, np.newaxis] * r_ij / r3[:, np.newaxis], axis=0
        )

    return accelerations


def compute_total_energy(positions, velocities, masses):
    """
    计算系统总机械能 E = K + U

    K = ½ Σ m_i·v_i²
    U = -G Σ_{i<j} m_i·m_j / r_ij

    返回:
        total_energy: 标量 [M_sun·AU²/yr²]

    注: 对于有界轨道，E < 0
    """
    kinetic = 0.5 * np.sum(masses[:, np.newaxis] * velocities ** 2)

    potential = 0.0
    n = len(masses)
    for i in range(n):
        for j in range(i + 1, n):
            r = np.linalg.norm(positions[j] - positions[i])
            potential -= G * masses[i] * masses[j] / r

    return kinetic + potential


def compute_angular_momentum(positions, velocities, masses):
    """
    计算系统总角动量矢量 L = Σ m_i (r_i × v_i)

    返回:
        angular_momentum: (3,) [M_sun·AU²/yr]
    """
    L = np.zeros(3)
    for i in range(len(masses)):
        L += masses[i] * np.cross(positions[i], velocities[i])
    return L


def compute_orbital_elements(positions, velocities, masses,
                              body_idx, central_idx=0):
    """
    计算天体 body_idx 相对中心天体 central_idx 的瞬时开普勒轨道根数

    使用二体问题公式（忽略其他天体的扰动）:
      - 比能量 ε = v²/2 - μ/r → 半长轴 a = -μ/(2ε)
      - 偏心率矢量 e = (v×h)/μ - r̂
      - 倾角 cos i = h_z / |h|

    返回:
        dict: {semi_major_axis, eccentricity, inclination_deg,
               specific_energy, specific_angular_momentum}
    """
    mu = G * (masses[central_idx] + masses[body_idx])

    r_vec = positions[body_idx] - positions[central_idx]
    v_vec = velocities[body_idx] - velocities[central_idx]
    r = np.linalg.norm(r_vec)
    v = np.linalg.norm(v_vec)

    specific_energy = 0.5 * v ** 2 - mu / r
    if abs(specific_energy) > 1e-15:
        semi_major_axis = -mu / (2 * specific_energy)
    else:
        semi_major_axis = np.inf

    h_vec = np.cross(r_vec, v_vec)
    h = np.linalg.norm(h_vec)

    e_vec = np.cross(v_vec, h_vec) / mu - r_vec / r
    eccentricity = np.linalg.norm(e_vec)

    inclination = np.arccos(np.clip(h_vec[2] / h, -1.0, 1.0)) if h > 1e-15 else 0.0

    return {
        'semi_major_axis': semi_major_axis,
        'eccentricity': eccentricity,
        'inclination_deg': np.degrees(inclination),
        'specific_energy': specific_energy,
        'specific_angular_momentum': h,
    }


# ============================================================
# 初始化
# ============================================================

def init_solar_system_2body():
    """
    太阳-地球两体系统

    太阳位于原点静止，地球在 x=1 AU 处，具有圆轨道切向速度。
    圆轨道条件: v = √(GM/r)，离心力 = 引力
    """
    AU = 1.0
    M_sun = 1.0
    earth_mass = 3.003e-6

    positions = np.array([
        [0.0, 0.0, 0.0],
        [AU, 0.0, 0.0],
    ])

    v_earth = np.sqrt(G * M_sun / AU)
    velocities = np.array([
        [0.0, 0.0, 0.0],
        [0.0, v_earth, 0.0],
    ])

    masses = np.array([M_sun, earth_mass])
    names = ['Sun', 'Earth']

    return positions, velocities, masses, names


def init_solar_system_3body():
    """
    太阳-木星-小行星三体系统

    用于研究 Kozai-Lidov 机制:
    在木星引力扰动下，小行星轨道的偏心率 e 与倾角 i 发生
    周期性耦合振荡 (e ↔ i)，满足 √(1-e²) cos i ≈ 常数。

    初始条件: 小行星 a=3.0 AU, e=0.3, i=30°, 从近心点出发
    """
    M_sun = 1.0
    jupiter_mass = 9.5479e-4
    asteroid_mass = 1e-10

    jupiter_a = 5.2034
    jupiter_v = np.sqrt(G * M_sun / jupiter_a)

    asteroid_a = 3.0
    asteroid_e = 0.3
    asteroid_inc = np.pi / 6
    r_peri = asteroid_a * (1 - asteroid_e)
    # 活力公式: v² = GM(2/r - 1/a)
    v_ast = np.sqrt(G * M_sun * (2 / r_peri - 1 / asteroid_a))

    positions = np.array([
        [0.0, 0.0, 0.0],
        [jupiter_a, 0.0, 0.0],
        [asteroid_a * (1 - asteroid_e), 0.0, 0.0],
    ])

    velocities = np.array([
        [0.0, 0.0, 0.0],
        [0.0, jupiter_v, 0.0],
        [0.0, v_ast * np.cos(asteroid_inc), v_ast * np.sin(asteroid_inc)],
    ])

    masses = np.array([M_sun, jupiter_mass, asteroid_mass])
    names = ['Sun', 'Jupiter', 'Asteroid']

    return positions, velocities, masses, names
