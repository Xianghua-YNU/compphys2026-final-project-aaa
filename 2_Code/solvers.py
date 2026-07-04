"""
数值积分器模块：Euler 方法与 Leapfrog（速度 Verlet）方法

Euler:      一阶精度, 非辛, 能量单调漂移
Leapfrog:   二阶精度, 辛算法, 能量长期有界振荡
"""

import numpy as np
from physics import gravitational_acceleration


def euler_integrate(positions, velocities, masses, dt, n_steps):
    """
    显式 Euler 法（一阶精度）

    迭代格式:
        v_{n+1} = v_n + a_n·Δt
        r_{n+1} = r_n + v_n·Δt

    特点: 简单但不保持相空间体积 → 能量单调漂移

    返回:
        pos_history: (n_steps+1, N, 3)
        vel_history: (n_steps+1, N, 3)
    """
    n_bodies = len(masses)
    pos_history = np.zeros((n_steps + 1, n_bodies, 3))
    vel_history = np.zeros((n_steps + 1, n_bodies, 3))

    pos_history[0] = positions.copy()
    vel_history[0] = velocities.copy()

    for step in range(n_steps):
        acc = gravitational_acceleration(pos_history[step], masses)
        vel_history[step + 1] = vel_history[step] + acc * dt
        pos_history[step + 1] = pos_history[step] + vel_history[step] * dt

    return pos_history, vel_history


def leapfrog_integrate(positions, velocities, masses, dt, n_steps):
    """
    速度 Verlet (Leapfrog) 法（二阶辛积分器）

    迭代格式:
        v_{1/2} = v_0 + ½·a_0·Δt
        r_{n+1} = r_n + v_{n+1/2}·Δt
        a_{n+1} = f(r_{n+1})
        v_{n+1} = v_{n+1/2} + ½·a_{n+1}·Δt
        v_{n+3/2} = v_{n+1} + ½·a_{n+1}·Δt

    辛性质: 准确保留相空间体积 → 能量长期有界振荡，无系统漂移

    返回:
        pos_history: (n_steps+1, N, 3)
        vel_history: (n_steps+1, N, 3)
    """
    n_bodies = len(masses)
    pos_history = np.zeros((n_steps + 1, n_bodies, 3))
    vel_history = np.zeros((n_steps + 1, n_bodies, 3))

    pos_history[0] = positions.copy()
    vel_history[0] = velocities.copy()

    # 半步速度: v(dt/2) = v(0) + a(0)·dt/2
    acc = gravitational_acceleration(pos_history[0], masses)
    vel_half = vel_history[0] + 0.5 * acc * dt

    for step in range(n_steps):
        # 位置整步: r(t+dt) = r(t) + v(t+dt/2)·dt
        pos_history[step + 1] = pos_history[step] + vel_half * dt
        # 新加速度: a(t+dt)
        acc_new = gravitational_acceleration(pos_history[step + 1], masses)
        # 速度整步: v(t+dt) = v(t+dt/2) + a(t+dt)·dt/2
        vel_history[step + 1] = vel_half + 0.5 * acc_new * dt
        # 下半步速度: v(t+3dt/2) = v(t+dt) + a(t+dt)·dt/2
        vel_half = vel_history[step + 1] + 0.5 * acc_new * dt

    return pos_history, vel_history
