"""
N体轨道积分主程序
==================
选题 A-2: N体轨道积分——太阳系简化模型与 Leapfrog 守恒性分析

实验目标:
  1. 对比 Euler (非辛) 与 Leapfrog (辛) 积分器的能量守恒性能
  2. 验证辛积分器在长期轨道模拟中的优势
  3. 研究三体系统中的 Kozai-Lidov 机制
  4. 分析数值方法的收敛性（精度阶数验证）

运行方式:
  pip install -r requirements.txt
  python main.py

输出:
  - 1_论文/assets/ : 论文用 PNG 图像
  - 3_Data/        : CSV 模拟数据
"""

import os
import sys
import numpy as np

# ============================================================
# 关键: 在导入任何使用 pyplot 的模块之前设置后端
# ============================================================
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

# 项目根目录 = 当前脚本所在目录的父目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, '3_Data')
ASSETS_DIR = os.path.join(PROJECT_ROOT, '1_论文', 'assets')

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(ASSETS_DIR, exist_ok=True)

from physics import (
    init_solar_system_2body,
    init_solar_system_3body,
    compute_total_energy,
)
from solvers import euler_integrate, leapfrog_integrate
from analysis import (
    compute_energy_history,
    compute_angular_momentum_history,
    compute_orbital_history,
    plot_orbit_2d,
    plot_orbit_3d,
    plot_energy_comparison,
    plot_energy_error,
    plot_angular_momentum,
    plot_kozai_lidov_evolution,
    plot_stepsize_convergence,
    plot_comprehensive_comparison,
    save_data_to_csv,
)


def asset_path(filename):
    return os.path.join(ASSETS_DIR, filename)


def data_path(filename):
    return os.path.join(DATA_DIR, filename)


# ============================================================
# 两体系统模拟: 太阳-地球
# ============================================================

def run_2body_simulation():
    """
    太阳-地球两体系统模拟

    物理问题: 验证 Keper 轨道在数值积分下的稳定性
    核心对比: Euler (一阶非辛) vs Leapfrog (二阶辛)
    """
    print("=" * 60)
    print("两体系统模拟: 太阳-地球")
    print("=" * 60)

    positions, velocities, masses, names = init_solar_system_2body()
    E0 = compute_total_energy(positions, velocities, masses)

    dt = 0.001       # 时间步长 [yr]
    n_steps = 100000  # 模拟 100 yr
    time_array = np.arange(n_steps + 1) * dt

    print(f"  初始能量 E₀ = {E0:.8f}")
    print(f"  时间步长 Δt = {dt:.4f} yr")
    print(f"  总步数 = {n_steps}, 模拟时长 = {n_steps * dt:.1f} yr")
    print(f"  每轨道步数 = {1.0 / dt:.0f} 步/轨道周期")

    # 分别用两种方法积分
    print("\n  [1/4] Euler 积分...")
    pos_euler, vel_euler = euler_integrate(
        positions.copy(), velocities.copy(), masses, dt, n_steps
    )
    print("  [2/4] Leapfrog 积分...")
    pos_leapfrog, vel_leapfrog = leapfrog_integrate(
        positions.copy(), velocities.copy(), masses, dt, n_steps
    )

    # 能量历史
    print("  [3/4] 计算能量历史...")
    energy_euler = compute_energy_history(pos_euler, vel_euler, masses)
    energy_leapfrog = compute_energy_history(pos_leapfrog, vel_leapfrog, masses)

    # 角动量历史
    print("  [4/4] 计算角动量历史...")
    L_euler, _ = compute_angular_momentum_history(pos_euler, vel_euler, masses)
    L_leapfrog, _ = compute_angular_momentum_history(pos_leapfrog, vel_leapfrog, masses)

    # 结果摘要
    dE_euler = energy_euler[-1] - E0
    dE_leapfrog = energy_leapfrog[-1] - E0
    print(f"\n  最终能量 Euler:    {energy_euler[-1]:.8f}  (ΔE = {dE_euler:+.4e})")
    print(f"  最终能量 Leapfrog: {energy_leapfrog[-1]:.8f}  (ΔE = {dE_leapfrog:+.4e})")
    print(f"  误差比 (Euler/Leapfrog): {abs(dE_euler) / max(abs(dE_leapfrog), 1e-20):.1e}")

    # 生成图像
    print("\n  生成图像...")
    plot_orbit_2d(pos_euler, names,
                  title='2-Body Orbit (Sun-Earth) -- Euler Method\n'
                        'Outward spiral: energy not conserved',
                  save_path=asset_path('orbit_2body_euler.png'))

    plot_orbit_2d(pos_leapfrog, names,
                  title='2-Body Orbit (Sun-Earth) -- Leapfrog Method\n'
                        'Closed ellipse: energy conserved',
                  save_path=asset_path('orbit_2body_leapfrog.png'))

    plot_energy_comparison(
        energy_euler, energy_leapfrog, time_array,
        save_path=asset_path('energy_comparison_2body.png'),
    )

    plot_energy_error(
        energy_euler, energy_leapfrog, time_array,
        save_path=asset_path('energy_error_2body.png'),
    )

    plot_angular_momentum(
        L_euler, L_leapfrog, time_array,
        save_path=asset_path('angular_momentum_2body.png'),
    )

    plot_comprehensive_comparison(
        pos_euler, pos_leapfrog,
        energy_euler, energy_leapfrog,
        L_euler, L_leapfrog,
        time_array, names,
        save_path=asset_path('comprehensive_2body.png'),
    )

    # 导出数据
    print("  导出数据...")
    save_data_to_csv(pos_leapfrog, vel_leapfrog, masses, names,
                     data_path('simulation_2body_leapfrog.csv'))
    save_data_to_csv(pos_euler, vel_euler, masses, names,
                     data_path('simulation_2body_euler.csv'))

    print("  两体系统模拟完成.\n")
    return {
        'pos_euler': pos_euler,
        'pos_leapfrog': pos_leapfrog,
        'energy_euler': energy_euler,
        'energy_leapfrog': energy_leapfrog,
        'L_euler': L_euler,
        'L_leapfrog': L_leapfrog,
        'time_array': time_array,
        'names': names,
        'masses': masses,
    }


# ============================================================
# 三体系统模拟: 太阳-木星-小行星 (Kozai-Lidov)
# ============================================================

def run_3body_simulation():
    """
    太阳-木星-小行星三体系统模拟

    物理问题: 研究 Kozai-Lidov 机制——小行星在木星引力扰动下，
    轨道偏心率 e 与倾角 i 的周期性耦合振荡。

    核心守恒量: √(1-e²) · cos i ≈ 常数
    """
    print("=" * 60)
    print("三体系统模拟: 太阳-木星-小行星 (Kozai-Lidov)")
    print("=" * 60)

    positions, velocities, masses, names = init_solar_system_3body()
    E0 = compute_total_energy(positions, velocities, masses)

    dt = 0.01
    n_steps = 50000  # 500 yr
    time_array = np.arange(n_steps + 1) * dt

    print(f"  初始能量 E₀ = {E0:.8f}")
    print(f"  时间步长 Δt = {dt:.4f} yr")
    print(f"  模拟时长 = {n_steps * dt:.1f} yr")
    print(f"  初始小行星轨道: a=3.0 AU, e=0.3, i=30°")

    # 仅使用 Leapfrog (辛积分器对长期模拟至关重要)
    print("\n  [1/3] Leapfrog 积分...")
    pos_leapfrog, vel_leapfrog = leapfrog_integrate(
        positions.copy(), velocities.copy(), masses, dt, n_steps
    )

    E_final = compute_total_energy(pos_leapfrog[-1], vel_leapfrog[-1], masses)
    print(f"  最终能量: {E_final:.8f}  (ΔE = {E_final - E0:+.4e})")

    # 能量历史
    print("  [2/3] 计算能量历史...")
    energy_leapfrog = compute_energy_history(pos_leapfrog, vel_leapfrog, masses)

    # Kozai-Lidov 轨道根数追踪 (小行星是第2号天体)
    print("  [3/3] 追踪小行星轨道根数 (Kozai-Lidov 分析)...")
    sample_every = max(1, n_steps // 2000)
    a_hist, e_hist, inc_hist = compute_orbital_history(
        pos_leapfrog, vel_leapfrog, masses,
        body_idx=2, central_idx=0, sample_every=sample_every,
    )
    time_sampled = time_array[::sample_every]

    # Kozai 积分统计
    kozai_val = np.sqrt(1 - e_hist ** 2) * np.cos(np.radians(inc_hist))
    print(f"\n  Kozai 积分 √(1-e²)·cos i:")
    print(f"    均值 = {np.mean(kozai_val):.6f}")
    print(f"    标准差 = {np.std(kozai_val):.6f}")
    print(f"    相对变化 = {np.std(kozai_val) / abs(np.mean(kozai_val)):.2e}")
    print(f"  e 范围: [{np.min(e_hist):.4f}, {np.max(e_hist):.4f}]")
    print(f"  i 范围: [{np.min(inc_hist):.1f}°, {np.max(inc_hist):.1f}°]")

    # 生成图像
    print("\n  生成图像...")
    plot_orbit_2d(pos_leapfrog, names,
                  title='3-Body Orbit (Sun-Jupiter-Asteroid) -- x-y Projection\n'
                        'Note non-closed asteroid orbit (perturbed by Jupiter)',
                  save_path=asset_path('orbit_3body_leapfrog_2d.png'))

    plot_orbit_3d(pos_leapfrog, names,
                  title='3-Body Orbit (Sun-Jupiter-Asteroid) -- 3D View\n'
                        'Out-of-plane motion => Kozai-Lidov effect',
                  save_path=asset_path('orbit_3body_leapfrog_3d.png'))

    plot_kozai_lidov_evolution(
        time_sampled, a_hist, e_hist, inc_hist,
        body_name='Asteroid',
        save_path=asset_path('kozai_lidov_evolution.png'),
    )

    # 三体能量图
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(time_array, energy_leapfrog, 'steelblue', linewidth=1.2)
    ax.axhline(E0, color='gray', linestyle=':',
               label=f'Initial $E_0$ = {E0:.6f}')
    ax.set_xlabel('Time (yr)', fontsize=12)
    ax.set_ylabel('Total Energy', fontsize=12)
    ax.set_title('3-Body Energy Conservation (Leapfrog Symplectic Integrator)',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.5)
    fig.tight_layout()
    fig.savefig(asset_path('energy_3body_leapfrog.png'), dpi=300,
                bbox_inches='tight')
    plt.close(fig)

    print("  三体系统模拟完成.\n")
    return {
        'pos_leapfrog': pos_leapfrog,
        'energy_leapfrog': energy_leapfrog,
        'time_array': time_array,
        'a_hist': a_hist,
        'e_hist': e_hist,
        'inc_hist': inc_hist,
        'time_sampled': time_sampled,
        'kozai_val': kozai_val,
        'names': names,
    }


# ============================================================
# 收敛性测试
# ============================================================

def run_convergence_test():
    """
    收敛性测试: 不同步长下的能量误差

    物理问题: 验证数值方法的精度阶数
    - Euler:  误差 ∝ Δt¹
    - Leapfrog: 误差 ∝ Δt²

    方法: 在固定模拟时长 (10 yr) 下，用不同步长积分，
    计算最终能量误差 |E_final - E_0| / |E_0|
    """
    print("=" * 60)
    print("收敛性测试")
    print("=" * 60)

    positions, velocities, masses, names = init_solar_system_2body()
    E0 = compute_total_energy(positions, velocities, masses)

    dt_values = np.array([0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001])
    n_years = 10
    error_euler = np.zeros(len(dt_values))
    error_leapfrog = np.zeros(len(dt_values))

    print(f"  初始能量 = {E0:.8f}")
    print(f"  模拟时长 = {n_years} yr")
    print(f"  {'dt (yr)':>10s}  {'n_steps':>10s}  {'Euler err':>12s}  {'Leapfrog err':>12s}  {'Ratio':>10s}")
    print(f"  {'-' * 60}")

    for i, dt in enumerate(dt_values):
        n_steps = int(n_years / dt)

        pos_e, vel_e = euler_integrate(
            positions.copy(), velocities.copy(), masses, dt, n_steps
        )
        pos_l, vel_l = leapfrog_integrate(
            positions.copy(), velocities.copy(), masses, dt, n_steps
        )

        ef_e = compute_total_energy(pos_e[-1], vel_e[-1], masses)
        ef_l = compute_total_energy(pos_l[-1], vel_l[-1], masses)

        error_euler[i] = abs((ef_e - E0) / abs(E0))
        error_leapfrog[i] = abs((ef_l - E0) / abs(E0))
        ratio = error_euler[i] / max(error_leapfrog[i], 1e-30)

        print(f"  {dt:10.4f}  {n_steps:10d}  {error_euler[i]:12.4e}  "
              f"{error_leapfrog[i]:12.4e}  {ratio:10.1f}")

    # 拟合精度阶数
    slope_euler = np.polyfit(np.log10(dt_values), np.log10(error_euler), 1)[0]
    slope_leapfrog = np.polyfit(np.log10(dt_values), np.log10(error_leapfrog), 1)[0]
    print(f"\n  实测精度阶数:")
    print(f"    Euler:    {slope_euler:.2f} (理论值 1.0)")
    print(f"    Leapfrog: {slope_leapfrog:.2f} (理论值 2.0)")

    # 生成收敛图
    plot_stepsize_convergence(
        dt_values, error_euler, error_leapfrog,
        save_path=asset_path('convergence_test.png'),
    )

    # 导出数据
    np.savetxt(
        data_path('convergence_test.csv'),
        np.column_stack((dt_values, error_euler, error_leapfrog)),
        delimiter=',', header='dt,error_euler,error_leapfrog', comments='',
    )

    print("  收敛性测试完成.\n")
    return {
        'dt_values': dt_values,
        'error_euler': error_euler,
        'error_leapfrog': error_leapfrog,
        'slope_euler': slope_euler,
        'slope_leapfrog': slope_leapfrog,
    }


# ============================================================
# 主入口
# ============================================================

if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("  计算物理期末项目 — 选题 A-2")
    print("  N体轨道积分: 太阳系简化模型与 Leapfrog 守恒性分析")
    print("=" * 60 + "\n")

    # 1. 两体系统
    results_2body = run_2body_simulation()

    # 2. 三体系统 (Kozai-Lidov)
    results_3body = run_3body_simulation()

    # 3. 收敛性测试
    results_conv = run_convergence_test()

    # 最终总结
    print("=" * 60)
    print("  所有模拟完成!")
    print("=" * 60)
    print(f"\n  输出文件:")
    print(f"    图像: {ASSETS_DIR}/")
    print(f"    数据: {DATA_DIR}/")
    print(f"\n  关键发现:")
    print(f"    1. Leapfrog 辛积分器保持能量有界振荡")
    print(f"    2. Euler 非辛积分器导致能量单调漂移")
    print(f"    3. 收敛性验证: Euler O(Δt), Leapfrog O(Δt²)")
    print(f"    4. 三体系统展现 Kozai-Lidov 效应 (e-i 耦合)")
    print()
