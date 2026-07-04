"""
Analysis & Visualization Module

Provides energy history, orbit plotting, energy conservation analysis,
convergence testing, Kozai-Lidov mechanism tracking, and data export.

All plot functions support save_path=None (interactive) or file export.
All plot labels use English for proper font rendering (DejaVu Sans).
Chinese physical explanations of each figure are provided in 思路/.
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rcParams
import warnings

warnings.filterwarnings('ignore', message='Glyph.*missing from font')
warnings.filterwarnings('ignore', message='Font.*does not have a glyph')

from physics import (
    compute_total_energy,
    compute_angular_momentum,
    compute_orbital_elements,
)

# Font: DejaVu Sans (full Latin/math glyph coverage)
rcParams['font.sans-serif'] = ['DejaVu Sans']
rcParams['font.family'] = 'sans-serif'
rcParams['mathtext.fontset'] = 'dejavusans'
rcParams['axes.unicode_minus'] = False


# ============================================================
# Energy & Angular Momentum History
# ============================================================

def compute_energy_history(pos_history, vel_history, masses):
    """Compute total energy at each time step."""
    n_steps = len(pos_history) - 1
    energy = np.zeros(n_steps + 1)
    for step in range(n_steps + 1):
        energy[step] = compute_total_energy(
            pos_history[step], vel_history[step], masses
        )
    return energy


def compute_angular_momentum_history(pos_history, vel_history, masses):
    """Compute total angular momentum magnitude at each time step."""
    n_steps = len(pos_history) - 1
    L_mag = np.zeros(n_steps + 1)
    L_z = np.zeros(n_steps + 1)
    for step in range(n_steps + 1):
        L = compute_angular_momentum(
            pos_history[step], vel_history[step], masses
        )
        L_mag[step] = np.linalg.norm(L)
        L_z[step] = L[2]
    return L_mag, L_z


def compute_orbital_history(pos_history, vel_history, masses,
                            body_idx, central_idx=0, sample_every=1):
    """Track osculating orbital elements of a body over time."""
    n_steps = len(pos_history) - 1
    n_samples = n_steps // sample_every + 1
    a_hist = np.zeros(n_samples)
    e_hist = np.zeros(n_samples)
    inc_hist = np.zeros(n_samples)
    for k in range(n_samples):
        step = k * sample_every
        elem = compute_orbital_elements(
            pos_history[step], vel_history[step], masses,
            body_idx, central_idx
        )
        a_hist[k] = elem['semi_major_axis']
        e_hist[k] = elem['eccentricity']
        inc_hist[k] = elem['inclination_deg']
    return a_hist, e_hist, inc_hist


# ============================================================
# Orbit Plots
# ============================================================

def plot_orbit_2d(pos_history, names, title='N-body Orbit (x-y)',
                  save_path=None, show_final=True):
    """
    2D orbit plot (x-y projection).

    Physical meaning: Shows the trajectory in the orbital plane.
    Closed ellipse -> stable Kepler motion.
    Outward spiral -> energy drift (non-symplectic integrator).
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    colors = ['gold', 'dodgerblue', 'crimson', 'seagreen']
    for i, name in enumerate(names):
        color = colors[i % len(colors)]
        ax.plot(pos_history[:, i, 0], pos_history[:, i, 1],
                label=name, linewidth=1.2, color=color)
        if show_final:
            ax.scatter(pos_history[-1, i, 0], pos_history[-1, i, 1],
                       s=60, color=color, zorder=5, marker='o')
        ax.scatter(pos_history[0, i, 0], pos_history[0, i, 1],
                   s=30, marker='s', color=color, zorder=5)

    ax.set_xlabel('x (AU)', fontsize=12)
    ax.set_ylabel('y (AU)', fontsize=12)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_aspect('equal')

    textstr = 'square = start    circle = end'
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes,
            fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
    return fig


def plot_orbit_3d(pos_history, names, title='N-body Orbit (3D)',
                  save_path=None):
    """
    3D orbit plot.

    Physical meaning: Reveals the full spatial structure.
    For 3-body systems, out-of-plane motion indicates Kozai-Lidov effect.
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    colors = ['gold', 'dodgerblue', 'crimson', 'seagreen']
    for i, name in enumerate(names):
        color = colors[i % len(colors)]
        ax.plot(pos_history[:, i, 0], pos_history[:, i, 1],
                pos_history[:, i, 2], label=name,
                linewidth=1.2, color=color)
        ax.scatter(pos_history[-1, i, 0], pos_history[-1, i, 1],
                   pos_history[-1, i, 2], s=50, color=color)

    ax.set_xlabel('x (AU)', fontsize=11)
    ax.set_ylabel('y (AU)', fontsize=11)
    ax.set_zlabel('z (AU)', fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
    return fig


# ============================================================
# Energy Conservation Analysis
# ============================================================

def plot_energy_comparison(energy_euler, energy_leapfrog, time_array,
                           save_path=None):
    """
    Energy vs time comparison.

    Physical insight:
      - Leapfrog (symplectic): No long-term drift, only bounded oscillation
      - Euler (non-symplectic): Energy grows monotonically -> violates conservation
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(time_array, energy_euler, 'coral', linewidth=1.5,
            label='Euler (non-symplectic, order 1)')
    ax.plot(time_array, energy_leapfrog, 'steelblue', linewidth=1.5,
            label='Leapfrog (symplectic, order 2)')
    ax.axhline(energy_euler[0], color='gray', linestyle=':',
               linewidth=1, label=f'Initial energy $E_0$')

    ax.set_xlabel('Time (yr)', fontsize=12)
    ax.set_ylabel('Total Energy $E$', fontsize=12)
    ax.set_title('Energy Conservation: Leapfrog vs Euler',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.5)

    drift = energy_euler[-1] - energy_euler[0]
    textstr = (f'Euler energy drift: {drift:.2e}\n'
               f'Leapfrog: bounded oscillation (conserved)')
    props = dict(boxstyle='round', facecolor='lightyellow', alpha=0.8)
    ax.text(0.98, 0.05, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='bottom', horizontalalignment='right',
            bbox=props)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
    return fig


def plot_energy_error(energy_euler, energy_leapfrog, time_array,
                      save_path=None):
    """
    Relative energy error comparison (semi-log).

    Physical insight:
      - Euler: |dE/E0| grows linearly with time
      - Leapfrog: error stays bounded at ~1e-10 to 1e-12 level
    """
    E0 = energy_euler[0]
    err_euler = np.abs((energy_euler - E0) / np.abs(E0))
    err_leapfrog = np.abs((energy_leapfrog - E0) / np.abs(E0))

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.semilogy(time_array, err_euler, 'coral', linewidth=1.5,
                label='Euler (non-symplectic)')
    ax.semilogy(time_array, err_leapfrog, 'steelblue', linewidth=1.5,
                label='Leapfrog (symplectic)')

    mid_t = time_array[len(time_array) // 2]
    ax.annotate(f'Euler final error: {err_euler[-1]:.2e}',
                xy=(time_array[-1], err_euler[-1]),
                xytext=(mid_t, err_euler[-1] * 10),
                arrowprops=dict(arrowstyle='->', color='coral'),
                fontsize=9, color='darkred')
    ax.annotate(f'Leapfrog error: ~{np.mean(err_leapfrog):.2e}',
                xy=(mid_t, np.median(err_leapfrog)),
                xytext=(mid_t * 0.3, np.median(err_leapfrog) * 0.1),
                arrowprops=dict(arrowstyle='->', color='steelblue'),
                fontsize=9, color='darkblue')

    ax.set_xlabel('Time (yr)', fontsize=12)
    ax.set_ylabel('Relative Energy Error $|\\Delta E / E_0|$', fontsize=12)
    ax.set_title('Energy Error Comparison (Semi-log)',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.5)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
    return fig


# ============================================================
# Angular Momentum Analysis
# ============================================================

def plot_angular_momentum(L_euler, L_leapfrog, time_array, save_path=None):
    """
    Angular momentum conservation comparison.

    Physical insight: In a central force field, angular momentum is
    strictly conserved. Numerical errors can be compared to energy errors.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    L0 = L_euler[0]
    err_euler = np.abs((L_euler - L0) / np.abs(L0))
    err_leapfrog = np.abs((L_leapfrog - L0) / np.abs(L0))

    ax1.plot(time_array, L_euler, 'coral', linewidth=1.5, label='Euler')
    ax1.plot(time_array, L_leapfrog, 'steelblue', linewidth=1.5,
             label='Leapfrog')
    ax1.axhline(L0, color='gray', linestyle=':', label='Initial $|L|$')
    ax1.set_xlabel('Time (yr)')
    ax1.set_ylabel('$|\\mathbf{L}|$')
    ax1.set_title('Angular Momentum Magnitude')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.5)

    ax2.semilogy(time_array, err_euler, 'coral', linewidth=1.5,
                 label='Euler')
    ax2.semilogy(time_array, err_leapfrog, 'steelblue', linewidth=1.5,
                 label='Leapfrog')
    ax2.set_xlabel('Time (yr)')
    ax2.set_ylabel('$|\\Delta L / L_0|$')
    ax2.set_title('Angular Momentum Relative Error')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.5)

    fig.suptitle('Angular Momentum Conservation Analysis',
                 fontsize=14, fontweight='bold')
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
    return fig


# ============================================================
# Kozai-Lidov Mechanism Tracking
# ============================================================

def plot_kozai_lidov_evolution(time_sampled, a_hist, e_hist, inc_hist,
                               body_name='Asteroid', save_path=None):
    """
    Kozai-Lidov mechanism evolution plot.

    Physical insight:
      In a hierarchical triple system, the outer perturber drives coupled
      oscillations between the inner body's eccentricity (e) and inclination (i).
      Conserved quantity: sqrt(1-e^2) * cos(i) ~ constant (Kozai integral).
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Semi-major axis
    ax = axes[0, 0]
    ax.plot(time_sampled, a_hist, 'steelblue', linewidth=1.2)
    ax.set_xlabel('Time (yr)')
    ax.set_ylabel('Semi-major axis $a$ (AU)')
    ax.set_title(f'{body_name} Semi-major Axis Evolution')
    ax.grid(True, linestyle='--', alpha=0.5)
    if len(a_hist) > 1:
        ax.axhline(np.median(a_hist), color='gray', linestyle=':',
                   label=f'Median = {np.median(a_hist):.2f} AU')
        ax.legend()

    # Eccentricity
    ax = axes[0, 1]
    ax.plot(time_sampled, e_hist, 'crimson', linewidth=1.2)
    ax.set_xlabel('Time (yr)')
    ax.set_ylabel('Eccentricity $e$')
    ax.set_title(f'{body_name} Eccentricity (Kozai-Lidov oscillation)')
    ax.grid(True, linestyle='--', alpha=0.5)

    # Inclination
    ax = axes[1, 0]
    ax.plot(time_sampled, inc_hist, 'seagreen', linewidth=1.2)
    ax.set_xlabel('Time (yr)')
    ax.set_ylabel('Inclination $i$ (deg)')
    ax.set_title(f'{body_name} Inclination Evolution')
    ax.grid(True, linestyle='--', alpha=0.5)

    # Kozai integral
    ax = axes[1, 1]
    kozai_const = np.sqrt(1 - e_hist ** 2) * np.cos(np.radians(inc_hist))
    ax.plot(time_sampled, kozai_const, 'darkorange', linewidth=1.2)
    ax.set_xlabel('Time (yr)')
    ax.set_ylabel('$\\sqrt{1-e^2} \\cos i$')
    ax.set_title('Kozai Integral (theoretically conserved)')
    ax.grid(True, linestyle='--', alpha=0.5)
    if len(kozai_const) > 1:
        mean_val = np.mean(kozai_const)
        ax.axhline(mean_val, color='gray', linestyle=':',
                   label=f'Mean = {mean_val:.4f}')
        ax.legend()

    fig.suptitle('Kozai-Lidov Mechanism: e-i Coupled Oscillation',
                 fontsize=15, fontweight='bold')
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
    return fig


# ============================================================
# Convergence Analysis
# ============================================================

def plot_stepsize_convergence(dt_values, error_euler, error_leapfrog,
                              save_path=None):
    """
    Convergence test (log-log) with theoretical slope reference lines.

    Physical insight:
      - Euler error ~ dt^1  (1st-order accuracy)
      - Leapfrog error ~ dt^2 (2nd-order accuracy)
      The slope in log-log space reveals the order of accuracy.
    """
    fig, ax = plt.subplots(figsize=(10, 7))

    ax.loglog(dt_values, error_euler, 'o-', color='coral', linewidth=2,
              markersize=10, label='Euler (1st-order)')
    ax.loglog(dt_values, error_leapfrog, 's-', color='steelblue',
              linewidth=2, markersize=10, label='Leapfrog (2nd-order)')

    # Theoretical slope reference lines
    dt_ref = np.array([dt_values[0], dt_values[-1]])
    ref_euler = error_euler[0] * (dt_ref / dt_values[0]) ** 1
    ref_leapfrog = error_leapfrog[0] * (dt_ref / dt_values[0]) ** 2

    ax.loglog(dt_ref, ref_euler, '--', color='darkred', linewidth=1,
              alpha=0.7, label='$\\propto \\Delta t$ (theory: order 1)')
    ax.loglog(dt_ref, ref_leapfrog, '--', color='darkblue', linewidth=1,
              alpha=0.7, label='$\\propto \\Delta t^2$ (theory: order 2)')

    ax.set_xlabel('Time Step $\\Delta t$ (yr)', fontsize=13)
    ax.set_ylabel('Relative Energy Error after 10 yr', fontsize=13)
    ax.set_title('Convergence & Order-of-Accuracy Verification',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.invert_xaxis()

    # Measured slopes
    if len(dt_values) >= 2:
        sl_e = np.polyfit(np.log10(dt_values), np.log10(error_euler), 1)[0]
        sl_l = np.polyfit(np.log10(dt_values), np.log10(error_leapfrog), 1)[0]
        textstr = (f'Measured slopes:\n'
                   f'  Euler: {sl_e:.2f} (theory: 1.0)\n'
                   f'  Leapfrog: {sl_l:.2f} (theory: 2.0)')
        props = dict(boxstyle='round', facecolor='lightyellow', alpha=0.8)
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=11,
                verticalalignment='top', bbox=props)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
    return fig


# ============================================================
# Comprehensive Comparison Dashboard
# ============================================================

def plot_comprehensive_comparison(pos_euler, pos_leapfrog,
                                  energy_euler, energy_leapfrog,
                                  L_mag_euler, L_mag_leapfrog,
                                  time_array, names, save_path=None):
    """
    Comprehensive comparison dashboard: Orbit + Energy + Angular Momentum.

    6-panel figure suitable for direct inclusion in the paper.
    """
    fig = plt.figure(figsize=(16, 12))

    # (a) Euler orbit
    ax1 = fig.add_subplot(2, 3, 1)
    for i, name in enumerate(names):
        ax1.plot(pos_euler[:, i, 0], pos_euler[:, i, 1],
                 linewidth=1.2, label=name)
    ax1.set_title('(a) Euler Orbit (x-y)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('x (AU)')
    ax1.set_ylabel('y (AU)')
    ax1.set_aspect('equal')
    ax1.legend(fontsize=8)
    ax1.grid(True, linestyle='--', alpha=0.4)
    ax1.text(0.5, 0.95, '<- orbital expansion (energy drift)',
             transform=ax1.transAxes, fontsize=9, color='darkred', ha='center',
             bbox=dict(facecolor='white', alpha=0.7))

    # (b) Leapfrog orbit
    ax2 = fig.add_subplot(2, 3, 2)
    for i, name in enumerate(names):
        ax2.plot(pos_leapfrog[:, i, 0], pos_leapfrog[:, i, 1],
                 linewidth=1.2, label=name)
    ax2.set_title('(b) Leapfrog Orbit (x-y)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('x (AU)')
    ax2.set_ylabel('y (AU)')
    ax2.set_aspect('equal')
    ax2.legend(fontsize=8)
    ax2.grid(True, linestyle='--', alpha=0.4)
    ax2.text(0.5, 0.95, '<- closed ellipse (energy conserved)',
             transform=ax2.transAxes, fontsize=9, color='darkgreen', ha='center',
             bbox=dict(facecolor='white', alpha=0.7))

    # (c) Energy deviation
    ax3 = fig.add_subplot(2, 3, 3)
    E0 = energy_euler[0]
    ax3.plot(time_array, (energy_euler - E0) / abs(E0), 'coral',
             linewidth=1.2, label='Euler')
    ax3.plot(time_array, (energy_leapfrog - E0) / abs(E0), 'steelblue',
             linewidth=1.2, label='Leapfrog')
    ax3.set_title('(c) Relative Energy Deviation', fontsize=12,
                  fontweight='bold')
    ax3.set_xlabel('Time (yr)')
    ax3.set_ylabel('$(E - E_0) / |E_0|$')
    ax3.legend(fontsize=8)
    ax3.grid(True, linestyle='--', alpha=0.4)

    # (d) Energy error (semi-log)
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.semilogy(time_array, np.abs((energy_euler - E0) / abs(E0)),
                 'coral', linewidth=1.2, label='Euler')
    ax4.semilogy(time_array, np.abs((energy_leapfrog - E0) / abs(E0)),
                 'steelblue', linewidth=1.2, label='Leapfrog')
    ax4.set_title('(d) Energy Error (Semi-log)', fontsize=12,
                  fontweight='bold')
    ax4.set_xlabel('Time (yr)')
    ax4.set_ylabel('$|\\Delta E / E_0|$')
    ax4.legend(fontsize=8)
    ax4.grid(True, linestyle='--', alpha=0.4)

    # (e) Angular momentum error
    ax5 = fig.add_subplot(2, 3, 5)
    L0 = L_mag_euler[0]
    ax5.plot(time_array, (L_mag_euler - L0) / abs(L0), 'coral',
             linewidth=1.2, label='Euler')
    ax5.plot(time_array, (L_mag_leapfrog - L0) / abs(L0), 'steelblue',
             linewidth=1.2, label='Leapfrog')
    ax5.set_title('(e) Relative Angular Momentum Error', fontsize=12,
                  fontweight='bold')
    ax5.set_xlabel('Time (yr)')
    ax5.set_ylabel('$(L - L_0) / |L_0|$')
    ax5.legend(fontsize=8)
    ax5.grid(True, linestyle='--', alpha=0.4)

    # (f) Summary
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')
    err_ratio = abs(energy_euler[-1] - E0) / max(abs(energy_leapfrog[-1] - E0), 1e-20)
    summary = (
        'COMPREHENSIVE COMPARISON\n\n'
        f'Euler orbit: outward spiral -> energy not conserved\n'
        f'Leapfrog orbit: closed ellipse -> symplectic advantage\n'
        f'Euler final energy error: {abs((energy_euler[-1]-E0)/abs(E0)):.2e}\n'
        f'Leapfrog final energy error: {abs((energy_leapfrog[-1]-E0)/abs(E0)):.2e}\n'
        f'Error ratio (Euler/Leapfrog): {err_ratio:.1e}\n\n'
        f'Conclusion: Symplectic integrators are\n'
        f'essential for long-term celestial mechanics.'
    )
    ax6.text(0.1, 0.9, summary, transform=ax6.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8))

    fig.suptitle('N-body Orbit Integration: Euler vs Leapfrog (Symplectic)',
                 fontsize=15, fontweight='bold')
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
    return fig


# ============================================================
# Data Export
# ============================================================

def save_data_to_csv(pos_history, vel_history, masses, names, filename):
    """Export simulation data to CSV."""
    n_steps = len(pos_history) - 1
    n_bodies = len(names)

    header = 'time,'
    for name in names:
        header += f'{name}_x,{name}_y,{name}_z,{name}_vx,{name}_vy,{name}_vz,'
    header = header[:-1]

    data = np.zeros((n_steps + 1, 1 + n_bodies * 6))
    data[:, 0] = np.arange(n_steps + 1)
    for i in range(n_bodies):
        start = 1 + i * 6
        data[:, start:start + 3] = pos_history[:, i, :]
        data[:, start + 3:start + 6] = vel_history[:, i, :]

    np.savetxt(filename, data, delimiter=',', header=header, comments='')
    print(f"  Data saved to {filename}")
