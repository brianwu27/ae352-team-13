# simulate_hover.py
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

from dynamics import QuadParams


def hover_reference(t: float) -> np.ndarray:
    """
    Desired hover state:
    x_ref = [px, py, pz, vx, vy, vz, φ, θ, ψ, p, q, r]^T
    Hover at (0, 0, 1), level, zero rates.
    """
    x_ref = np.zeros(12)
    x_ref[2] = 1.0   # pz = 1 m
    return x_ref


def make_closed_loop_rhs(A, B, K, params):
    """
    Build f(t, x) = x_dot for the linear closed-loop system:
        Δx = x - x_ref
        Δu = -K Δx
        ẋ = A Δx + B Δu
    For hover, x_ref is constant in time, so this is valid.
    
    Also track inputs for analysis.
    """
    input_history = []
    
    def f(t, x):
        x_ref = hover_reference(t)
        dx = x - x_ref          # state error
        du = -K @ dx            # state feedback
        
        # Store input for later analysis
        # u = [ΔT_total, τφ, τθ, τψ]
        u_hover = np.array([params.m * params.g, 0, 0, 0])  # hover equilibrium
        u = u_hover + du
        input_history.append((t, u.copy()))
        
        xdot = A @ dx + B @ du  # linear error dynamics
        return xdot
    
    f.input_history = input_history
    return f


def main():
    params = QuadParams()
    A, B = params.linear_matrices()
    K = params.lqr_gain()

    # Initial state:
    # start at z = 0.0 (on ground) with small attitude error
    x0 = np.zeros(12)
    x0[2] = 0.0        # pz = 0 m
    x0[6] = np.deg2rad(1.0)   
    x0[7] = np.deg2rad(-1.0)  

    t_span = (0.0, 15.0)
    t_eval = np.linspace(t_span[0], t_span[1], 1501)

    f_cl = make_closed_loop_rhs(A, B, K, params)

    sol = solve_ivp(
        fun=f_cl,
        t_span=t_span,
        y0=x0,
        t_eval=t_eval,
        rtol=1e-6,
        atol=1e-9,
    )

    t = sol.t
    x = sol.y

    px   = x[0, :]
    py   = x[1, :]
    pz   = x[2, :]
    vx   = x[3, :]
    vy   = x[4, :]
    vz   = x[5, :]
    phi  = x[6, :]
    theta = x[7, :]
    psi  = x[8, :]
    p    = x[9, :]
    q    = x[10, :]
    r    = x[11, :]

    # Extract input history
    t_input = np.array([item[0] for item in f_cl.input_history])
    u_history = np.array([item[1] for item in f_cl.input_history])
    
    T_total = u_history[:, 0]
    tau_phi = u_history[:, 1]
    tau_theta = u_history[:, 2]
    tau_psi = u_history[:, 3]
    
    # Interpolate states onto the input timestamps so shapes match
    vz_i = np.interp(t_input, t, vz)
    p_i  = np.interp(t_input, t, p)
    q_i  = np.interp(t_input, t, q)
    r_i  = np.interp(t_input, t, r)

    # Power model (same idea as yours, just vectorized and shape-consistent)
    power = np.abs(T_total) * np.abs(vz_i) \
        + np.abs(tau_phi)   * np.abs(p_i) \
        + np.abs(tau_theta) * np.abs(q_i) \
        + np.abs(tau_psi)   * np.abs(r_i)

    energy = np.trapz(power, t_input)  # Total energy consumption

        # ------------------------------------------------------
    # Plotting - 3-figure grouped layout
    # ------------------------------------------------------

    # Common derived quantities
    radius = np.sqrt(px**2 + py**2)
    speed  = np.sqrt(vx**2 + vy**2 + vz**2)

    # Cumulative energy vs time
    energy_cumulative = np.cumsum(power[:-1] * np.diff(t_input))
    energy_cumulative = np.insert(energy_cumulative, 0, 0)

    # -----------------------------
    # FIGURE 1: 3D trajectory (own figure)
    # -----------------------------
    fig1 = plt.figure(figsize=(8, 6))
    ax = fig1.add_subplot(111, projection='3d')
    ax.plot(px, py, pz, label='Trajectory', linewidth=2)
    ax.scatter([0], [0], [1.0], c='r', s=100, label='Hover target')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_zlabel('z [m]')
    ax.set_title('3D Position (Hover)')
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(-0.5, 0.5)
    ax.set_zlim(0, 1.2)
    ax.grid(True)
    ax.legend()
    fig1.tight_layout()

    # -----------------------------
    # FIGURE 2: 12 DOFs grouped (2x2) in one figure
    # -----------------------------
    fig2, axs = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
    axs = axs.ravel()

    # Position
    axs[0].plot(t, px, linewidth=2, label='px')
    axs[0].plot(t, py, linewidth=2, label='py')
    axs[0].plot(t, pz, linewidth=2, label='pz')
    axs[0].axhline(1.0, linestyle='--', alpha=0.5, label='Target z')
    axs[0].set_ylabel('Position [m]')
    axs[0].set_title('Position')
    axs[0].grid(True)
    axs[0].legend()

    # Velocity
    axs[1].plot(t, vx, linewidth=2, label='vx')
    axs[1].plot(t, vy, linewidth=2, label='vy')
    axs[1].plot(t, vz, linewidth=2, label='vz')
    axs[1].axhline(0.0, linestyle='--', alpha=0.3)
    axs[1].set_ylabel('Velocity [m/s]')
    axs[1].set_title('Velocity')
    axs[1].grid(True)
    axs[1].legend()

    # Euler angles
    axs[2].plot(t, np.rad2deg(phi), linewidth=2, label='φ (roll)')
    axs[2].plot(t, np.rad2deg(theta), linewidth=2, label='θ (pitch)')
    axs[2].plot(t, np.rad2deg(psi), linewidth=2, label='ψ (yaw)')
    axs[2].axhline(0.0, linestyle='--', alpha=0.3)
    axs[2].set_xlabel('Time [s]')
    axs[2].set_ylabel('Angle [deg]')
    axs[2].set_title('Euler Angles')
    axs[2].grid(True)
    axs[2].legend()

    # Angular rates
    axs[3].plot(t, np.rad2deg(p), linewidth=2, label='p (roll rate)')
    axs[3].plot(t, np.rad2deg(q), linewidth=2, label='q (pitch rate)')
    axs[3].plot(t, np.rad2deg(r), linewidth=2, label='r (yaw rate)')
    axs[3].axhline(0.0, linestyle='--', alpha=0.3)
    axs[3].set_xlabel('Time [s]')
    axs[3].set_ylabel('Rate [deg/s]')
    axs[3].set_title('Angular Rates')
    axs[3].grid(True)
    axs[3].legend()

    fig2.suptitle('State Time Histories (12 DOF)', fontsize=14)
    fig2.tight_layout()
    fig2.subplots_adjust(top=0.92)

        # -----------------------------
    # FIGURE 3: Inputs + Motor Thrusts + Power + Energy (2x2)
    # -----------------------------
    fig3, axs3 = plt.subplots(2, 2, figsize=(14, 9))
    axs3 = axs3.ravel()

    # (0) Inputs u(t): thrust + torques (twin y-axis so torques are visible)
    ax_u = axs3[0]
    ax_u.plot(t_input, T_total, linewidth=2, label='F (total thrust)')
    ax_u.set_xlabel('Time [s]')
    ax_u.set_ylabel('Thrust [N]')
    ax_u.set_title('Inputs u(t)')
    ax_u.grid(True)

    ax_u2 = ax_u.twinx()
    ax_u2.plot(t_input, tau_phi,   linewidth=2, linestyle='--', label='τφ')
    ax_u2.plot(t_input, tau_theta, linewidth=2, linestyle='--', label='τθ')
    ax_u2.plot(t_input, tau_psi,   linewidth=2, linestyle='--', label='τψ')
    ax_u2.set_ylabel('Torque [N·m]')

    # Combined legend
    lines1, labels1 = ax_u.get_legend_handles_labels()
    lines2, labels2 = ax_u2.get_legend_handles_labels()
    ax_u2.legend(lines1 + lines2, labels1 + labels2, loc='best')

    # (1) Motor thrusts (hover: equal split)
    T1 = T_total / 4.0
    T2 = T_total / 4.0
    T3 = T_total / 4.0
    T4 = T_total / 4.0

    axs3[1].plot(t_input, T1, linewidth=2, label='T1')
    axs3[1].plot(t_input, T2, linewidth=2, label='T2')
    axs3[1].plot(t_input, T3, linewidth=2, label='T3')
    axs3[1].plot(t_input, T4, linewidth=2, label='T4')
    axs3[1].set_xlabel('Time [s]')
    axs3[1].set_ylabel('Thrust [N]')
    axs3[1].set_title('Motor Thrusts')
    axs3[1].grid(True)
    axs3[1].legend()

    # (2) Power
    axs3[2].plot(t_input, power, linewidth=2)
    axs3[2].set_xlabel('Time [s]')
    axs3[2].set_ylabel('Power [W] (proxy)')
    axs3[2].set_title('Power')
    axs3[2].grid(True)

    # (3) Energy
    axs3[3].plot(t_input, energy_cumulative, linewidth=2)
    axs3[3].set_xlabel('Time [s]')
    axs3[3].set_ylabel('Energy [J] (proxy)')
    axs3[3].set_title('Cumulative Energy')
    axs3[3].grid(True)

    fig3.suptitle('Control Effort and Energy', fontsize=14)
    fig3.tight_layout()
    fig3.subplots_adjust(top=0.92)


    plt.show()


    # Performance statistics
    print("\n" + "="*60)
    print("LINEAR HOVER PERFORMANCE ANALYSIS")
    print("="*60)
    print("\nPOSITION TRACKING:")
    print(f"  Final altitude: {pz[-1]:.6f} m (target: 1.0 m)")
    print(f"  Altitude error: {abs(pz[-1] - 1.0)*1000:.3f} mm")
    print(f"  Mean altitude (last 5s): {np.mean(pz[t > 10]):.6f} m")
    print(f"  Altitude std dev: {np.std(pz[t > 10])*1000:.4f} mm")
    print(f"  Max XY drift: {np.max(radius)*1000:.4f} mm")
    print(f"  Final XY position: ({px[-1]*1000:.3f}, {py[-1]*1000:.3f}) mm")
    
    print("\nATTITUDE:")
    print(f"  Max |φ|: {np.max(np.abs(phi))*180/np.pi:.4f} deg")
    print(f"  Max |θ|: {np.max(np.abs(theta))*180/np.pi:.4f} deg")
    print(f"  Max |ψ|: {np.max(np.abs(psi))*180/np.pi:.4f} deg")
    print(f"  Final attitude: φ={np.rad2deg(phi[-1]):.4f}°, "
          f"θ={np.rad2deg(theta[-1]):.4f}°, ψ={np.rad2deg(psi[-1]):.4f}°")
    
    print("\nCONTROL EFFORT:")
    print(f"  Mean thrust: {np.mean(T_total):.4f} N")
    print(f"  Thrust std dev: {np.std(T_total):.6f} N")
    print(f"  Max |τφ|: {np.max(np.abs(tau_phi))*1000:.4f} mN·m")
    print(f"  Max |τθ|: {np.max(np.abs(tau_theta))*1000:.4f} mN·m")
    print(f"  Max |τψ|: {np.max(np.abs(tau_psi))*1000:.4f} mN·m")
    
    print("\nENERGY:")
    print(f"  Total energy consumed: {energy:.2f} J")
    print(f"  Average power: {np.mean(power):.2f} W")
    print(f"  Peak power: {np.max(power):.2f} W")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    main()