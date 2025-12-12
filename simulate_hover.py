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

    t_span = (0.0, 130)
    t_eval = np.linspace(t_span[0], t_span[1], 13001)

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
    
        # -----------------------------
    # Motor thrusts T1..T4 (mixer inversion, same idea as circle)
    # -----------------------------
    try:
        L = params.L
        k_tau = params.k_tau

        # Standard "plus" configuration mixer:
        # [T_total]   [ 1   1   1   1 ] [T1]
        # [tau_phi] = [ 0  -L   0  +L ] [T2]
        # [tau_theta] [ -L  0  +L  0 ] [T3]
        # [tau_psi]   [ k -k   k  -k ] [T4]
        M = np.array([
            [1.0,   1.0,   1.0,   1.0],
            [0.0,  -L,     0.0,  +L  ],
            [-L,    0.0,  +L,     0.0],
            [k_tau, -k_tau, k_tau, -k_tau]
        ])

        U = np.vstack((T_total, tau_phi, tau_theta, tau_psi))  # (4, N)
        T_motors = np.linalg.solve(M, U)                       # (4, N)
    except Exception:
        # Fallback: equal split (still valid for symmetric hover)
        T_motors = np.vstack((T_total/4, T_total/4, T_total/4, T_total/4))

    T1, T2, T3, T4 = T_motors[0, :], T_motors[1, :], T_motors[2, :], T_motors[3, :]

    # -----------------------------
    # Rotor-based power + energy (includes baseline hover power)
    # Actuator disk / momentum theory proxy:
    #   P_i ≈ T_i^(3/2) / sqrt(2 ρ A)
    # Electrical: P_elec = P_mech / η
    # -----------------------------
    rho = 1.225  # kg/m^3
    R_rotor = getattr(params, "rotor_radius", 0.0635)  # m (default ~5" props)
    A_rotor = np.pi * R_rotor**2
    eta = getattr(params, "eta", 0.7)  # overall efficiency (motor+ESC+prop)

    def thrust_to_power(T):
        T = np.maximum(T, 0.0)  # avoid negative thrust -> NaNs
        return (T**1.5) / np.sqrt(2.0 * rho * A_rotor)

    P_mech1 = thrust_to_power(T1)
    P_mech2 = thrust_to_power(T2)
    P_mech3 = thrust_to_power(T3)
    P_mech4 = thrust_to_power(T4)
    P_mech = P_mech1 + P_mech2 + P_mech3 + P_mech4
    P_elec = P_mech / eta

    E_cumulative = np.cumsum(P_elec[:-1] * np.diff(t_input))
    E_cumulative = np.insert(E_cumulative, 0, 0.0)
    E_total = np.trapezoid(P_elec, t_input)
    print(E_total)
    
    k = params.k
    k_tau = params.k_tau
    omega1 = np.sqrt(T1/k)
    omega2 = np.sqrt(T2/k)
    omega3 = np.sqrt(T3/k)
    omega4 = np.sqrt(T4/k)
    print('omega1: ', omega1[10])
    
    rpm1 = omega1*60/(2*np.pi)
    rpm2 = omega2*60/(2*np.pi)
    rpm3 = omega3*60/(2*np.pi)
    rpm4 = omega4*60/(2*np.pi)
    
    torque_motor1 = P_mech1/omega1
    torque_motor2 = P_mech2/omega2
    torque_motor3 = P_mech3/omega3
    torque_motor4 = P_mech4/omega4
    print('motor torque: ', torque_motor1[10])
    
    current1 = torque_motor1/k_tau
    current2 = torque_motor2/k_tau
    current3 = torque_motor3/k_tau
    current4 = torque_motor4/k_tau
    current_total = current1 + current2 + current3 + current4
    print(current_total[10])
    
    battery_consumption = np.trapezoid(current_total, t_input) # [Ah]
    print(battery_consumption)


    

        # ------------------------------------------------------
    # Plotting - 3-figure grouped layout
    # ------------------------------------------------------

    # Common derived quantities
    radius = np.sqrt(px**2 + py**2)
    speed  = np.sqrt(vx**2 + vy**2 + vz**2)

    # -----------------------------
    # FIGURE 1: 3D trajectory (own figure)
    # -----------------------------
    fig1 = plt.figure(figsize=(8, 7))
    ax = fig1.add_subplot(111, projection='3d')
    ax.plot(px, py, pz, label='Trajectory', linewidth=2)
    ax.scatter([0], [0], [1.0], c='r', s=100, label='Hover target')
    ax.set_xlabel('x [m]', fontsize=16)
    ax.set_ylabel('y [m]', fontsize=16)
    ax.set_zlabel('z [m]', fontsize=16)
    ax.set_title('3D Position (Hover)', fontsize=18)
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
    axs[0].set_ylabel('Position [m]', fontsize=14)
    axs[0].set_title('Position', fontsize=16)
    axs[0].grid(True)
    axs[0].legend()

    # Velocity
    axs[1].plot(t, vx, linewidth=2, label='vx')
    axs[1].plot(t, vy, linewidth=2, label='vy')
    axs[1].plot(t, vz, linewidth=2, label='vz')
    axs[1].axhline(0.0, linestyle='--', alpha=0.3)
    axs[1].set_ylabel('Velocity [m/s]', fontsize=14)
    axs[1].set_title('Velocity', fontsize=16)
    axs[1].grid(True)
    axs[1].legend()

    # Euler angles
    axs[2].plot(t, np.rad2deg(phi), linewidth=2, label='φ (roll)')
    axs[2].plot(t, np.rad2deg(theta), linewidth=2, label='θ (pitch)')
    axs[2].plot(t, np.rad2deg(psi), linewidth=2, label='ψ (yaw)')
    axs[2].axhline(0.0, linestyle='--', alpha=0.3)
    axs[2].set_xlabel('Time [s]', fontsize=14)
    axs[2].set_ylabel('Angle [deg]', fontsize=14)
    axs[2].set_title('Euler Angles', fontsize=16)
    axs[2].grid(True)
    axs[2].legend()

    # Angular rates
    axs[3].plot(t, np.rad2deg(p), linewidth=2, label='p (roll rate)')
    axs[3].plot(t, np.rad2deg(q), linewidth=2, label='q (pitch rate)')
    axs[3].plot(t, np.rad2deg(r), linewidth=2, label='r (yaw rate)')
    axs[3].axhline(0.0, linestyle='--', alpha=0.3)
    axs[3].set_xlabel('Time [s]', fontsize=14)
    axs[3].set_ylabel('Rate [deg/s]', fontsize=14)
    axs[3].set_title('Angular Rates', fontsize=16)
    axs[3].grid(True)
    axs[3].legend()

    fig2.suptitle('State Time Histories (12 DOF)', fontsize=18)
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
    ax_u.set_xlabel('Time [s]', fontsize=12)
    ax_u.set_ylabel('Thrust [N]', fontsize=12)
    ax_u.set_title('Inputs u(t)', fontsize=14)
    ax_u.set_ylim(-1, 12)
    ax_u.grid(True)

    ax_u2 = ax_u.twinx()
    ax_u2.plot(t_input, tau_phi,   linewidth=2, linestyle='--', label='τφ')
    ax_u2.plot(t_input, tau_theta, linewidth=2, linestyle='--', label='τθ')
    ax_u2.plot(t_input, tau_psi,   linewidth=2, linestyle='--', label='τψ')
    ax_u2.set_ylabel('Torque [N·m]', fontsize=12)

    # Combined legend
    lines1, labels1 = ax_u.get_legend_handles_labels()
    lines2, labels2 = ax_u2.get_legend_handles_labels()
    ax_u2.legend(lines1 + lines2, labels1 + labels2, loc='best')
    
    # (2) RPM
    axs3[1].plot(t_input, rpm1, linewidth=2, label='rpm1')
    axs3[1].plot(t_input, rpm2, linewidth=2, label='rpm2')
    axs3[1].plot(t_input, rpm3, linewidth=2, label='rpm3')
    axs3[1].plot(t_input, rpm4, linewidth=2, label='rpm4')
    axs3[1].set_xlabel('Time [s]', fontsize=12)
    axs3[1].set_ylabel('Revolutions per Minute', fontsize=12)
    axs3[1].set_title('Motor RPMs', fontsize=14)
    #axs3[1].set_ylim(-0.1, 1.5)
    axs3[1].grid(True)
    axs3[1].legend()


    # (2) Power
    # (2) Power (rotor-based electrical power)
    axs3[2].plot(t_input, P_elec, linewidth=2)
    axs3[2].set_xlabel('Time [s]', fontsize=12)
    axs3[2].set_ylabel('Power [W]', fontsize=12)
    axs3[2].set_title('Electrical Power', fontsize=14)
    axs3[2].set_ylim(-10, 180)
    axs3[2].grid(True)


    # (3) Energy
    axs3[3].plot(t_input, E_cumulative, linewidth=2)
    axs3[3].set_xlabel('Time [s]', fontsize=12)
    axs3[3].set_ylabel('Energy [J]', fontsize=12)
    axs3[3].set_title('Cumulative Electrical Energy', fontsize=14)
    axs3[3].set_ylim(-500, 6000)
    axs3[3].grid(True)


    fig3.suptitle('Control Effort and Energy', fontsize=18)
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
    print(f"  Total energy consumed: {E_total:.2f} J")
    print(f"  Average power: {np.mean(P_elec):.2f} W")
    print(f"  Peak power: {np.max(P_elec):.2f} W")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    main()