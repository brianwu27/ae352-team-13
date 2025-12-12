# simulate_waypoints.py
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from dynamics import QuadParams


# -----------------------------
# Phase timing
# -----------------------------
v_asc   = 0.5    # m/s vertical ascent speed
v_horiz = 1.0    # m/s horizontal segments
v_desc  = -0.01  # m/s downward (landing) speed, magnitude 1 cm/s

t1 = 2.0
t2 = t1 + 5.0    # 7
t3 = t2 + 2.0    # 9
t4 = t3 + 3.0    # 12
t5 = t4 + 5.0    # 17
t6 = t5 + 2.0    # 19
t7 = t6 + 100.0  # 119


# -----------------------------
# Reference trajectory
# -----------------------------
def waypoint_x_ref(t: float) -> np.ndarray:
    """
    State: x = [px, py, pz, vx, vy, vz, φ, θ, ψ, p, q, r]^T
    """
    px = py = pz = 0.0
    vx = vy = vz = 0.0
    phi = theta = 0.0
    psi = 0.0
    p = q = r = 0.0

    if t < t1:
        pz = v_asc * t
        pz = min(pz, 1.0)
        vz = v_asc

    elif t < t2:
        tau = t - t1
        px = v_horiz * tau
        px = min(px, 5.0)
        py = 0.0
        pz = 1.0
        vx = v_horiz
        vy = 0.0
        vz = 0.0

    elif t < t3:
        px, py, pz = 5.0, 0.0, 1.0

    elif t < t4:
        px, py, pz = 5.0, 0.0, 1.0
        psi0 = 0.0
        psi_f = np.pi / 2.0
        tau = t - t3
        T_yaw = t4 - t3
        psi = psi0 + (psi_f - psi0) * tau / T_yaw
        r = (psi_f - psi0) / T_yaw

    elif t < t5:
        px = 5.0
        tau = t - t4
        py = v_horiz * tau
        py = min(py, 5.0)
        pz = 1.0
        vx = 0.0
        vy = v_horiz
        vz = 0.0
        psi = np.pi / 2.0

    elif t < t6:
        px, py, pz = 5.0, 5.0, 1.0
        psi = np.pi / 2.0

    elif t < t7:
        px, py = 5.0, 5.0
        tau = t - t6
        pz = 1.0 + v_desc * tau
        pz = max(pz, 0.0)
        vz = v_desc
        psi = np.pi / 2.0

    else:
        px, py, pz = 5.0, 5.0, 0.0
        psi = np.pi / 2.0

    return np.array([px, py, pz, vx, vy, vz, phi, theta, psi, p, q, r])


def waypoint_x_ref_dot(t: float) -> np.ndarray:
    px_dot = py_dot = pz_dot = 0.0
    vx_dot = vy_dot = vz_dot = 0.0
    phi_dot = theta_dot = 0.0
    psi_dot = 0.0
    p_dot = q_dot = r_dot = 0.0

    if t < t1:
        pz_dot = v_asc

    elif t < t2:
        px_dot = v_horiz

    elif t < t3:
        pass

    elif t < t4:
        psi0 = 0.0
        psi_f = np.pi / 2.0
        T_yaw = t4 - t3
        psi_dot = (psi_f - psi0) / T_yaw
        r_dot = 0.0

    elif t < t5:
        py_dot = v_horiz

    elif t < t6:
        pass

    elif t < t7:
        pz_dot = v_desc

    else:
        pass

    return np.array([
        px_dot, py_dot, pz_dot,
        vx_dot, vy_dot, vz_dot,
        phi_dot, theta_dot, psi_dot,
        p_dot, q_dot, r_dot
    ])


# -----------------------------
# Closed-loop dynamics
# -----------------------------
def make_closed_loop_rhs(A, B, K):
    """
    ẋ = A x + B (u_ref(t) - K (x - x_ref(t)))
    where u_ref solves (least squares): ẋ_ref ≈ A x_ref + B u_ref
    """
    input_history = []

    def f(t, x):
        x_ref = waypoint_x_ref(t)
        x_ref_dot = waypoint_x_ref_dot(t)

        rhs = x_ref_dot - A @ x_ref
        u_ref, *_ = la.lstsq(B, rhs, rcond=None)  # (4,)

        e = x - x_ref
        du = -K @ e
        u = u_ref + du

        input_history.append((t, u.copy()))

        xdot = A @ x + B @ u
        return xdot

    f.input_history = input_history
    return f


# -----------------------------
# Simulation + plotting
# -----------------------------
def main():
    params = QuadParams()
    A, B = params.linear_matrices()
    K = params.lqr_gain()

    x0 = np.zeros(12)

    t_span = (0.0, t7 + 5.0)
    t_eval = np.linspace(t_span[0], t_span[1], 4001)

    f_cl = make_closed_loop_rhs(A, B, K)

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

    px    = x[0, :]
    py    = x[1, :]
    pz    = x[2, :]
    vx    = x[3, :]
    vy    = x[4, :]
    vz    = x[5, :]
    phi   = x[6, :]
    theta = x[7, :]
    psi   = x[8, :]
    p     = x[9, :]
    q     = x[10, :]
    r     = x[11, :]

    speed_xy = np.sqrt(vx**2 + vy**2)

    # -----------------------------
    # Inputs u(t) from logged history
    # -----------------------------
    t_input   = np.array([item[0] for item in f_cl.input_history])
    u_history = np.array([item[1] for item in f_cl.input_history])

    F         = u_history[:, 0]   # total thrust input (as modeled in your linear system)
    tau_phi   = u_history[:, 1]
    tau_theta = u_history[:, 2]
    tau_psi   = u_history[:, 3]
    
    F_phys = params.m * params.g + F

    # -----------------------------
    # Motor thrusts T1..T4 (mixer inversion if params has L, k_tau)
    # -----------------------------
    try:
        L = params.L
        k_tau = params.k_tau

        M = np.array([
            [1.0,   1.0,   1.0,   1.0],
            [0.0,  -L,     0.0,  +L  ],
            [-L,    0.0,  +L,     0.0],
            [k_tau, -k_tau, k_tau, -k_tau]
        ])

        U = np.vstack((F_phys, tau_phi, tau_theta, tau_psi))  # (4, N)
        T_motors = np.linalg.solve(M, U)                 # (4, N)
    except Exception:
        T_motors = np.vstack((F_phys/4, F_phys/4, F_phys/4, F_phys/4))

    T1, T2, T3, T4 = T_motors[0, :], T_motors[1, :], T_motors[2, :], T_motors[3, :]

    # -----------------------------
    # Rotor-based electrical power + energy
    # (actuator disk proxy, includes baseline because it uses absolute thrust)
    # -----------------------------
    rho = 1.225  # kg/m^3
    R_rotor = getattr(params, "rotor_radius", 0.0635)  # m
    A_rotor = np.pi * R_rotor**2
    eta = getattr(params, "eta", 0.7)

    def thrust_to_power(T):
        T = np.maximum(T, 0.0)
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

    # -----------------------------
    # FIGURE 1: 3D trajectory (own figure)
    # -----------------------------
    fig1 = plt.figure(figsize=(8, 6))
    ax = fig1.add_subplot(111, projection='3d')
    ax.plot(px, py, pz, linewidth=2, label='Trajectory')

    waypoints = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [5.0, 0.0, 1.0],
        [5.0, 5.0, 1.0],
        [5.0, 5.0, 0.0],
    ])
    ax.scatter(waypoints[:, 0], waypoints[:, 1], waypoints[:, 2], c='red', s=40, label='Waypoints')

    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_zlabel('z [m]')
    ax.set_title('3D Position (Waypoints)')
    ax.grid(True)
    ax.legend()
    fig1.tight_layout()

    # -----------------------------
    # FIGURE 2: 12 DOFs grouped (2x2)
    # -----------------------------
    fig2, axs = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
    axs = axs.ravel()

    axs[0].plot(t, px, linewidth=2, label='px')
    axs[0].plot(t, py, linewidth=2, label='py')
    axs[0].plot(t, pz, linewidth=2, label='pz')
    axs[0].axhline(1.0, linestyle='--', alpha=0.4, label='z=1 m')
    axs[0].set_ylabel('Position [m]')
    axs[0].set_title('Position')
    axs[0].set_ylim(-0.5, 6)
    axs[0].grid(True)
    axs[0].legend()

    axs[1].plot(t, vx, linewidth=2, label='vx')
    axs[1].plot(t, vy, linewidth=2, label='vy')
    axs[1].plot(t, vz, linewidth=2, label='vz')
    axs[1].axhline(0.0, linestyle='--', alpha=0.3)
    axs[1].set_ylabel('Velocity [m/s]')
    axs[1].set_title('Velocity')
    axs[1].set_ylim(-0.5, 1.4)
    axs[1].grid(True)
    axs[1].legend()

    axs[2].plot(t, np.rad2deg(phi), linewidth=2, label='φ (roll)')
    axs[2].plot(t, np.rad2deg(theta), linewidth=2, label='θ (pitch)')
    axs[2].plot(t, np.rad2deg(psi), linewidth=2, label='ψ (yaw)')
    axs[2].axhline(0.0, linestyle='--', alpha=0.3)
    axs[2].set_xlabel('Time [s]')
    axs[2].set_ylabel('Angle [deg]')
    axs[2].set_title('Euler Angles')
    axs[2].set_ylim(-100, 100)
    axs[2].grid(True)
    axs[2].legend()

    axs[3].plot(t, np.rad2deg(p), linewidth=2, label='p (roll rate)')
    axs[3].plot(t, np.rad2deg(q), linewidth=2, label='q (pitch rate)')
    axs[3].plot(t, np.rad2deg(r), linewidth=2, label='r (yaw rate)')
    axs[3].axhline(0.0, linestyle='--', alpha=0.3)
    axs[3].set_xlabel('Time [s]')
    axs[3].set_ylabel('Rate [deg/s]')
    axs[3].set_title('Angular Rates')
    axs[3].set_ylim(-200, 200)
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
    ax_u.plot(t_input, F_phys, linewidth=2, label='F (total thrust)')
    ax_u.set_xlabel('Time [s]')
    ax_u.set_ylabel('Thrust [N]')
    ax_u.set_title('Inputs u(t)')
    ax_u.set_ylim(-0.5, 8)
    ax_u.grid(True)

    ax_u2 = ax_u.twinx()
    ax_u2.plot(t_input, tau_phi,   linewidth=2, linestyle='--', label='τφ')
    ax_u2.plot(t_input, tau_theta, linewidth=2, linestyle='--', label='τθ')
    ax_u2.plot(t_input, tau_psi,   linewidth=2, linestyle='--', label='τψ')
    ax_u2.set_ylim(-3300, 3300)
    ax_u2.set_ylabel('Torque [N·m]')

    lines1, labels1 = ax_u.get_legend_handles_labels()
    lines2, labels2 = ax_u2.get_legend_handles_labels()
    ax_u2.legend(lines1 + lines2, labels1 + labels2, loc='best')

    # (2) RPM
    axs3[1].plot(t_input, rpm1, linewidth=2, label='rpm1')
    axs3[1].plot(t_input, rpm2, linewidth=2, label='rpm2')
    axs3[1].plot(t_input, rpm3, linewidth=2, label='rpm3')
    axs3[1].plot(t_input, rpm4, linewidth=2, label='rpm4')
    axs3[1].set_xlabel('Time [s]')
    axs3[1].set_ylabel('Revolutions per Minute')
    axs3[1].set_title('Motor RPMs')
    #axs3[1].set_ylim(-0.1, 1.5)
    axs3[1].grid(True)
    axs3[1].legend()

    # (2) Electrical power (rotor-based)
    axs3[2].plot(t_input, P_elec, linewidth=2)
    axs3[2].set_xlabel('Time [s]')
    axs3[2].set_ylabel('Power [W]')
    axs3[2].set_title('Electrical Power (rotor-based)')
    axs3[2].set_ylim(-5, 80)
    axs3[2].grid(True)

    # (3) Cumulative energy
    axs3[3].plot(t_input, E_cumulative, linewidth=2)
    axs3[3].set_xlabel('Time [s]')
    axs3[3].set_ylabel('Energy [J]')
    axs3[3].set_title('Cumulative Electrical Energy')
    axs3[3].set_ylim(-500, 5000)
    axs3[3].grid(True)

    fig3.suptitle('Control Effort and Energy', fontsize=14)
    fig3.tight_layout()
    fig3.subplots_adjust(top=0.92)

    plt.show()

    # -----------------------------
    # Some stats
    # -----------------------------
    mask1 = (t >= t1) & (t <= t2)
    avg_speed_seg1 = np.trapezoid(np.sqrt(vx[mask1]**2 + vy[mask1]**2), t[mask1]) / (t2 - t1)
    print(f"Average horizontal speed segment 1: {avg_speed_seg1:.3f} m/s")

    mask2 = (t >= t4) & (t <= t5)
    avg_speed_seg2 = np.trapezoid(np.sqrt(vx[mask2]**2 + vy[mask2]**2), t[mask2]) / (t5 - t4)
    print(f"Average horizontal speed segment 2: {avg_speed_seg2:.3f} m/s")

    print("\n=== Waypoint Mission Stats ===")
    print(f"Final position: x={px[-1]:.3f} m, y={py[-1]:.3f} m, z={pz[-1]:.3f} m")
    print(f"Final vertical speed: v_z={vz[-1]:.5f} m/s")
    print(f"Max |v_z| during final descent segment: {np.max(np.abs(vz[t > t6])):.5f} m/s")

    print("\nENERGY:")
    print(f"  Total energy consumed: {E_total:.2f} J")
    print(f"  Average power: {np.mean(P_elec):.2f} W")
    print(f"  Peak power: {np.max(P_elec):.2f} W")


if __name__ == "__main__":
    main()
