# simulate_circle.py
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from dynamics import QuadParams


# -----------------------------
# Reference trajectory
# -----------------------------
def circle_x_ref(t: float) -> np.ndarray:
    """
    Reference state for a circle of radius 2 m at 0.5 m/s,
    altitude 1 m, centered at the origin.

    x = [px, py, pz, vx, vy, vz, φ, θ, ψ, p, q, r]^T
    """
    R = 2.0       # m
    v = 0.5       # m/s
    omega = v / R # rad/s

    px = R * np.cos(omega * t)
    py = R * np.sin(omega * t)
    pz = 1.0

    vx = -R * omega * np.sin(omega * t)
    vy =  R * omega * np.cos(omega * t)
    vz = 0.0

    x_ref = np.zeros(12)
    x_ref[0] = px
    x_ref[1] = py
    x_ref[2] = pz
    x_ref[3] = vx
    x_ref[4] = vy
    x_ref[5] = vz
    # φ, θ, ψ, p, q, r remain 0
    return x_ref


def circle_x_ref_dot(t: float) -> np.ndarray:
    """
    Time derivative of the reference state.
    Needed for feed-forward input u_ref.
    """
    R = 2.0
    v = 0.5
    omega = v / R

    # velocities (same as vx, vy above)
    vx = -R * omega * np.sin(omega * t)
    vy =  R * omega * np.cos(omega * t)
    vz = 0.0

    # accelerations
    ax = -R * omega**2 * np.cos(omega * t)
    ay = -R * omega**2 * np.sin(omega * t)
    az = 0.0

    x_ref_dot = np.zeros(12)
    x_ref_dot[0:3] = [vx,  vy,  vz]
    x_ref_dot[3:6] = [ax,  ay,  az]
    # rest remain 0
    return x_ref_dot


def make_closed_loop_rhs(A, B, K):
    """
    f(t, x) for:
        ẋ = A x + B u
        u = u_ref(t) - K (x - x_ref(t))
    """
    input_history = []  # <-- log (t, u)

    def f(t, x):
        x_ref = circle_x_ref(t)
        x_ref_dot = circle_x_ref_dot(t)

        rhs = x_ref_dot - A @ x_ref
        u_ref, *_ = la.lstsq(B, rhs, rcond=None)  # shape (4,)

        e = x - x_ref
        du = -K @ e
        u = u_ref + du

        input_history.append((t, u.copy()))  # <-- log inputs for plots

        xdot = A @ x + B @ u
        return xdot

    f.input_history = input_history  # <-- attach
    return f




# -----------------------------
# Simulation + plotting
# -----------------------------
def main():
    params = QuadParams()
    A, B = params.linear_matrices()
    K = params.lqr_gain()

    # Start exactly on the reference trajectory at t = 0
    x0 = circle_x_ref(0.0)

    # Simulate for at least 60 s
    t_span = (0.0, 60.0)
    t_eval = np.linspace(t_span[0], t_span[1], 3001)

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

    # -----------------------------
    # Build derived quantities (needed for plots)
    # -----------------------------
    speed_xy  = np.sqrt(vx**2 + vy**2)
    radius_xy = np.sqrt(px**2 + py**2)

        # -----------------------------
    # Inputs u(t) from logged history
    # -----------------------------
    t_input   = np.array([item[0] for item in f_cl.input_history])
    u_history = np.array([item[1] for item in f_cl.input_history])

    F         = u_history[:, 0]
    tau_phi   = u_history[:, 1]
    tau_theta = u_history[:, 2]
    tau_psi   = u_history[:, 3]
    
    F_phys = params.m * params.g + F   # interpret F as ΔF around hover


     

    # -----------------------------
    # Motor thrusts T1..T4 (best-effort mixer inversion)
    # If params.L and params.k_tau exist, invert standard "plus" mixer.
    # Otherwise fall back to equal split (still gives a meaningful load plot).
    # -----------------------------
    try:
        L = params.L
        k_tau = params.k_tau

        M = np.array([
            [1.0,  1.0,  1.0,  1.0],
            [0.0, -L,   0.0, +L  ],
            [-L,  0.0, +L,   0.0],
            [k_tau, -k_tau, k_tau, -k_tau]
        ])

        U = np.vstack((F_phys, tau_phi, tau_theta, tau_psi))  # (4, N)
        T_motors = np.linalg.solve(M, U)                 # (4, N)
    except Exception:
        T_motors = np.vstack((F_phys/4, F_phys/4, F_phys/4, F_phys/4))

    T1, T2, T3, T4 = T_motors[0, :], T_motors[1, :], T_motors[2, :], T_motors[3, :]
    
        # -----------------------------
    # NEW: Rotor-based power + energy (includes baseline hover power)
    # Using actuator-disk / momentum theory proxy:
    #   P_i ≈ T_i^(3/2) / sqrt(2 ρ A)
    # Electrical power: P_elec = P_mech / η
    # -----------------------------
    rho = 1.225  # kg/m^3
    R_rotor = getattr(params, "rotor_radius", 0.0635)  # m 
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

    # -----------------------------
    # FIGURE 1: 3D trajectory (own figure)
    # -----------------------------
    fig1 = plt.figure(figsize=(8, 6))
    ax = fig1.add_subplot(111, projection='3d')
    ax.plot(px, py, pz, linewidth=2, label='Trajectory')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_zlabel('z [m]')
    ax.set_title('3D Position (Circle)')
    ax.grid(True)
    ax.legend()

    # -----------------------------
    # FIGURE 2: 12 DOFs grouped (2x2) in one figure
    # -----------------------------
    fig2, axs = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
    axs = axs.ravel()

    # Position
    axs[0].plot(t, px, linewidth=2, label='px')
    axs[0].plot(t, py, linewidth=2, label='py')
    axs[0].plot(t, pz, linewidth=2, label='pz')
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
    axs[2].set_ylim(-1.0, 1.0)
    axs[2].grid(True)
    axs[2].legend()

    # Body rates
    axs[3].plot(t, np.rad2deg(p), linewidth=2, label='p (roll rate)')
    axs[3].plot(t, np.rad2deg(q), linewidth=2, label='q (pitch rate)')
    axs[3].plot(t, np.rad2deg(r), linewidth=2, label='r (yaw rate)')
    axs[3].axhline(0.0, linestyle='--', alpha=0.3)
    axs[3].set_xlabel('Time [s]')
    axs[3].set_ylabel('Rate [deg/s]')
    axs[3].set_title('Angular Rates')
    axs[3].set_ylim(-1.9, 1.9)
    axs[3].grid(True)
    axs[3].legend()

    fig2.suptitle('State Time Histories (12 DOF)', fontsize=14)
    fig2.tight_layout()
    fig2.subplots_adjust(top=0.92)

    fig3, axs3 = plt.subplots(2, 2, figsize=(14, 9))
    axs3 = axs3.ravel()

    # (1) Inputs: thrust + torques (twin y-axis)
    ax_u = axs3[0]
    ax_u.plot(t_input, F_phys, linewidth=2, label='F (thrust)')  # F_plot = T_total (hover) OR F_phys (circle)
    ax_u.set_xlabel('Time [s]')
    ax_u.set_ylabel('Thrust [N]')
    ax_u.set_title('Inputs u(t)')
    ax_u.set_ylim(-0.5, 6)
    ax_u.grid(True)

    ax_u2 = ax_u.twinx()
    ax_u2.plot(t_input, tau_phi,   linewidth=2, linestyle='--', label='τφ')
    ax_u2.plot(t_input, tau_theta, linewidth=2, linestyle='--', label='τθ')
    ax_u2.plot(t_input, tau_psi,   linewidth=2, linestyle='--', label='τψ')
    ax_u2.set_ylim(-0.1, 0.1)
    ax_u2.set_ylabel('Torque [N·m]')

    # Combined legend
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

    # (3) Power
    axs3[2].plot(t_input, P_elec, linewidth=2)
    axs3[2].set_xlabel('Time [s]')
    axs3[2].set_ylabel('Power [W] (rotor model)')
    axs3[2].set_title('Power')
    axs3[2].set_ylim(-5, 50)
    axs3[2].grid(True)

    # (4) Energy
    axs3[3].plot(t_input, E_cumulative, linewidth=2)
    axs3[3].set_xlabel('Time [s]')
    axs3[3].set_ylabel('Energy [J] (rotor model)')
    axs3[3].set_title('Cumulative Energy')
    axs3[3].set_ylim(-100, 2500)
    axs3[3].grid(True)

    fig3.suptitle('Control Effort and Energy', fontsize=14)
    fig3.tight_layout()
    fig3.subplots_adjust(top=0.92)

    plt.show()




if __name__ == "__main__":
    main()
