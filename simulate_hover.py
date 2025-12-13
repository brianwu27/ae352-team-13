# simulate_hover.py
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

from dynamics import QuadParams


def hover_reference():   # Defined desired state, hover at (0,0,1), zero rates
    x_ref = np.zeros(12)
    x_ref[2] = 1.0   # pz = 1 m
    return x_ref


def make_closed_loop_rhs(A, B, K, params):     # Build f = x_dot = Ax + Bu
    input_history = []
    
    def f(t, x):
        x_ref = hover_reference()
        dx = x - x_ref          
        du = -K @ dx            
        
        u_hover = np.array([params.m * params.g, 0, 0, 0])  # hover equilibrium
        u = u_hover + du
        input_history.append((t, u.copy()))
        
        xdot = A @ dx + B @ du
        return xdot
    
    f.input_history = input_history
    return f

# Simulation and plotting
def main():
    params = QuadParams()
    
    L = params.L
    k_tau = params.k_tau
    k = params.k
    A, B = params.linear_matrices()     # Obtain state and input matrices
    K = params.lqr_gain()               # Obtain gain matrix

    # Initialize at z = 0.0 (on ground) with 1 degree roll and pitch error
    x0 = np.zeros(12)
    x0[2] = 0.0                  
    x0[6] = np.deg2rad(1.0)       
    x0[7] = np.deg2rad(-1.0)  

    # SImulate for 130 seconds
    t_span = (0.0, 130)
    t_eval = np.linspace(t_span[0], t_span[1], 13001)
    
    # Build linear closed-loop system
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

    px = x[0, :]
    py = x[1, :]
    pz = x[2, :]
    vx = x[3, :]
    vy = x[4, :]
    vz = x[5, :]
    phi = x[6, :]
    theta = x[7, :]
    psi = x[8, :]
    p = x[9, :]
    q = x[10, :]
    r = x[11, :]

    # Extract input history
    t_input = np.array([item[0] for item in f_cl.input_history])
    u_history = np.array([item[1] for item in f_cl.input_history])
    
    T_total = u_history[:, 0]
    tau_phi = u_history[:, 1]
    tau_theta = u_history[:, 2]
    tau_psi = u_history[:, 3]
    
    # Calculating motor thrusts
    M = np.array([
        [1.0,   1.0,   1.0,   1.0],
        [0.0,  -L,     0.0,  +L  ],
        [-L,    0.0,  +L,     0.0],
        [k_tau, -k_tau, k_tau, -k_tau]
    ])

    U = np.vstack((T_total, tau_phi, tau_theta, tau_psi))
    T_motors = np.linalg.solve(M, U) 
    T1, T2, T3, T4 = T_motors[0, :], T_motors[1, :], T_motors[2, :], T_motors[3, :]

    # Power and Energy Calculations
    rho = 1.225  # kg/m^3
    R_rotor = getattr(params, "rotor_radius", 0.0635)   # radius
    A_rotor = np.pi * R_rotor**2                        # area
    eta = getattr(params, "eta", 0.7)                   # overall efficiency

    def thrust_to_power(T):
        T = np.maximum(T, 0.0)    # avoid negative thrust
        return (T**1.5) / np.sqrt(2.0 * rho * A_rotor)

    # Mechanical power for each rotor
    P_mech1 = thrust_to_power(T1)
    P_mech2 = thrust_to_power(T2)
    P_mech3 = thrust_to_power(T3)
    P_mech4 = thrust_to_power(T4)
    P_mech = P_mech1 + P_mech2 + P_mech3 + P_mech4
    
    # Electrical power using mechanical and efficiency factor
    P_elec = P_mech / eta
    energy_used = np.mean(P_elec)*120/3600   # Battery power used in Watt-hours
    print(f'Battery energy used: {energy_used:.2f} Wh')

    # Cumulative energy
    E_cumulative = np.cumsum(P_elec[:-1] * np.diff(t_input))
    E_cumulative = np.insert(E_cumulative, 0, 0.0)
    
    # Angular velocity
    omega1 = np.sqrt(T1/k)
    omega2 = np.sqrt(T2/k)
    omega3 = np.sqrt(T3/k)
    omega4 = np.sqrt(T4/k)
    
    # Convert to RPM
    rpm1 = omega1*60/(2*np.pi)
    rpm2 = omega2*60/(2*np.pi)
    rpm3 = omega3*60/(2*np.pi)
    rpm4 = omega4*60/(2*np.pi)
    
    # Plotting 3D position
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

    
    # Plotting drone state over time
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
    axs[2].plot(t, np.rad2deg(phi), linewidth=2, label=r'$\phi$ (roll)')
    axs[2].plot(t, np.rad2deg(theta), linewidth=2, label=r'$\theta$ (pitch)')
    axs[2].plot(t, np.rad2deg(psi), linewidth=2, label=r'$\psi$ (yaw)')
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


    # Plotting control inputs, RPMs, and energy
    fig3, axs3 = plt.subplots(2, 2, figsize=(14, 9))
    axs3 = axs3.ravel()

    # Control inputs
    ax_u = axs3[0]
    ax_u.plot(t_input, T_total, linewidth=2, label='F (total thrust)')
    ax_u.set_xlabel('Time [s]', fontsize=12)
    ax_u.set_ylabel('Thrust [N]', fontsize=12)
    ax_u.set_title('Inputs u(t)', fontsize=14)
    ax_u.set_ylim(-1, 12)
    ax_u.grid(True)
    ax_u2 = ax_u.twinx()
    ax_u2.plot(t_input, tau_phi, linewidth=2, linestyle='--', label=r'$\tau \phi$')
    ax_u2.plot(t_input, tau_theta, linewidth=2, linestyle='--', label=r'$\tau \theta$')
    ax_u2.plot(t_input, tau_psi, linewidth=2, linestyle='--', label=r'$\tau\psi$')
    ax_u2.set_ylabel('Torque [N·m]', fontsize=12)
    lines1, labels1 = ax_u.get_legend_handles_labels()
    lines2, labels2 = ax_u2.get_legend_handles_labels()
    ax_u2.legend(lines1 + lines2, labels1 + labels2, loc='best')
    
    # RPM
    axs3[1].plot(t_input, rpm1, linewidth=2, label='rpm1')
    axs3[1].plot(t_input, rpm2, linewidth=2, label='rpm2')
    axs3[1].plot(t_input, rpm3, linewidth=2, label='rpm3')
    axs3[1].plot(t_input, rpm4, linewidth=2, label='rpm4')
    axs3[1].set_xlabel('Time [s]', fontsize=12)
    axs3[1].set_ylabel('Revolutions per Minute', fontsize=12)
    axs3[1].set_title('Motor RPMs', fontsize=14)
    axs3[1].grid(True)
    axs3[1].legend()

    # Power
    axs3[2].plot(t_input, P_elec, linewidth=2)
    axs3[2].set_xlabel('Time [s]', fontsize=12)
    axs3[2].set_ylabel('Power [W]', fontsize=12)
    axs3[2].set_title('Electrical Power', fontsize=14)
    axs3[2].set_ylim(-10, 180)
    axs3[2].grid(True)

    # Cumulative electrical energy
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


if __name__ == "__main__":
    main()