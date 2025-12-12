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


def make_closed_loop_rhs(A, B, K):
    """
    Build f(t, x) = x_dot for the linear closed-loop system:
        Δx = x - x_ref
        Δu = -K Δx
        ẋ = A Δx + B Δu
    For hover, x_ref is constant in time, so this is valid.
    """
    def f(t, x):
        x_ref = hover_reference(t)
        dx = x - x_ref          # state error
        du = -K @ dx            # state feedback
        xdot = A @ dx + B @ du  # linear error dynamics
        return xdot

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

    t_span = (0.0, 150)
    t_eval = np.linspace(t_span[0], t_span[1], 15001)

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

    # ------------------------------------------------------
    # Plotting
    # ------------------------------------------------------
    plt.figure(figsize=(12, 8))

    # 3D trajectory (but here hover at origin)
    ax = plt.subplot(2, 2, 1, projection='3d')
    ax.plot(px, py, pz, label='Trajectory')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_zlabel('z [m]')
    ax.set_title('3D Position')
    ax.scatter([0], [0], [1.0], c='r', label='Hover target')
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(-0.5, 0.5)
    ax.set_zlim(0, 1.2)
    ax.legend()

    # Altitude vs time
    ax2 = plt.subplot(2, 2, 2)
    ax2.plot(t, pz, label='pz(t)')
    ax2.axhline(1.0, color='r', linestyle='--', label='Target: 1 m')
    ax2.set_xlabel('Time [s]')
    ax2.set_ylabel('Altitude z [m]')
    ax2.set_title('Altitude vs Time')
    ax2.grid(True)
    ax2.legend()

    # Horizontal position vs time
    ax3 = plt.subplot(2, 2, 3)
    ax3.plot(t, px, label='px')
    ax3.plot(t, py, label='py')
    ax3.axhline(0.0, color='k', linestyle='--', linewidth=0.5)
    ax3.set_xlabel('Time [s]')
    ax3.set_ylabel('Position [m]')
    ax3.set_title('Horizontal Drift')
    ax3.grid(True)
    ax3.set_ylim(-0.005, 0.005)
    ax3.legend()

    # Roll & pitch angles
    ax4 = plt.subplot(2, 2, 4)
    ax4.plot(t, np.rad2deg(phi), label='φ (roll)')
    ax4.plot(t, np.rad2deg(theta), label='θ (pitch)')
    ax4.axhline(0.0, color='k', linestyle='--', linewidth=0.5)
    ax4.set_xlabel('Time [s]')
    ax4.set_ylabel('Angle [deg]')
    ax4.set_title('Attitude')
    ax4.grid(True)
    ax4.legend()

    plt.tight_layout()

    # Some stats
    radius = np.sqrt(px**2 + py**2)
    print("\n=== Linear Hover Performance ===")
    print(f"Mean altitude: {np.mean(pz):.4f} m (target: 1.0 m)")
    print(f"Altitude std dev: {np.std(pz):.6f} m")
    print(f"Max XY drift radius: {np.max(radius):.6f} m")
    print(f"Max |φ|: {np.max(np.abs(phi))*180/np.pi:.3f} deg")
    print(f"Max |θ|: {np.max(np.abs(theta))*180/np.pi:.3f} deg")

    plt.show()


if __name__ == "__main__":
    main()
