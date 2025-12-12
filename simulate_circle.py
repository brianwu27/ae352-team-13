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


# -----------------------------
# Closed-loop dynamics
# -----------------------------
def make_closed_loop_rhs(A, B, K):
    """
    f(t, x) for:
        ẋ = A x + B u
        u = u_ref(t) - K (x - x_ref(t))

    where u_ref makes (x_ref, u_ref) follow the desired circle
    in the *linear* dynamics.
    """
    def f(t, x):
        x_ref = circle_x_ref(t)
        x_ref_dot = circle_x_ref_dot(t)

        # Solve for feed-forward input u_ref:
        #   ẋ_ref = A x_ref + B u_ref
        # => u_ref (least-squares) = B^+ (ẋ_ref - A x_ref)
        rhs = x_ref_dot - A @ x_ref
        u_ref, *_ = la.lstsq(B, rhs, rcond=None)  # shape (4,)

        # Feedback on tracking error
        e = x - x_ref
        du = -K @ e
        u = u_ref + du

        xdot = A @ x + B @ u
        return xdot

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

    speed_xy = np.sqrt(vx**2 + vy**2)
    radius_xy = np.sqrt(px**2 + py**2)

    # ---------- plots ----------
    plt.figure(figsize=(13, 8))

    # 3D position
    ax1 = plt.subplot(2, 3, 1, projection='3d')
    ax1.plot(px, py, pz, label='Trajectory')
    ax1.set_xlabel('x [m]')
    ax1.set_ylabel('y [m]')
    ax1.set_zlabel('z [m]')
    ax1.set_title('3D Position')
    ax1.grid(True)
    ax1.legend()

    # XY path
    ax2 = plt.subplot(2, 3, 2)
    ax2.plot(px, py, label='Trajectory')
    R = 2.0
    th = np.linspace(0, 2*np.pi, 200)
    ax2.plot(R*np.cos(th), R*np.sin(th), 'r--', label='Reference circle')
    ax2.set_aspect('equal', 'box')
    ax2.set_xlabel('x [m]')
    ax2.set_ylabel('y [m]')
    ax2.set_title('XY Path')
    ax2.grid(True)
    ax2.legend()

    # Altitude
    ax3 = plt.subplot(2, 3, 3)
    ax3.plot(t, pz, label='pz(t)')
    ax3.axhline(1.0, color='r', linestyle='--', label='Target: 1 m')
    ax3.set_xlabel('Time [s]')
    ax3.set_ylabel('Altitude z [m]')
    ax3.set_title('Altitude vs Time')
    ax3.grid(True)
    ax3.legend()

    # Horizontal speed
    ax4 = plt.subplot(2, 3, 4)
    ax4.plot(t, speed_xy, label='|v_xy|')
    ax4.axhline(0.5, color='r', linestyle='--', label='Target: 0.5 m/s')
    ax4.set_xlabel('Time [s]')
    ax4.set_ylabel('Speed [m/s]')
    ax4.set_title('Horizontal Speed')
    ax4.set_ylim(0, 1)
    ax4.grid(True)
    ax4.legend()

    # Radius from origin
    ax5 = plt.subplot(2, 3, 5)
    ax5.plot(t, radius_xy, label='radius')
    ax5.axhline(2.0, color='r', linestyle='--', label='Target: 2 m')
    ax5.set_xlabel('Time [s]')
    ax5.set_ylabel('Radius [m]')
    ax5.set_title('Distance from Origin')
    ax5.set_ylim(0, 4)
    ax5.grid(True)
    ax5.legend()

    # Attitude
    ax6 = plt.subplot(2, 3, 6)
    ax6.plot(t, np.rad2deg(phi), label='φ (roll)')
    ax6.plot(t, np.rad2deg(theta), label='θ (pitch)')
    ax6.set_xlabel('Time [s]')
    ax6.set_ylabel('Angle [deg]')
    ax6.set_title('Attitude')
    ax6.grid(True)
    ax6.legend()

    plt.tight_layout()

    # Stats over last 30 s
    mask = t > 30.0
    print("\n=== Circle Tracking Performance (last 30 s) ===")
    print(f"Mean radius: {np.mean(radius_xy[mask]):.3f} m (target 2.0)")
    print(f"Radius std dev: {np.std(radius_xy[mask]):.4f} m")
    print(f"Mean speed:  {np.mean(speed_xy[mask]):.3f} m/s (target 0.5)")
    print(f"Speed std dev: {np.std(speed_xy[mask]):.4f} m/s")
    print(f"Mean altitude: {np.mean(pz[mask]):.3f} m (target 1.0)")
    print(f"Altitude std dev: {np.std(pz[mask]):.4f} m")

    plt.show()


if __name__ == "__main__":
    main()
