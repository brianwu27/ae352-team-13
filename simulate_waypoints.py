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

# ascent: 0 -> 1 m at 0.5 m/s -> 2 s
t1 = 2.0
# first straight line: 5 m at 1 m/s -> 5 s
t2 = t1 + 5.0    # 7
# hover 1: 2 s
t3 = t2 + 2.0    # 9
# yaw 90 deg left over 3 s
t4 = t3 + 3.0    # 12
# second straight line: 5 m at 1 m/s
t5 = t4 + 5.0    # 17
# hover 2: 2 s
t6 = t5 + 2.0    # 19
# descent: from 1 m to 0 m at 0.01 m/s -> 100 s
t7 = t6 + 100.0  # 119


# -----------------------------
# Reference trajectory
# -----------------------------
def waypoint_x_ref(t: float) -> np.ndarray:
    """
    Reference state x_ref(t) for the waypoint mission.

    State: x = [px, py, pz, vx, vy, vz, φ, θ, ψ, p, q, r]^T
    """
    # initialize
    px = py = pz = 0.0
    vx = vy = vz = 0.0
    phi = theta = 0.0
    psi = 0.0
    p = q = r = 0.0

    if t < t1:
        # Phase 1: vertical ascent from z=0 to z=1
        pz = v_asc * t
        if pz > 1.0:
            pz = 1.0
        vz = v_asc

    elif t < t2:
        # Phase 2: move in +x for 5 m at 1 m/s, at z = 1
        tau = t - t1
        px = v_horiz * tau
        if px > 5.0:
            px = 5.0
        py = 0.0
        pz = 1.0
        vx = v_horiz
        vy = 0.0
        vz = 0.0

    elif t < t3:
        # Phase 3: hover at (5, 0, 1)
        px, py, pz = 5.0, 0.0, 1.0

    elif t < t4:
        # Phase 4: yaw 90 deg to the left, position fixed
        px, py, pz = 5.0, 0.0, 1.0
        psi0 = 0.0
        psi_f = np.pi / 2.0
        tau = t - t3
        T_yaw = t4 - t3
        psi = psi0 + (psi_f - psi0) * tau / T_yaw
        r = (psi_f - psi0) / T_yaw  # constant yaw rate during maneuver

    elif t < t5:
        # Phase 5: move in +y for 5 m at 1 m/s, at z = 1
        px = 5.0
        tau = t - t4
        py = v_horiz * tau
        if py > 5.0:
            py = 5.0
        pz = 1.0
        vx = 0.0
        vy = v_horiz
        vz = 0.0
        psi = np.pi / 2.0

    elif t < t6:
        # Phase 6: hover at (5, 5, 1), yaw = 90 deg
        px, py, pz = 5.0, 5.0, 1.0
        psi = np.pi / 2.0

    elif t < t7:
        # Phase 7: slow vertical landing with speed 0.01 m/s
        px, py = 5.0, 5.0
        tau = t - t6
        pz = 1.0 + v_desc * tau
        if pz < 0.0:
            pz = 0.0
        vz = v_desc
        psi = np.pi / 2.0

    else:
        # After landing: stay on the ground at final position
        px, py, pz = 5.0, 5.0, 0.0
        psi = np.pi / 2.0

    x_ref = np.array([
        px, py, pz,
        vx, vy, vz,
        phi, theta, psi,
        p, q, r
    ])

    return x_ref


def waypoint_x_ref_dot(t: float) -> np.ndarray:
    """
    Time derivative of x_ref(t) for feed-forward computation.
    """
    # initialize
    px_dot = py_dot = pz_dot = 0.0
    vx_dot = vy_dot = vz_dot = 0.0
    phi_dot = theta_dot = 0.0
    psi_dot = 0.0
    p_dot = q_dot = r_dot = 0.0

    if t < t1:
        # Phase 1: ascent
        px_dot = 0.0
        py_dot = 0.0
        pz_dot = v_asc
        # velocities are constant => accelerations zero

    elif t < t2:
        # Phase 2: +x motion
        px_dot = v_horiz
        py_dot = 0.0
        pz_dot = 0.0

    elif t < t3:
        # Phase 3: hover
        pass  # all zeros

    elif t < t4:
        # Phase 4: yaw 90 deg
        psi0 = 0.0
        psi_f = np.pi / 2.0
        T_yaw = t4 - t3
        psi_dot = (psi_f - psi0) / T_yaw
        # r is the body yaw rate; in the linear model: psi̇ = r
        r_dot = 0.0

    elif t < t5:
        # Phase 5: +y motion
        px_dot = 0.0
        py_dot = v_horiz
        pz_dot = 0.0

    elif t < t6:
        # Phase 6: hover
        pass

    elif t < t7:
        # Phase 7: descent
        pz_dot = v_desc
        # vz is constant => vz_dot = 0

    else:
        # Landed: everything zero
        pass

    x_ref_dot = np.array([
        px_dot, py_dot, pz_dot,
        vx_dot, vy_dot, vz_dot,
        phi_dot, theta_dot, psi_dot,
        p_dot, q_dot, r_dot
    ])

    return x_ref_dot


# -----------------------------
# Closed-loop dynamics
# -----------------------------
def make_closed_loop_rhs(A, B, K):
    """
    ẋ = A x + B (u_ref(t) - K (x - x_ref(t)))
    with u_ref chosen so that A x_ref + B u_ref ≈ ẋ_ref.
    """
    def f(t, x):
        x_ref = waypoint_x_ref(t)
        x_ref_dot = waypoint_x_ref_dot(t)

        rhs = x_ref_dot - A @ x_ref
        u_ref, *_ = la.lstsq(B, rhs, rcond=None)  # 4x1 feed-forward

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

    # Start at ground, zero everything
    x0 = np.zeros(12)

    # Simulate from 0 to a bit past t7
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
    
    mask1 = (t >= t1) & (t <= t2)
    avg_speed_seg1 = np.trapezoid(np.sqrt(vx[mask1]**2 + vy[mask1]**2), t[mask1]) / (t2 - t1)
    print(f"Average horizontal speed segment 1: {avg_speed_seg1:.3f} m/s")

    mask2 = (t >= t4) & (t <= t5)
    avg_speed_seg2 = np.trapezoid(np.sqrt(vx[mask2]**2 + vy[mask2]**2), t[mask2]) / (t5 - t4)
    print(f"Average horizontal speed segment 2: {avg_speed_seg2:.3f} m/s")

    # ---------- plots ----------
    plt.figure(figsize=(13, 8))

    # 3D position
    ax1 = plt.subplot(2, 3, 1, projection='3d')
    ax1.plot(px, py, pz, label='Trajectory')

    # --- Add waypoint red dots ---
    waypoints = np.array([
        [0.0, 0.0, 0.0],   # launch point
        [0.0, 0.0, 1.0],   # end of ascent
        [5.0, 0.0, 1.0],   # end of first straight segment
        [5.0, 5.0, 1.0],   # end of second straight segment
        [5.0, 5.0, 0.0],   # landing point
    ])
    ax1.scatter(
        waypoints[:,0],
        waypoints[:,1],
        waypoints[:,2],
        c='red',
        s=40,
        label='Waypoints'
    )
    # -----------------------------

    ax1.set_xlabel('x [m]')
    ax1.set_ylabel('y [m]')
    ax1.set_zlabel('z [m]')
    ax1.set_title('3D Position')
    ax1.grid(True)
    ax1.legend()


    # XY path
    ax2 = plt.subplot(2, 3, 2)
    ax2.plot(px, py, label='Trajectory')
    ax2.set_xlabel('x [m]')
    ax2.set_ylabel('y [m]')
    ax2.set_title('XY Path')
    ax2.grid(True)
    ax2.legend()
    ax2.set_aspect('equal', 'box')

    # Altitude vs time
    ax3 = plt.subplot(2, 3, 3)
    ax3.plot(t, pz, label='z(t)')
    ax3.axhline(1.0, color='r', linestyle='--', alpha=0.5, label='1 m')
    ax3.set_xlabel('Time [s]')
    ax3.set_ylabel('Altitude [m]')
    ax3.set_title('Altitude vs Time')
    ax3.grid(True)
    ax3.legend()

    # Horizontal speed
    ax4 = plt.subplot(2, 3, 4)
    ax4.plot(t, speed_xy, label='|v_xy|')
    ax4.axhline(1.0, color='r', linestyle='--', alpha=0.5, label='1 m/s target')
    ax4.set_xlabel('Time [s]')
    ax4.set_ylabel('Speed [m/s]')
    ax4.set_title('Horizontal Speed')
    ax4.grid(True)
    ax4.legend()

    # Vertical speed
    ax5 = plt.subplot(2, 3, 5)
    ax5.plot(t, vz, label='v_z')
    ax5.axhline(v_desc, color='r', linestyle='--', alpha=0.5,
                label='Landing ref: -0.01 m/s')
    ax5.set_xlabel('Time [s]')
    ax5.set_ylabel('v_z [m/s]')
    ax5.set_title('Vertical Speed')
    ax5.grid(True)
    ax5.legend()

    # Yaw angle
    ax6 = plt.subplot(2, 3, 6)
    ax6.plot(t, np.rad2deg(psi), label='ψ (deg)')
    ax6.set_xlabel('Time [s]')
    ax6.set_ylabel('Yaw [deg]')
    ax6.set_title('Yaw vs Time')
    ax6.grid(True)
    ax6.legend()

    plt.tight_layout()

    # Some stats at end of flight
    print("\n=== Waypoint Mission Stats ===")
    print(f"Final position: x={px[-1]:.3f} m, y={py[-1]:.3f} m, z={pz[-1]:.3f} m")
    print(f"Final vertical speed: v_z={vz[-1]:.5f} m/s "
          "(should be ≈ -0.01 or smaller magnitude)")
    print(f"Max |v_z| during final descent segment: "
          f"{np.max(np.abs(vz[t > t6])):.5f} m/s")

    plt.show()


if __name__ == "__main__":
    main()
