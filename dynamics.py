# dynamics.py
import numpy as np
from scipy.linalg import solve_continuous_are


class QuadParams:
    def __init__(self):
        # From your report, Table 2
        self.m   = 0.4581      # kg
        self.g   = 9.81        # m/s^2
        self.Jx  = 1.3e-3      # kg m^2
        self.Jy  = 1.3e-3      # kg m^2
        self.Jz  = 2.6e-3      # kg m^2

    def linear_matrices(self):
        """
        Build the small-angle linearized A, B matrices for state:
        x = [px, py, pz, vx, vy, vz, φ, θ, ψ, p, q, r]^T
        and inputs:
        u = [ΔT_total, τφ, τθ, τψ]^T
        """
        m  = self.m
        g  = self.g
        Jx = self.Jx
        Jy = self.Jy
        Jz = self.Jz

        A = np.zeros((12, 12))

        # position kinematics
        A[0, 3] = 1.0   # pẋ = vx
        A[1, 4] = 1.0   # pẏ = vy
        A[2, 5] = 1.0   # pż = vz

        # attitude kinematics
        A[6,  9] = 1.0  # φ̇ = p
        A[7, 10] = 1.0  # θ̇ = q
        A[8, 11] = 1.0  # ψ̇ = r

        # small-angle couplings: acceleration from tilt
        A[3, 7] = g      # vẋ ≈ g * θ
        A[4, 6] = -g     # vẏ ≈ -g * φ

        B = np.zeros((12, 4))

        # vertical acceleration from total thrust deviation
        B[5, 0] = 1.0 / m        # vż = (1/m) ΔT_total

        # rotational dynamics
        B[9, 1]  = 1.0 / Jx      # ṗ   = (1/Jx) τφ
        B[10, 2] = 1.0 / Jy      # q̇   = (1/Jy) τθ
        B[11, 3] = 1.0 / Jz      # ṙ   = (1/Jz) τψ

        return A, B

    def lqr_gain(self):
        """
        Design an LQR gain K for hover regulation.
        Returns:
            K: (4x12) numpy array such that Δu = -K (x - x_ref).
        """
        A, B = self.linear_matrices()

        # State cost: emphasize position, altitude, and attitude
        Q = np.diag([
            10.0, 10.0, 50.0,    # px, py, pz
             5.0,  5.0, 10.0,    # vx, vy, vz
            50.0, 50.0, 10.0,    # φ, θ, ψ
             1.0,  1.0,  1.0     # p, q, r
        ])

        # Input cost: keep torques "cheap", thrust a bit more expensive
        R = np.diag([1.0, 0.1, 0.1, 0.1])

        # Solve continuous-time LQR
        P = solve_continuous_are(A, B, Q, R)
        K = np.linalg.inv(R) @ B.T @ P

        return K
