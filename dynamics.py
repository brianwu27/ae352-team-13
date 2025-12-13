# dynamics.py
import numpy as np
from scipy.linalg import solve_continuous_are


class QuadParams:
    def __init__(self):
        self.m   = 0.4581
        self.g   = 9.81
        self.Jx  = 1.3e-3
        self.Jy  = 1.3e-3
        self.Jz  = 2.6e-3
        self.k = 1.2e-6        
        self.k_tau = 8.0e-9
        self.L = 0.114

    def linear_matrices(self):      # Build linearized A and B matrices
        m  = self.m
        g  = self.g
        Jx = self.Jx
        Jy = self.Jy
        Jz = self.Jz

        A = np.zeros((12, 12))

        # position kinematics
        A[0, 3] = 1.0
        A[1, 4] = 1.0
        A[2, 5] = 1.0

        # attitude kinematics
        A[6,  9] = 1.0
        A[7, 10] = 1.0
        A[8, 11] = 1.0

        # small-angle couplings: acceleration from tilt
        A[3, 7] = g
        A[4, 6] = -g

        B = np.zeros((12, 4))

        # vertical acceleration from total thrust deviation
        B[5, 0] = 1/m

        # rotational dynamics
        B[9, 1]  = 1/Jx
        B[10, 2] = 1/Jy
        B[11, 3] = 1/Jz

        return A, B

    def lqr_gain(self):             # Build gain matrix
        A, B = self.linear_matrices()

        # State cost
        Q = np.diag([
            10.0, 10.0, 50.0,    # px, py, pz
             5.0,  5.0, 10.0,    # vx, vy, vz
            50.0, 50.0, 10.0,    # roll, pitch, yaw
             1.0,  1.0,  1.0     # p, q, r
        ])

        # Input cost
        R = np.diag([1.0, 0.1, 0.1, 0.1])

        # Solve continuous-time LQR
        P = solve_continuous_are(A, B, Q, R)
        K = np.linalg.inv(R) @ B.T @ P

        return K
