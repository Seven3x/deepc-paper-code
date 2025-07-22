import numpy as np
import control

class LQR:
    def __init__(self, system, noise = False):
        self.name = f"LQR{'_noise' if noise else ''}"
        self.system = system
        self.noise = noise

        Q = np.eye(system.n)
        R = np.eye(system.m)

        self.L, P, E = control.dlqr(system.Ad, system.Bd, Q, R)

    def compute_input(self, x):
        u = -self.L @ x + self.L @ self.system.x_r + self.system.u_eq + self.noise*(np.random.randn(self.system.m))
        return u