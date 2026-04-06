import numpy as np


class AdditiveExcitationController:
    def __init__(self, system, base_controller, excitation_controller):
        self.system = system
        self.base_controller = base_controller
        self.excitation_controller = excitation_controller

    def compute_input(self, x_current, ref):
        base_u = np.asarray(self.base_controller.compute_input(x_current, ref), dtype=float).reshape(-1)
        excitation_u = np.asarray(self.excitation_controller.compute_input(x_current, ref), dtype=float).reshape(-1)
        perturbation = excitation_u - np.asarray(self.system.u_eq, dtype=float).reshape(-1)
        return np.clip(base_u + perturbation, self.system.u_lower, self.system.u_upper)
