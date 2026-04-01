import numpy as np


class RandomExcitationController:
    def __init__(self, system, amplitude=0.15, seed=42):
        self.system = system
        self.amplitude = amplitude
        self.rng = np.random.default_rng(seed)

    def compute_input(self, x_current, ref):
        del x_current, ref
        perturbation = self.rng.uniform(-self.amplitude, self.amplitude, size=self.system.m)
        return np.clip(self.system.u_eq + perturbation, self.system.u_lower, self.system.u_upper)
