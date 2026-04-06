import numpy as np


class PRBSExcitationController:
    def __init__(self, system, amplitude=0.15, hold_steps=5, seed=42):
        self.system = system
        self.amplitude = float(amplitude)
        self.hold_steps = max(int(hold_steps), 1)
        self.rng = np.random.default_rng(seed)
        self.current_step = 0
        self.current_perturbation = np.zeros(self.system.m)

    def _sample_perturbation(self):
        signs = self.rng.choice([-1.0, 1.0], size=self.system.m)
        return self.amplitude * signs

    def compute_input(self, x_current, ref):
        del x_current, ref
        if self.current_step % self.hold_steps == 0:
            self.current_perturbation = self._sample_perturbation()
        self.current_step += 1
        return np.clip(self.system.u_eq + self.current_perturbation, self.system.u_lower, self.system.u_upper)
