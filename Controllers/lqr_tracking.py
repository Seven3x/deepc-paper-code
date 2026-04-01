import numpy as np

from Controllers.lqr import LQR


class LQRTrackingController:
    def __init__(self, system, trajectory, noise=0.0, seed=1):
        self.name = "LQR_tracking"
        self.system = system
        self.trajectory = trajectory
        self.lqr = LQR(system, noise=noise, seed=seed)
        self.remaining_output_reference = trajectory.extend_reference(
            trajectory.output_reference,
            1,
        )

    def compute_input(self, x_current):
        ref = self.remaining_output_reference[:, 0]
        u = self.lqr.compute_input(x_current, ref)
        self.remaining_output_reference = self.trajectory.extend_reference(
            self.remaining_output_reference[:, 1:],
            1,
        )
        return np.asarray(u).reshape(-1)
