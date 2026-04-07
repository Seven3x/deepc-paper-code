import numpy as np


class ReferenceProbeExcitationController:
    def __init__(
        self,
        system,
        base_controller,
        sampling_time,
        position_amplitude=0.12,
        z_amplitude=0.08,
        yaw_amplitude=0.20,
        seed=42,
    ):
        self.system = system
        self.base_controller = base_controller
        self.sampling_time = float(sampling_time)
        self.position_amplitude = float(position_amplitude)
        self.z_amplitude = float(z_amplitude)
        self.yaw_amplitude = float(yaw_amplitude)
        self.step = 0
        self.rng = np.random.default_rng(seed)
        self.phase = self.rng.uniform(0.0, 2.0 * np.pi, size=4)

    def _probe(self, ref):
        ref_aug = np.asarray(ref, dtype=float).reshape(-1).copy()
        t = self.step * self.sampling_time

        # Output set xyzpsi follows [roll, pitch, yaw, x, y, z].
        if ref_aug.size >= 6:
            ref_aug[2] += self.yaw_amplitude * np.sin(2.0 * np.pi * 0.09 * t + self.phase[0])
            ref_aug[3] += self.position_amplitude * np.sin(2.0 * np.pi * 0.07 * t + self.phase[1])
            ref_aug[4] += self.position_amplitude * np.sin(2.0 * np.pi * 0.11 * t + self.phase[2])
            ref_aug[5] += self.z_amplitude * np.sign(np.sin(2.0 * np.pi * 0.13 * t + self.phase[3]))
        elif ref_aug.size >= 3:
            ref_aug[0] += self.position_amplitude * np.sin(2.0 * np.pi * 0.07 * t + self.phase[1])
            ref_aug[1] += self.position_amplitude * np.sin(2.0 * np.pi * 0.11 * t + self.phase[2])
            ref_aug[2] += self.z_amplitude * np.sign(np.sin(2.0 * np.pi * 0.13 * t + self.phase[3]))

        return ref_aug

    def compute_input(self, x_current, ref):
        ref_aug = self._probe(ref)
        self.step += 1
        return np.asarray(self.base_controller.compute_input(x_current, ref_aug), dtype=float).reshape(-1)
