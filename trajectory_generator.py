import numpy as np
import matplotlib.pyplot as plt

class TrajectoryGenerator:
    def __init__(self, sort, system, duration, has_initial_ref=False):
        self.system = system
        self.sort = sort
        self.has_initial_ref = has_initial_ref

        if sort == 'constant':
            self.output_reference = system.constant_ref.reshape(-1,1)
        elif sort == 'figure8':
            self.output_reference = self.generate_figure_eight_reference(duration=duration)
        elif sort == 'step':
            self.output_reference = self.generate_step_reference(duration=duration)
        elif sort == 'box':
            self.output_reference = self.generate_box_reference(duration=duration)
        else:
            raise ValueError(f"Reference type: {sort} is invalid. Please choose a valid reference")
        

    def generate_figure_eight_reference(self, a=1, max_frequency=0.2, start_time = 1.0, duration=60):
        """
        Generate a figure eight reference trajectory.
        
        Arguments:
            a (float): radius.
            frequency (float): Frequency of the x sinusoid in Hz.
            duration (float): Total duration of the trajectory in seconds.
        
        Returns:
            numpy.ndarray: Figure eight trajectory of shape (p, num_samples).
        """
        Ts = self.system.h  # Sampling time
        t = np.arange(0, duration, Ts)  #Time vector
        ref_indieces = t >= start_time
        t_ref = t[ref_indieces]-start_time
        ref = np.zeros((self.system.p, len(t)))
        frequency = np.zeros_like(t_ref)
        positive_mask = t_ref > 0
        frequency[positive_mask] = max_frequency * np.exp(-3 / t_ref[positive_mask])  # Exponential increase of frequency
        z_step = -1
        ref[:,ref_indieces] = np.vstack((np.zeros((self.system.p-3,len(t_ref))),
                        np.array([a*np.sin(2 * np.pi * frequency * t_ref),
                        a/3*np.sin(4 * np.pi * frequency * t_ref), z_step*np.ones(len(t_ref))])))
        
        return ref
    
    def generate_step_reference(
        self,
        start_value=[-0.5, -0.5, 0],
        end_value=[0.5, 0.5, -1],
        step_time=3.0,
        duration=20.0,
        ramp_duration=5.0,
    ):
        """
        Generate a step reference trajectory.

        Arguments:
            start_value (float): Value before the step.
            end_value (float): Value after the step.
            step_time (float): Time at which the step occurs.
            duration (float): Total duration of the trajectory.

        Returns:
            numpy.ndarray: Step trajectory of shape (p, num_samples).
        """
        Ts = self.system.h
        t = np.arange(0, duration, Ts)

        ref = np.zeros((self.system.p, len(t)))

        start = np.asarray(start_value, dtype=float).reshape(3, 1)
        end = np.asarray(end_value, dtype=float).reshape(3, 1)
        if ramp_duration <= 0:
            alpha = (t >= step_time).astype(float)
        else:
            alpha = np.clip((t - step_time) / ramp_duration, 0.0, 1.0)
            # Use a half-cosine easing to avoid an impulsive reference jump that
            # can destabilize the nominal step benchmark.
            alpha = 0.5 - 0.5 * np.cos(np.pi * alpha)
        xyz = start @ np.ones((1, len(t))) * (1.0 - alpha) + end @ np.ones((1, len(t))) * alpha

        ref[-3:, :] = xyz

        return ref
    
    def generate_box_reference(self, duration = 20.0):
        """
        Generate a box reference trajectory.

        Arguments:
            duration (float): Total duration of the trajectory.

        Returns:
            numpy.ndarray: Box trajectory of shape (p, num_samples).
        """
        corners = [np.array([0, 0, 0]),
                   np.array([1, 1, 0]),
                   np.array([1, 1, -1]),
                   np.array([1, -1, -1]),
                   np.array([-1, -1, -1]),
                   np.array([-1, -1, 0]),
                   np.array([0, 0, 0])]
    
        Ts = self.system.h
        t = np.arange(0, duration, Ts)
        step_time_interval = duration / len(corners)

        ref = np.zeros((self.system.p, len(t)))

        for i, corner in enumerate(corners):
            mask = (t >= step_time_interval * i) & (t < step_time_interval * (i + 1))
            ref[-3:, mask] = corner.reshape(-1,1)

        return ref

    def generate_box_sweep_reference(self, duration=20.0):
        """
        Generate a smooth box sweep reference that linearly interpolates
        between corners instead of jumping across them.
        """
        corners = [
            np.array([0.0, 0.0, 0.0]),
            np.array([1.0, 1.0, 0.0]),
            np.array([1.0, 1.0, -1.0]),
            np.array([1.0, -1.0, -1.0]),
            np.array([-1.0, -1.0, -1.0]),
            np.array([-1.0, -1.0, 0.0]),
            np.array([0.0, 0.0, 0.0]),
        ]

        Ts = self.system.h
        t = np.arange(0, duration, Ts)
        ref = np.zeros((self.system.p, len(t)))
        if len(t) == 0:
            return ref

        segment_count = len(corners) - 1
        segment_duration = duration / max(segment_count, 1)

        for seg in range(segment_count):
            start_t = seg * segment_duration
            end_t = (seg + 1) * segment_duration
            if seg == segment_count - 1:
                mask = (t >= start_t) & (t <= end_t)
            else:
                mask = (t >= start_t) & (t < end_t)
            if not np.any(mask):
                continue
            local_alpha = (t[mask] - start_t) / max(segment_duration, Ts)
            local_alpha = np.clip(local_alpha, 0.0, 1.0)
            start_corner = corners[seg].reshape(3, 1)
            end_corner = corners[seg + 1].reshape(3, 1)
            interp = start_corner @ np.ones((1, local_alpha.size)) * (1.0 - local_alpha)
            interp += end_corner @ np.ones((1, local_alpha.size)) * local_alpha
            ref[-3:, mask] = interp

        return ref

    def initial_reference(self, length):
        '''
        Generate an initial reference trajectory to help excite the system.
        
        Arguments:
            length (int): Length of the reference trajectory.

        Returns:
            numpy.ndarray: Initial reference trajectory of shape (p, length).
        '''

        if self.sort == 'box':
            return self.generate_box_sweep_reference(duration=self.system.h * length)

        ref = self.extend_reference(self.output_reference[:,0].reshape(-1,1), length)
        time = np.linspace(0, self.system.h * length, length)

        if getattr(self.system, "output_set", "xyzpsi") == "xyz":
            ref[0, :] = 0.15 * np.sign(np.sin(2 * np.pi * 0.13 * time))
            ref[1, :] = 0.12 * np.sign(np.sin(2 * np.pi * 0.17 * time + np.pi / 3))
            ref[2, :] = 0.10 * np.sign(np.sin(2 * np.pi * 0.21 * time))
        else:
            yaw_ref = 0.10 * np.sin(2 * np.pi * 0.11 * time)
            x_ref = 0.15 * np.sign(np.sin(2 * np.pi * 0.13 * time))
            y_ref = 0.12 * np.sign(np.sin(2 * np.pi * 0.17 * time + np.pi / 3))
            z_ref = 0.10 * np.sign(np.sin(2 * np.pi * 0.21 * time))
            ref[2, :] = yaw_ref
            ref[3, :] = x_ref
            ref[4, :] = y_ref
            ref[5, :] = z_ref

        return ref
    
    def extend_reference(self, ref, N):
        """
        Extend matrix ref to have exactly N columns.
        If ref has fewer than self.N columns, repeat the last column to fill up.
        If ref has N or more columns, return ref as is.
        
        Arguments:
            ref (numpy.ndarray): Input matrix of shape (self.p, -).
            N (int): Desired number of columns.
        
        Returns:
            ref_extended (numpy.ndarray): Matrix of shape (m, N).
        """
        m, n = ref.shape  # Get current shape of A

        if n == 0:
            return np.tile(self.output_reference[:, -1].reshape(m, 1), (1, N))
        
        if n >= N:
            return ref  # If A already has N or more columns, return as is
        
        # Repeat the last column to reach N columns
        last_col = ref[:, -1].reshape(m, 1)  # Extract last column as (m, 1)
        num_extra_cols = N - n  # How many more columns we need
        
        # Tile the last column to match the required number of extra columns
        extra_cols = np.tile(last_col, (1, num_extra_cols))
        
        # Concatenate A with the repeated last column
        ref_extended = np.hstack((ref, extra_cols))
        
        return ref_extended
    
# Short script to test and plot sinusoidal reference
if __name__ == "__main__":
    class DummySystem:
        def __init__(self, h):
            self.h = h
    
    system = DummySystem(h=0.1)
    traj_gen = TrajectoryGenerator(sort='figure8', system=system, duration=10)
