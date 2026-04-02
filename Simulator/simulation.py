import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
from scipy.integrate import quad_vec, quad

from paths import PLOTS_DIR, ensure_output_dirs

class Simulation:
    """
    A simulation class that integrates a given system's dynamics and applies a controller.
    """
    
    def __init__(self, system, controller, dt=0.001, t_final=10, verbose=True):
        """
        Initializes the simulation.
        
        Arguments:
        - system: A system object with a dynamics function f(x, u) and an output function h(x).
        - controller: A controller object with a method calculate_input(x).
        - dt: Time step for integration.
        - t_final: Total simulation time.
        """
        self.system = system
        self.controller = controller
        self.dt = dt
        self.t_final = t_final
        self.verbose = verbose
        self.measurement_delay_steps = int(getattr(system, "measurement_config", {}).get("delay_steps", 0))
        measurement_config = getattr(system, "measurement_config", {})
        self.async_period_steps = np.asarray(
            measurement_config.get("async_period_steps", np.ones(system.p, dtype=int)),
            dtype=int,
        ).reshape(system.p)
        self.dropout_rate = float(measurement_config.get("burst_dropout_rate", 0.0))
        self.dropout_burst_length = int(measurement_config.get("burst_dropout_length", 0))
        self.measurement_rng = np.random.default_rng(int(measurement_config.get("seed", 0)) + 20260402)
        self.dropout_remaining = np.zeros(system.p, dtype=int)
        self.last_delivered_output = None
        self.last_delivered_true_output = None
        self.last_delivered_source_steps = None

    def rk4(self, x, u):
        """
        Implements the Runge-Kutta 4th order (RK4) method for integrating system dynamics.
        
        Arguments:
        - x: Current state.
        - u: Control input.

        Returns:
        - x_next: The next state after applying RK4 integration.
        """

        dt = self.dt
        f = self.system.dynamics  # System's state transition function

        k1 = f(x, u)
        k2 = f(x + 0.5 * dt * k1, u)
        k3 = f(x + 0.5 * dt * k2, u)
        k4 = f(x + dt * k3, u)

        x_next = x + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

        return x_next

    def simulate(self):
        """
        Runs the simulation using the given system and controller.

        Returns:
        - time: Array of time steps.
        - x_data: Array of state trajectories.
        - u_data: Array of control inputs.
        - y_data: Array of measured outputs.
        """

        if self.verbose:
            print("Simulating...")
        
        time_steps = int(self.t_final / self.dt)
        time = np.linspace(0, self.t_final, time_steps + 1)
        h = self.system.h
        discrete_time = np.arange(0, self.t_final, h)
        k = 0

        x = self.system.x0 # Initial state
        x_data = [x]
        u_data = []
        y_data = []
        y_true_data = []
        measurement_source_steps = []
        measurement_source_steps_by_output = []
        measurement_delivered_steps = []
        measurement_delay_steps = []
        measurement_output_masks = []
        measurement_output_timestamps = []
        measurement_buffer = []

        epsilon = 1e-10 # Small value to avoid numerical errors

        for t in time[:-1]:
            
            if t >= h*k - epsilon: # ZOH sampling
                
                y = self.system.measure_output(x)  # Measure system output
                y_true = self.system.C @ x
                measurement_packet = self._build_measurement_packet(
                    measured_output=y,
                    true_output=y_true,
                    sample_step=k,
                    measurement_buffer=measurement_buffer,
                )
                u = self.compute_input(x, measurement_packet)        # Compute control input from the same sample
                u = self.constrain_input(u)

                # u += 2*np.random.randn()

                u_data.append(u)
                y_data.append(measurement_packet["output"])
                y_true_data.append(y_true)
                measurement_source_steps.append(measurement_packet["source_step"])
                measurement_source_steps_by_output.append(measurement_packet["output_source_steps"])
                measurement_delivered_steps.append(measurement_packet["delivered_step"])
                measurement_delay_steps.append(measurement_packet["delay_steps"])
                measurement_output_masks.append(measurement_packet["output_mask"])
                measurement_output_timestamps.append(measurement_packet["output_timestamps"])
                k += 1

                if self.verbose:
                    print(
                        f"Time: {t:.2f}s, Input: {u}, Output: {measurement_packet['output']}, "
                        f"MeasurementSourceStep: {measurement_packet['source_step']}"
                    )

            x = self.rk4(x, u)                 # Integrate system dynamics
            # Store data
            x_data.append(x)

        return {
        "time": time,
        "discrete time": discrete_time,
        "x": np.array(x_data).T,
        "u": np.array(u_data).T,
        "y": np.array(y_data).T,
        "y_true": np.array(y_true_data).T,
        "measurement_source_step": np.array(measurement_source_steps, dtype=int),
        "measurement_source_step_by_output": np.array(measurement_source_steps_by_output, dtype=int).T,
        "measurement_delivered_step": np.array(measurement_delivered_steps, dtype=int),
        "measurement_delay_steps": np.array(measurement_delay_steps, dtype=int),
        "measurement_output_mask": np.array(measurement_output_masks, dtype=float).T,
        "measurement_output_timestamps": np.array(measurement_output_timestamps, dtype=float).T,
        }

    def compute_input(self, x, y=None): 
        return self.controller.compute_input(x, y)

    def constrain_input(self, x):
        return np.clip(x, self.system.u_lower, self.system.u_upper)

    def _build_measurement_packet(self, measured_output, true_output, sample_step, measurement_buffer):
        measurement_buffer.append(
            {
                "output": np.asarray(measured_output, dtype=float).copy(),
                "true_output": np.asarray(true_output, dtype=float).copy(),
                "source_step": int(sample_step),
            }
        )

        delay_steps = self.measurement_delay_steps
        if delay_steps <= 0:
            source_entry = measurement_buffer[-1]
        else:
            source_index = max(0, len(measurement_buffer) - 1 - delay_steps)
            source_entry = measurement_buffer[source_index]

        target_source_step = max(int(sample_step) - int(delay_steps), 0)

        delivered_output = source_entry["output"].copy()
        delivered_true_output = source_entry["true_output"].copy()
        output_source_steps = np.full(self.system.p, int(source_entry["source_step"]), dtype=int)
        output_mask = self._sample_output_mask(int(sample_step))

        # Force a fully observed bootstrap packet so the hold-last logic starts from a valid state.
        if self.last_delivered_output is None:
            output_mask[:] = 1.0
        else:
            missing = output_mask < 0.5
            delivered_output[missing] = self.last_delivered_output[missing]
            delivered_true_output[missing] = self.last_delivered_true_output[missing]
            output_source_steps[missing] = self.last_delivered_source_steps[missing]

        self.last_delivered_output = delivered_output.copy()
        self.last_delivered_true_output = delivered_true_output.copy()
        self.last_delivered_source_steps = output_source_steps.copy()

        return {
            "output": delivered_output,
            "true_output": delivered_true_output,
            "source_step": int(np.max(output_source_steps)),
            "target_source_step": int(target_source_step),
            "delivered_step": int(sample_step),
            "delay_steps": int(delay_steps),
            "output_mask": output_mask.astype(float),
            "output_source_steps": output_source_steps.astype(int),
            "output_timestamps": output_source_steps.astype(float) * self.system.h,
        }

    def _sample_output_mask(self, sample_step):
        async_ready = (sample_step % self.async_period_steps) == 0
        dropout_ready = np.ones(self.system.p, dtype=bool)

        if self.dropout_burst_length > 0 and self.dropout_rate > 0.0:
            active_dropout = self.dropout_remaining > 0
            dropout_ready[active_dropout] = False
            self.dropout_remaining[active_dropout] -= 1

            start_probability = min(self.dropout_rate / max(self.dropout_burst_length, 1), 1.0)
            available_to_start = ~active_dropout
            burst_starts = self.measurement_rng.random(self.system.p) < start_probability
            burst_starts &= available_to_start
            dropout_ready[burst_starts] = False
            self.dropout_remaining[burst_starts] = self.dropout_burst_length - 1

        return np.logical_and(async_ready, dropout_ready).astype(float)
    
class SimulationPlotter:
    def __init__(self, system):
        # Initialize with time, states (x_data), control inputs (u_data), and outputs (y_data)
        self.name = system.name
        self.labels = system.labels

    def plot(self, result, ref):
        """Plot the simulation data."""
        # Extracting the data from result dictionary
        self.time = result["time"]
        self.discrete_time = result["discrete time"]
        self.x_data = result["x"]
        self.u_data = result["u"]
        self.y_data = result["y"]

        # Number of states, inputs, and outputs
        n = self.x_data.shape[0]
        m = self.u_data.shape[0]
        p = self.y_data.shape[0]

        total_discrete_data_points = result["y"].shape[1]

        self.traj = None

        if ref is not None:
            # Extend the reference trajectory to the total number of discrete data points
            self.traj = ref.extend_reference(ref.output_reference, total_discrete_data_points)
            self.traj = self.traj[:,:total_discrete_data_points]

        # Plotting states (x_data) over time
        self._plot_subplots(self.x_data, n, self.labels["x"], 'State Evolution', self.time, 'blue')

        # Plotting control input (u_data) over time
        self._plot_subplots(self.u_data, m, self.labels["u"], 'Control Input Evolution', self.discrete_time, 'red', stairs=True)

        # Plotting output (y_data) over time
        self._plot_subplots(self.y_data, p, self.labels["y"], 'Output Evolution', self.discrete_time, 'green')

    def _plot_subplots(self, data, num_plots, labels, title, x_data, color, stairs=False):
        """Helper function to plot subplots in a grid."""
        ensure_output_dirs()
        rows = int(np.floor(np.sqrt(num_plots)))  # Number of rows
        cols = int(np.ceil(num_plots / rows))     # Number of columns

        plt.figure(figsize=(10, 8))  # Set a figure size

        for i in range(num_plots):
            plt.subplot(rows, cols, i + 1)
            if stairs:
                plt.step(x_data, data[i, :], color=color)
            else:
                plt.plot(x_data, data[i, :], color=color)
                if title == 'Output Evolution' and self.traj is not None:
                    plt.plot(x_data, self.traj[i,:], color = 'black', label='Reference')
                    plt.legend()
            plt.ylabel(labels[i])
            plt.xlabel('Time [s]')
            plt.grid(True)  # Add grid to each subplot

        plt.suptitle(title)
        plt.tight_layout()  # Adjust layout to avoid overlapping elements
        plt.subplots_adjust(top=0.9)  # Adjust the top margin for the suptitle
        
        plt.savefig(PLOTS_DIR / f"{self.name}_{title}.png")
