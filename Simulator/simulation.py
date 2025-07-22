import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
from scipy.integrate import quad_vec, quad

class Simulation:
    """
    A simulation class that integrates a given system's dynamics and applies a controller.
    """
    
    def __init__(self, system, controller, dt=0.01, t_final=10):
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

        epsilon = 1e-10 # Small value to avoid numerical errors

        for t in time[:-1]:
            
            if t >= h*k - epsilon: # ZOH sampling
                
                y = self.system.measure_output(x)  # Measure system output
                u = self.compute_input(x)        # Compute control input
                u = self.constrain_input(u)

                # u += 2*np.random.randn()

                u_data.append(u)
                y_data.append(y)
                k += 1

                print(f"input calculated at time: {t}")
                print(f"y: {y}")
                print(f"u: {u}")

            x = self.rk4(x, u)                 # Integrate system dynamics
            # Store data
            x_data.append(x)

        return {
        "time": time,
        "discrete time": discrete_time,
        "x": np.array(x_data).T,
        "u": np.array(u_data).T,
        "y": np.array(y_data).T
        }

    def compute_input(self, x): 
        return self.controller.compute_input(x)

    def constrain_input(self, x):
        return np.clip(x, self.system.u_lower, self.system.u_upper)
    
class SimulationPlotter:
    def __init__(self, system):
        # Initialize with time, states (x_data), control inputs (u_data), and outputs (y_data)
        self.name = system.name
        self.labels = system.labels

    def plot(self, result, ref, start_plotting_from=0):
        """Plot the simulation data."""
        # Extracting the data from result dictionary
        self.time = result["time"]
        self.discrete_time = result["discrete time"][start_plotting_from:]
        self.x_data = result["x"]
        self.u_data = result["u"][:,start_plotting_from:]
        self.y_data = result["y"][:,start_plotting_from:]

        # Number of states, inputs, and outputs
        n = self.x_data.shape[0]
        m = self.u_data.shape[0]
        p = self.y_data.shape[0]

        total_discrete_data_points = result["y"].shape[1]

        self.ref = ref

        if ref is not None:
            # Extend the reference trajectory to the total number of discrete data points
            self.ref = ref.extend_reference(ref.output_reference, total_discrete_data_points)
            self.ref = self.ref[:,start_plotting_from:]

        # Plotting states (x_data) over time
        self._plot_subplots(self.x_data, n, self.labels["x"], 'State Evolution', self.time, 'blue')

        # Plotting control input (u_data) over time
        self._plot_subplots(self.u_data, m, self.labels["u"], 'Control Input Evolution', self.discrete_time, 'red', stairs=True)

        # Plotting output (y_data) over time
        self._plot_subplots(self.y_data, p, self.labels["y"], 'Output Evolution', self.discrete_time, 'green')

    def _plot_subplots(self, data, num_plots, labels, title, x_data, color, stairs=False):
        """Helper function to plot subplots in a grid."""
        rows = int(np.floor(np.sqrt(num_plots)))  # Number of rows
        cols = int(np.ceil(num_plots / rows))     # Number of columns

        plt.figure(figsize=(10, 8))  # Set a figure size

        for i in range(num_plots):
            plt.subplot(rows, cols, i + 1)
            if stairs:
                plt.step(x_data, data[i, :], color=color)
            else:
                plt.plot(x_data, data[i, :], color=color)
                if title == 'Output Evolution' and self.ref is not None:
                    self.ref
                    plt.plot(x_data, self.ref[i,:], color = 'black', label='Reference')
            plt.ylabel(labels[i])
            plt.xlabel('Time [s]')
            plt.legend()
            plt.grid(True)  # Add grid to each subplot

        plt.suptitle(title)
        plt.tight_layout()  # Adjust layout to avoid overlapping elements
        plt.subplots_adjust(top=0.9)  # Adjust the top margin for the suptitle
        
        plt.savefig(f"masterthesis/Python/Figures/{self.name}_{title}.png")
