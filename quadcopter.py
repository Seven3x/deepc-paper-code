import numpy as np
import sympy as sp
from scipy.linalg import expm
from scipy.integrate import quad_vec, quad
import control

class Quadcopter:
    def __init__(self, h, measurement_config=None, output_set="xyzpsi"):
        self.n = 12
        self.m = 4
        self.h = h #ZOH sampling time
        self.x0 = np.array([0,0,0,
                            0,0,0,
                            0,0,0,
                            0,0,0]) # Initial state vector

        self.output_set = output_set
        self.output_indices = self._build_output_indices(output_set)
        self.p = len(self.output_indices)
        self.C = np.zeros((len(self.output_indices), self.n))
        for i, idx in enumerate(self.output_indices):
            self.C[i, idx] = 1.0

        self.name = f"Quadcopter_h={self.h}"
        
        self.labels = {
            "x": [r"$\Phi$ [rad]", r"$\Theta$ [rad]", r"$\Psi$ [rad]",
                  "$p$ [rad/s]", "$q$ [rad/s]", "$r$ [rad/s]",
                  "$u$ [m/s]", "$v$ [m/s]", "$w$ [m/s]",
                  "$x_{B_0}$ [m]", "$y_{B_0}$ [m]", "$z_{B_0}$ [m]"],
            "u": ["$u_1$", "$u_2$", "$u_3$", "$u_4$"],
        }
        self.labels["y"] = [self.labels["x"][i] for i in self.output_indices]

        # Properties
        self.mass = 0.5 # [kg]
        self.Inertia = np.diag([0.006, 0.006, 0.0011])
        self.gravity = 9.82 #[m/s^2]
        self.r = 0.1 # Distance from the center of the quadcopter to a propeller

        # Aerodynamic properties
        self.b = 1.e-05  # Thrust coefficient [N/(rad/s)^2]
        self.d = 3.5e-8  # Drag coefficient [Nm/(rad/s)^2]

        self.act_mat = self.actuator_matrix(self.b, self.r, self.d)

        # Stationary operating point
        self.x_eq= np.array([0,0,0,
                             0,0,0,
                             0,0,0,
                             0,0,0])

        self.constant_ref = self.x_eq[self.output_indices]
        self.Mx = np.zeros((self.n, self.p))
        for i, idx in enumerate(self.output_indices):
            self.Mx[idx, i] = 1.0
        self.x_r = self.Mx @ self.constant_ref

        self.max_thrust = self.gravity*self.mass/2
        self.u_eq = self.gravity*self.mass/4/self.max_thrust*np.ones(self.m)

        # Input and output constraints
        self.x_lower = np.array([-np.pi/2, -np.pi/2, -100, -100, -100, -100, -100, -100, -100, -10, -10, -10])
        self.x_upper = np.array([np.pi/2, np.pi/2, 100, 100, 100, 100, 100, 100, 100, 10, 10, 10])
        self.y_lower = np.array([self.x_lower[i] for i in self.output_indices])
        self.y_upper = np.array([self.x_upper[i] for i in self.output_indices])
        self.u_lower = 0.0*np.ones(self.m)
        self.u_upper = np.ones(self.m)

        self.F = np.vstack((np.eye(self.p),-np.eye(self.p)))
        self.f = np.hstack((self.y_upper, -self.y_lower))

        self.G = np.vstack((np.eye(self.m), -np.eye(self.m)))
        self.g = np.hstack((self.u_upper, -self.u_lower))

        # np.cost parameters for predictive control
        self.Q = np.diag([self._output_cost(idx) for idx in self.output_indices])
        self.R = 1*np.eye(self.m)

        self.measurement_config = self._normalize_measurement_config(measurement_config)
        self._measurement_rng = np.random.default_rng(self.measurement_config["seed"])
        self._measurement_step = 0
        self._current_yaw_bias = self.measurement_config["yaw_bias"]

        # Linearize dynamics around the stationary point
        self.linearize()

    def dynamics(self, x, u_input, symbolic=False):
        ''' 
        Nonlinear continuous-time dynamics of the quadcopter.
        
        Arguments:
            x (12x1 np.ndarray): State vector
            u (4x1 np.ndarray): Normalized Input vector in range [0,1] 
        
        Returns:
            xdot (12x1 np.ndarray): Time derivative of the state vector
        '''

        u_input = u_input * self.max_thrust # Scale input from 0-1 to thrust

        Phi, Theta, Psi, p, q, r, u, v, w, _, _, _ = x
        Ix, Iy, Iz = np.diag(self.Inertia)
        ft, Tx, Ty, Tz = self.act_mat @ u_input
        g = self.gravity
        m = self.mass

        if symbolic:
            xdot = np.array([
                p + q * sp.tan(Theta) * sp.sin(Phi) - r * sp.cos(Phi) * sp.sin(Theta),
                q * sp.cos(Phi) + r * sp.sin(Phi),
                -q * sp.sin(Phi) / sp.cos(Theta) + r * sp.cos(Phi) / sp.cos(Theta),
                q * r * (Iy - Iz) / Ix + Tx / Ix,
                p * r * (Iz - Ix) / Iy + Ty / Iy,
                p * q * (Ix - Iy) / Iz + Tz / Iz,
                r * v - q * w - g * sp.sin(Theta),
                p * w - r * u + g * sp.sin(Phi) * sp.cos(Theta),
                q * u - p * v + g * sp.cos(Theta) * sp.cos(Phi) - ft / m,
                u * (sp.cos(Theta) * sp.cos(Psi)) + v * (sp.cos(Psi) * sp.sin(Phi) * sp.sin(Theta) - sp.cos(Phi) * sp.sin(Psi)) + w * (sp.cos(Phi) * sp.cos(Psi) * sp.sin(Theta) + sp.sin(Phi) * sp.sin(Psi)),
                u * (sp.cos(Theta) * sp.sin(Psi)) + v * (sp.sin(Phi) * sp.sin(Psi) * sp.sin(Theta) + sp.cos(Phi) * sp.cos(Psi)) + w * (sp.cos(Phi) * sp.sin(Psi) * sp.sin(Theta) - sp.cos(Psi) * sp.sin(Phi)),
                -u * sp.sin(Theta) + v * sp.cos(Theta) * sp.sin(Phi) + w * sp.cos(Phi) * sp.cos(Theta)
            ])

        else:
            xdot = np.array([
                p + q * np.tan(Theta) * np.sin(Phi) - r * np.cos(Phi) * np.sin(Theta),
                q * np.cos(Phi) + r * np.sin(Phi),
                -q * np.sin(Phi) / np.cos(Theta) + r * np.cos(Phi) / np.cos(Theta),
                q * r * (Iy - Iz) / Ix + Tx / Ix,
                p * r * (Iz - Ix) / Iy + Ty / Iy,
                p * q * (Ix - Iy) / Iz + Tz / Iz,
                r * v - q * w - g * np.sin(Theta),
                p * w - r * u + g * np.sin(Phi) * np.cos(Theta),
                q * u - p * v + g * np.cos(Theta) * np.cos(Phi) - ft / m,
                u * (np.cos(Theta) * np.cos(Psi)) + v * (np.cos(Psi) * np.sin(Phi) * np.sin(Theta) - np.cos(Phi) * np.sin(Psi)) + w * (np.cos(Phi) * np.cos(Psi) * np.sin(Theta) + np.sin(Phi) * np.sin(Psi)),
                u * (np.cos(Theta) * np.sin(Psi)) + v * (np.sin(Phi) * np.sin(Psi) * np.sin(Theta) + np.cos(Phi) * np.cos(Psi)) + w * (np.cos(Phi) * np.sin(Psi) * np.sin(Theta) - np.cos(Psi) * np.sin(Phi)),
                -u * np.sin(Theta) + v * np.cos(Theta) * np.sin(Phi) + w * np.cos(Phi) * np.cos(Theta)
            ])

        return xdot

    def _build_output_indices(self, output_set):
        if output_set == "xyzpsi":
            return [0, 1, 2, 9, 10, 11]
        if output_set == "xyz":
            return [9, 10, 11]
        raise ValueError(f"Unsupported output_set: {output_set}")

    def _output_cost(self, state_index):
        if state_index == 9:
            return 200.0
        if state_index == 10:
            return 200.0
        if state_index == 11:
            return 100.0
        return 1.0
        
    def discrete_dynamics(self, x, u, use_casadi=False): #Nonlinear discrete dynamics
        x_next = x + self.h * self.dynamics(x, u, use_casadi=use_casadi)
        return x_next

    def linear_dynamics(self, x, u):
        xdot = self.A @ (x - self.x_eq) + self.B @ (u - self.u_eq)
        return xdot

    def measure_output(self, x):
        y = self.C @ x

        noise_std = self.measurement_config["noise_std"]
        if np.any(noise_std > 0):
            y = y + self._measurement_rng.normal(loc=0.0, scale=noise_std, size=self.p)

        yaw_output_index = self._yaw_output_index()
        if yaw_output_index is not None:
            y[yaw_output_index] += self._current_yaw_bias
        self._current_yaw_bias += self.measurement_config["yaw_drift_per_sec"] * self.h
        self._measurement_step += 1

        return y

    def _yaw_output_index(self):
        for i, idx in enumerate(self.output_indices):
            if idx == 2:
                return i
        return None

    def _normalize_measurement_config(self, measurement_config):
        config = {
            "noise_std": np.zeros(self.p),
            "yaw_bias": 0.0,
            "yaw_drift_per_sec": 0.0,
            "seed": 0,
        }
        if measurement_config is None:
            return config

        config["yaw_bias"] = float(measurement_config.get("yaw_bias", 0.0))
        config["yaw_drift_per_sec"] = float(measurement_config.get("yaw_drift_per_sec", 0.0))
        config["seed"] = int(measurement_config.get("seed", 0))

        noise_std = measurement_config.get("noise_std", 0.0)
        if np.isscalar(noise_std):
            config["noise_std"] = float(noise_std) * np.ones(self.p)
        else:
            noise_std = np.asarray(noise_std, dtype=float)
            if noise_std.shape != (self.p,):
                raise ValueError(f"measurement noise_std must have shape ({self.p},), got {noise_std.shape}")
            config["noise_std"] = noise_std

        return config
    
    def output_constraint(self, y):
        return self.F@y <= self.f
    
    def input_constraint(self, u):
        return self.G@u <= self.g
            
    def linearize(self):
        ''' 
        Linearize the dynamics around the stationary reference point.
        The method computes and evaluates the Jacobian matrices A and B for the state and input dynamics
        and stores them as class attributes.
        '''

        # Define symbolic variables
        Phi, Theta, Psi, p, q, r, u, v, w, x, y, z = sp.symbols('Phi Theta Psi p q r u v w x y z')
        u1, u2, u3, u4 = sp.symbols('u1 u2 u3 u4')

        # State and input vectors
        state_vec = sp.Matrix([Phi, Theta, Psi, p, q, r, u, v, w, x, y, z])
        input_vec = sp.Matrix([u1, u2, u3, u4])

        # Define nonlinear equations
        f = sp.Matrix(self.dynamics(state_vec, input_vec, symbolic=True))

        # Compute Jacobians
        A_sym = f.jacobian(state_vec)
        B_sym = f.jacobian(input_vec)

        # Stationary point values
        stationary_values = {
            Phi: 0, Theta: 0, Psi: 0, p: 0, q: 0, r: 0,
            u: 0, v: 0, w: 0, x: 0, y: 0, z: 0,
            u1: self.gravity*self.mass/4/self.max_thrust,
            u2: self.gravity*self.mass/4/self.max_thrust,
            u3: self.gravity*self.mass/4/self.max_thrust,
            u4: self.gravity*self.mass/4/self.max_thrust,
        }

        # Evaluate the matrices at the equilibrium point
        A_eval = A_sym.subs(stationary_values)
        B_eval = B_sym.subs(stationary_values)

        # Convert to NumPy arrays
        self.A = np.array(A_eval).astype(np.float64)
        self.B = np.array(B_eval).astype(np.float64)

        # ZOH discrete-time dynamics matrices
        self.Ad = expm(self.A*self.h)
        self.Bd, _ = quad_vec(lambda s: expm(self.A*s) @ self.B, 0, self.h)
        
    def actuator_matrix(self, b, r, d):
        '''
        Generate the actuator matrix for the quadcopter, relating the squared propeller speeds to the thrust and torques.

        Arguments:
            b (float): Thrust coefficient.
            r (float): Distance from the center of the quadcopter to a propeller.
            d (float): Drag coefficient.

        Returns:
            A (4x4 numpy.ndarray))
        '''

        A = np.array([
            [1, 1, 1, 1],
            [(r/np.sqrt(2)), -(r/np.sqrt(2)), -(r/np.sqrt(2)), (r/np.sqrt(2))],
            [(r/np.sqrt(2)), (r/np.sqrt(2)), -(r/np.sqrt(2)), -(r/np.sqrt(2))],
            [d/b, -d/b, d/b, -d/b]
        ])

        return A
    
    def B0_to_S(self, x_B0):
        R = np.array([[-1, 0, 0],
                      [0, 1, 0],
                      [0, 0, -1]])
        x_S = R @ x_B0
        return x_S

    def B_to_B0_matrix(self, phi, theta, psi):
        R = np.array([[np.cos(theta)*np.cos(psi),   np.sin(phi)*np.sin(theta)*np.cos(psi) - np.cos(phi)*np.sin(psi),   np.cos(phi)*np.sin(theta)*np.cos(psi)+np.sin(phi)*np.sin(psi)],
                      [np.cos(theta)*np.sin(psi),   np.sin(phi)*np.sin(theta)*np.sin(psi) + np.cos(phi)*np.cos(psi),   np.cos(phi)*np.sin(theta)*np.sin(psi)-np.sin(phi)*np.cos(psi)],
                      [-np.sin(theta)        ,   np.sin(phi)*np.cos(theta)                             ,   np.cos(phi)*np.cos(theta)                           ]])
        return R

if __name__ == "__main__":
    h = 0.1
    quadcopter = Quadcopter(h)
    print(quadcopter.Ad)
    print(quadcopter.Bd)
    L, P, E = control.dlqr(quadcopter.Ad, quadcopter.Bd, quadcopter.Q, quadcopter.R)
