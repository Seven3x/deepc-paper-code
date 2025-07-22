import numpy as np
import sympy as sp
from scipy.linalg import expm
from scipy.integrate import quad_vec, quad

class Quadcopter:
    def __init__(self, h):
        self.n = 12
        self.m = 4
        self.h = h #ZOH sampling time
        self.x0 = np.zeros(self.n)

        # Assume full state measurement
        self.output_indices = [0,1,2,3,4,5,6,7,8,9,10,11]
        self.p = 12
        self.C = np.eye(self.p)

        self.name = f"Quadcopter_h={self.h}"

        self.constant_ref = np.array([0])
        self.Mx = np.array([[1],[0]])
        self.x_r = self.Mx @ self.constant_ref
        
        self.labels = {
            "x": [r"$\phi$", r"$\theta$", r"$\psi$", "$p$", "$q$", "$r$", "$u$", "$v$", "$w$", "$x_{B_0}$", "$y_{B_0}$", "$z_{B_0}$"],
            "u": [r"$\Omega_1^2$", r"$\Omega_2^2$", r"$\Omega_3^2$", r"$\Omega_4^2$"],
        }
        self.labels["y"] = [self.labels["x"][i] for i in self.output_indices]

        # Properties
        self.mass = 1
        self.Inertia = np.diag([0.006, 0.006, 0.0011])
        self.gravity = 9.82 #[m/s^2]
        self.r = 0.15 # Distance from the center of the quadcopter to a propeller

        # Goal operating point, In S coordinates
        self.xg = np.array([0,0,0,0,0,0,5,0,5])
        self.ug = 1/4*self.mass*self.gravity*np.ones(4)

        # Input and output constraints
        self.x_lower = np.array([-np.pi, -np.pi, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -10, -10, -10])
        self.x_upper = np.array([np.pi, np.pi, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, 10, 10, 10])
        self.y_lower = np.array([self.x_lower[i] for i in self.output_indices])
        self.y_upper = np.array([self.x_upper[i] for i in self.output_indices])
        self.u_lower = np.zeros(4)
        self.max_thrust = self.gravity*self.mass/2
        self.u_upper = self.max_thrust*np.ones(4)

        # Cost parameters for predictive control
        self.Q = 100*np.eye(self.p)
        self.R = 0.1*np.eye(self.m)
        self.Q_N = 100*np.eye(self.p)
    
        # Aerodynamic properties
        self.b = 1.e-05  # Thrust coefficient
        self.d = 3.5e-8  # Drag coefficient

        # Linearized dynamics around the stationary reference point
        self.linearize()

    def dynamics(self, x, u_input):
        ''' 
        Nonlinear continuous-time dynamics of the quadcopter.
        
        Arguments:
            x (12x1 np.ndarray): State vector
            u (4x1 np.ndarray): Input vector (squared propeller speeds)
        
        Returns:
            xdot (12x1 np.ndarray): Time derivative of the state vector
        '''

        Phi, Theta, Psi, p, q, r, u, v, w, _, _, _ = x
        Ix, Iy, Iz = np.diag(self.Inertia)
        ft, Tx, Ty, Tz = self.actuator_matrix(self.b, self.r, self.d) @ u_input
        g = self.gravity
        m = self.mass

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

        return xdot
        
    def discrete_dynamics(self, x, u, use_casadi=False): #Nonlinear discrete dynamics
        x_next = x + self.h * self.dynamics(x, u, use_casadi=use_casadi)
        return x_next

    def measure_output(self, x):
        y = self.C @ x #measure position
        return y
    
    def output_constraint(self, y):
        return self.F@y <= self.f
    
    def input_constraint(self, u):
        return self.G@u <= self.g
    
    def stage_cost(self, y, r, u):
        return (y-r).T @ self.Q @ (y-r) + u.T @ self.R @ u
        
    def terminal_cost(self, y, r):
        return (y-r).T @ self.Q_N @ (y-r)
        
    def linearize(self):
        ''' 
        Linearize the dynamics around the stationary reference point.
        The method computes and evaluates the Jacobian matrices A and B for the state and input dynamics
        and stores them as class attributes.
        '''

        # Define symbolic variables
        Phi, Theta, Psi, p, q, r, u, v, w, x, y, z = sp.symbols('Phi Theta Psi p q r u v w x y z')
        omgega1_sq, omgega2_sq, omgega3_sq, omgega4_sq = sp.symbols('omgega1_sq omgega2_sq omgega3_sq omgega4_sq')

        # State and input vectors
        state_vec = sp.Matrix([Phi, Theta, Psi, p, q, r, u, v, w, x, y, z])
        input_vec = sp.Matrix([omgega1_sq, omgega2_sq, omgega3_sq, omgega4_sq])

        # Define nonlinear equations
        f = sp.Matrix(self.dynamics(state_vec, input_vec))

        # Compute Jacobians
        A_sym = f.jacobian(state_vec)
        B_sym = f.jacobian(input_vec)

        # Stationary point values
        stationary_values = {
            Phi: 0, Theta: 0, Psi: 0, p: 0, q: 0, r: 0,
            u: 0, v: 0, w: 0, x: 0, y: 0, z: 0,
            omgega1_sq: self.gravity*self.mass/(4*self.b),
            omgega2_sq: self.gravity*self.mass/(4*self.b),
            omgega3_sq: self.gravity*self.mass/(4*self.b),
            omgega4_sq: self.gravity*self.mass/(4*self.b),
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
            [b, b, b, b],
            [(b*r/np.sqrt(2)), -(b*r/np.sqrt(2)), (b*r/np.sqrt(2)), -(b*r/np.sqrt(2))],
            [(b*r/np.sqrt(2)), (b*r/np.sqrt(2)), -(b*r/np.sqrt(2)), -(b*r/np.sqrt(2))],
            [d, -d, d, -d]
        ])

        return A

if __name__ == "__main__":
    h = 0.1
    quadcopter = Quadcopter(h)