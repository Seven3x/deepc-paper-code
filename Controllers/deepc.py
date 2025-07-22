import numpy as np
import cvxpy as cp
from hdf5_reader import HDF5Reader
import mosek

class DeePC:
    
    def __init__(self, system, trajectory, initial_controller, hankel_update_interval = np.inf):
        '''Initialize DeePC controller'''

        self.N = 10 # Prediction horizon
        self.N_u = 1 # Input horizon
        
        self.system = system
        self.n = system.n
        self.m = system.m
        self.p = system.p

        self.trajectory = trajectory
        self.remaining_output_reference = self.trajectory.extend_reference(trajectory.output_reference, self.N)

        self.initial_controller = initial_controller

        self.T_ini = 2 # >= l(B) (how many past datapoints we need for indirect initial state estimation)
        self.T = (self.m+1)*(self.T_ini+self.N+self.n-1) + 1

        # Set up QP optimization
        self.y_ini = cp.Parameter((self.p, self.T_ini))
        self.u_ini = cp.Parameter((self.m, self.T_ini))
        self.y_ini.value = np.zeros((self.p, self.T_ini))
        self.u_ini.value = np.zeros((self.m, self.T_ini))
        self.U_p = cp.Parameter((self.T_ini, self.T - self.T_ini - self.N + 1))
        self.Y_p = cp.Parameter((self.T_ini, self.T - self.T_ini - self.N + 1))
        self.U_f = cp.Parameter((self.N, self.T - self.T_ini - self.N + 1))
        self.Y_f = cp.Parameter((self.N, self.T - self.T_ini - self.N + 1))
    
        self.ref = cp.Parameter((self.p, self.N))
        self.ref.value = self.remaining_output_reference[:,:self.N]

        self.y = cp.Variable((self.p, self.N))
        self.u = cp.Variable((self.m, self.N))
        self.g = cp.Variable((self.T - self.T_ini - self.N + 1, 1))
 
        # Slack variable and cost parameters
        self.sigma_y = cp.Variable((self.p, self.T_ini))
        self.lambda_y = 10e4
        self.lambda_g = 300

        self.con = self.setup_constraints()
        self.cost = self.setup_cost()

        self.problem = cp.Problem(cp.Minimize(self.cost), self.con)

    def construct_hankel_matrix(self, x, L):
        '''
        Constructs a Hankel matrix from given time-series data.
        
        Arguments:
            x - numpy.ndsarray of size (n, T), where n is the number of variables and T is the number of time steps.
            L - int, the window length.
        
        Returns:
            H - np.array of size (L*n, T-L+1), the Hankel matrix.
        '''
        
        n, T = x.shape
        if L > T:
            raise ValueError("Hankel matrix not computable: L cannot be greater than T")
        
        H = np.zeros((L * n, T - L + 1))
        
        for i in range(L):
            H[i * n:(i + 1) * n, :] = x[:, i:i + T - L + 1]
        
        return H
    
    def create_and_partition_hankel_matrices(self):
        # Create Hankel matrices and partition them
        U_d = self.construct_hankel_matrix(self.u_d, self.T_ini + self.N)
        Y_d = self.construct_hankel_matrix(self.y_d, self.T_ini + self.N)

        self.U_p.value = U_d[:self.T_ini,:]
        self.U_f.value = U_d[self.T_ini:,:]
        self.Y_p.value = Y_d[:self.T_ini,:]
        self.Y_f.value = Y_d[self.T_ini:,:]

    def setup_constraints(self):
        # Equality constraint
        u_ini = np.reshape(self.u_ini, (self.m*self.T_ini, 1), order='F')
        y_ini = np.reshape(self.y_ini, (self.p*self.T_ini, 1), order='F')
        sigma_y = np.reshape(self.sigma_y, (self.p*self.T_ini, 1), order='F')
        u = np.reshape(self.u, (self.m*self.N, 1), order='F')
        y = np.reshape(self.y, (self.p*self.N, 1), order='F')

        eq_con = [self.U_p @ self.g == u_ini]
        eq_con += [self.Y_p @ self.g == y_ini + sigma_y]
        eq_con += [self.U_f @ self.g == u]
        eq_con += [self.Y_f @ self.g == y]
            
        # Predicted Ouput and Input Constraint
        ineq_con = []
        ineq_con += [self.system.output_constraint(self.y)]
        ineq_con += [self.system.input_constraint(self.u)]

        con = eq_con + ineq_con

        return con

    def setup_cost(self):
        cost = 0
        for k in range(self.N-1):
            cost += cp.quad_form(self.y[:,k]-self.ref[:,k], self.system.Q) + cp.quad_form(self.u[:,k], self.system.R)
        cost += self.lambda_g*cp.norm1(self.g) + self.lambda_y*cp.norm1(self.sigma_y)

        return cost

    def compute_input(self, x_current):
        y_current = self.system.measure_output(x_current)

        if self.initial_phase_complete():
            u_optimal = self.compute_optimal_control()
        else:
            u_optimal = self.initial_controller.compute_input(x_current)

        self.collect_and_update(y_current, u_optimal)

        return u_optimal
    
    def compute_optimal_control(self):
    
        self.problem.solve(solver=mosek, verbose=False)
        u_optimal = self.u[:, 0].value

        return u_optimal

    def collect_and_update(self, y_current, u_optimal):
        self.y_ini.value[:,:-1] = self.y_ini.value[:,1:]
        self.u_ini.value[:,:-1] = self.u_ini.value[:,1:]

        self.y_ini.value[:,-1] = y_current
        self.u_ini.value[:,-1] = u_optimal

        if not self.finished_collecting:
            if self.T == self.u_d.shape[1]:
                self.finished_collecting = True
            else:
                self.u_d = np.hstack([self.u_d, u_optimal.reshape(-1,1)])
                self.y_d = np.hstack([self.y_d, y_current.reshape(-1,1)])
            
        # Update the reference trajectory.
        self.shift_reference()

    def initial_phase_complete(self):
        return self.finished_collecting and self.is_persistently_excited_of_order_L(self.u_d, self.T_ini + self.N + self.n)

    def shift_reference(self):
        self.remaining_output_reference = self.trajectory.extend_reference(self.remaining_output_reference[:,1:], self.N)
        self.ref.value = self.remaining_output_reference[:,:self.N]
        
    def is_persistently_excited_of_order_L(self, x, L):
        H = self.construct_hankel_matrix(x, L)
        return np.linalg.matrix_rank(H) == L