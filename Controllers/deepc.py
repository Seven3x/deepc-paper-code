import numpy as np
import cvxpy as cp

class DeePC:
    
    def __init__(
        self,
        system,
        trajectory,
        initial_controller,
        is_regularized=True,
        prediction_horizon=25,
        t_ini=6,
        lambda_y=1.0e4,
        lambda_g=30.0,
        solver=None,
    ):
        '''Initialize DeePC controller'''

        self.N = prediction_horizon # Prediction horizon
        
        self.system = system
        self.n = system.n
        self.m = system.m
        self.p = system.p

        self.initial_controller = initial_controller
        self.data_is_persistently_exciting = False

        self.T_ini = t_ini # >= l(B) (how many past datapoints we need for indirect initial state estimation, consider observability matrix)
        self.T = (self.T_ini + self.N)*(1+self.m+self.p) - 1
        self.T += 0
        self.solver = solver

        self.trajectory = trajectory
        if self.trajectory.has_initial_ref:
            self.trajectory.output_reference = np.hstack((self.trajectory.initial_reference(self.T),
                                                        self.trajectory.output_reference))
        self.remaining_output_reference = self.trajectory.extend_reference(self.trajectory.output_reference, self.N)

        # Set up QP optimization
        self.y_ini = cp.Parameter((self.p, self.T_ini))
        self.u_ini = cp.Parameter((self.m, self.T_ini))
        self.y_ini.value = np.zeros((self.p, self.T_ini))
        self.u_ini.value = np.zeros((self.m, self.T_ini))
        self.U_p = cp.Parameter((self.T_ini*self.m, self.T - self.T_ini - self.N + 1))
        self.Y_p = cp.Parameter((self.T_ini*self.p, self.T - self.T_ini - self.N + 1))
        self.U_f = cp.Parameter((self.N*self.m, self.T - self.T_ini - self.N + 1))
        self.Y_f = cp.Parameter((self.N*self.p, self.T - self.T_ini - self.N + 1))
        self.ref = cp.Parameter((self.p, self.N))
        self.ref.value = self.remaining_output_reference[:,:self.N]
        self.g_r = cp.Parameter((self.T - self.T_ini - self.N + 1, 1))

        self.y = cp.Variable((self.p, self.N))
        self.u = cp.Variable((self.m, self.N))
        self.g = cp.Variable((self.T - self.T_ini - self.N + 1, 1))
 
        # Slack variable and cost parameters
        self.is_regularized = is_regularized
        if is_regularized:
            self.sigma_y = cp.Variable((self.p, self.T_ini))
            self.lambda_y = lambda_y
            self.lambda_g = lambda_g

        self.con = self.setup_constraints()
        self.cost = self.setup_cost()

        self.problem = cp.Problem(cp.Minimize(self.cost), self.con)

        # Empty data arrays
        self.u_d = []
        self.y_d = []

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
        U_d = self.construct_hankel_matrix(self.u_d, self.T_ini + self.N)
        Y_d = self.construct_hankel_matrix(self.y_d, self.T_ini + self.N)

        self.U_p.value = U_d[:self.m*self.T_ini,:]
        self.U_f.value = U_d[self.m*self.T_ini:,:]
        self.Y_p.value = Y_d[:self.p*self.T_ini,:]
        self.Y_f.value = Y_d[self.p*self.T_ini:,:]

        self.Hpinv = np.linalg.pinv(np.vstack((self.U_p.value, self.Y_p.value, self.U_f.value, self.Y_f.value)))

        print("Hankel matrices created and partitioned.")

    def setup_constraints(self):
        # Equality constraint
        u_ini = np.reshape(self.u_ini, (self.m*self.T_ini, 1), order='F')
        y_ini = np.reshape(self.y_ini, (self.p*self.T_ini, 1), order='F')
        u = np.reshape(self.u, (self.m*self.N, 1), order='F')
        y = np.reshape(self.y, (self.p*self.N, 1), order='F')

        if self.is_regularized:
            sigma_y = np.reshape(self.sigma_y, (self.p*self.T_ini, 1), order='F')
        
        eq_con = [self.U_p @ self.g == u_ini]
        eq_con += [self.Y_p @ self.g == y_ini + (sigma_y if self.is_regularized else 0)]
        eq_con += [self.U_f @ self.g == u]
        eq_con += [self.Y_f @ self.g == y]
            
        # Predicted Ouput and Input Constraint
        ineq_con = []
        
        for k in range(self.N):
            ineq_con += [self.system.output_constraint(self.y[:,k])]
            ineq_con += [self.system.input_constraint(self.u[:,k])]

        con = eq_con + ineq_con

        return con

    def setup_cost(self):
        cost = 0
        for k in range(self.N):
            cost += cp.quad_form(self.y[:,k]-self.ref[:,k], self.system.Q) + cp.quad_form(self.u[:,k]-self.system.u_eq, self.system.R)
    
        if self.is_regularized:
            cost += self.lambda_g*cp.norm2(self.g-self.g_r) + self.lambda_y*cp.norm2(self.sigma_y)

        return cost

    def compute_input(self, x_current):
        y_current = self.system.measure_output(x_current)

        if self.data_is_persistently_exciting:
            u_optimal = self.compute_optimal_control()
        else:
            u_optimal = self.initial_controller.compute_input(x_current, self.ref.value[:,0])
            
        self.collect_and_update(y_current, u_optimal)

        return u_optimal
    
    def compute_optimal_control(self):
        solve_kwargs = {
            "verbose": False,
            "ignore_dpp": True,
        }
        if self.solver is not None:
            solve_kwargs["solver"] = self.solver

        self.problem.solve(**solve_kwargs)
        u_optimal = self.u[:, 0].value

        return u_optimal

    def collect_and_update(self, y_current, u_optimal):

        self.y_ini.value[:,:-1] = self.y_ini.value[:,1:]
        self.u_ini.value[:,:-1] = self.u_ini.value[:,1:]

        self.y_ini.value[:,-1] = y_current
        self.u_ini.value[:,-1] = u_optimal

        # Update the reference trajectory.
        self.shift_reference()

        if not self.data_is_persistently_exciting:
            self.u_d.append(u_optimal)
            self.y_d.append(y_current)

            if len(self.u_d) == self.T:
                self.u_d = np.array(self.u_d).T
                self.y_d = np.array(self.y_d).T
                self.create_and_partition_hankel_matrices()
                self.data_is_persistently_exciting = self.is_persistently_excited_of_order_L(self.u_d, self.T_ini + self.N + self.n)
        
        if self.data_is_persistently_exciting:
            # Update steady state solution g_r 
            uy_r = np.hstack((np.tile(self.system.u_eq, self.T_ini),
                            np.tile(self.ref.value[:,0], self.T_ini),
                            np.tile(self.system.u_eq, self.N),
                            np.tile(self.ref.value[:,0], self.N),
                            )).reshape(-1, 1)
            
            self.g_r.value = self.Hpinv @ uy_r

    def shift_reference(self):
        self.remaining_output_reference = self.trajectory.extend_reference(self.remaining_output_reference[:,1:], self.N)
        self.ref.value = self.remaining_output_reference[:,:self.N]
        
    def is_persistently_excited_of_order_L(self, x, L):
        H = self.construct_hankel_matrix(x, L)
        
        if np.linalg.matrix_rank(H) == L*self.m:
            print(f"Data is persistently exciting of order {L}!")
            return True
        else:
            raise ValueError(f"Data is not persistently exciting, increase the amount of datapoints used in Hankelmatrix.")
