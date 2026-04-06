import cvxpy as cp
import numpy as np


class LinearMPC:
    def __init__(self, system, trajectory, horizon=10, solver="CLARABEL"):
        self.name = "linear_mpc"
        self.system = system
        self.trajectory = trajectory
        self.N = horizon
        self.solver = solver

        self.Ad = system.Ad
        self.Bd = system.Bd
        self.C = system.C
        self.n = system.n
        self.m = system.m
        self.p = system.p
        self.x_eq = system.x_eq
        self.u_eq = system.u_eq

        self.remaining_output_reference = trajectory.extend_reference(
            trajectory.output_reference,
            self.N,
        )

        self.x0 = cp.Parameter(self.n)
        self.ref = cp.Parameter((self.p, self.N))

        self.x = cp.Variable((self.n, self.N + 1))
        self.u = cp.Variable((self.m, self.N))
        self.last_u = np.asarray(self.u_eq).reshape(-1)

        self.problem = cp.Problem(cp.Minimize(self._build_cost()), self._build_constraints())

    def _build_cost(self):
        cost = 0
        for k in range(self.N):
            y_k = self.C @ self.x[:, k]
            cost += cp.quad_form(y_k - self.ref[:, k], self.system.Q)
            cost += cp.quad_form(self.u[:, k] - self.u_eq, self.system.R)
        terminal_error = self.C @ self.x[:, self.N] - self.ref[:, self.N - 1]
        cost += cp.quad_form(terminal_error, self.system.Q)
        return cost

    def _build_constraints(self):
        constraints = [self.x[:, 0] == self.x0]

        for k in range(self.N):
            constraints += [
                self.x[:, k + 1] - self.x_eq
                == self.Ad @ (self.x[:, k] - self.x_eq) + self.Bd @ (self.u[:, k] - self.u_eq)
            ]
            constraints += [
                self.system.F @ (self.C @ self.x[:, k]) <= self.system.f,
                self.system.G @ self.u[:, k] <= self.system.g,
            ]

        constraints += [
            self.system.F @ (self.C @ self.x[:, self.N]) <= self.system.f,
        ]
        return constraints

    def compute_input(self, x_current, y_current=None):
        del y_current
        self.x0.value = x_current
        self.ref.value = self.remaining_output_reference[:, :self.N]

        solve_kwargs = {
            "verbose": False,
            "ignore_dpp": True,
        }
        if self.solver is not None:
            solve_kwargs["solver"] = self.solver

        self.problem.solve(**solve_kwargs)
        if self.u[:, 0].value is None:
            u_optimal = self.last_u.copy()
        else:
            u_optimal = np.asarray(self.u[:, 0].value).reshape(-1)
            self.last_u = u_optimal.copy()

        self.remaining_output_reference = self.trajectory.extend_reference(
            self.remaining_output_reference[:, 1:],
            self.N,
        )

        return u_optimal
