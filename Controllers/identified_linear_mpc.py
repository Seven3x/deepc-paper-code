from types import SimpleNamespace

import cvxpy as cp
import numpy as np
import control


class IdentifiedLinearMPC:
    def __init__(
        self,
        system,
        trajectory,
        initial_controller,
        id_data_length,
        horizon=10,
        solver="CLARABEL",
        ridge=1.0e-6,
    ):
        self.name = "identified_linear_mpc"
        self.system = system
        self.trajectory = trajectory
        self.initial_controller = initial_controller
        self.id_data_length = max(int(id_data_length), 2)
        self.task_start_steps = self.id_data_length
        self.N = int(horizon)
        self.solver = solver
        self.ridge = max(float(ridge), 0.0)

        self.current_step = 0
        self.identified = False
        self.last_u = np.asarray(system.u_eq, dtype=float).reshape(-1)

        if self.trajectory.has_initial_ref:
            self.full_output_reference = np.hstack(
                (self.trajectory.initial_reference(self.id_data_length), self.trajectory.output_reference)
            )
        else:
            self.full_output_reference = self.trajectory.output_reference.copy()

        self.x_hist = []
        self.u_hist = []
        self._pending_x = None
        self._pending_u = None

        self._identified_system = None
        self.x0 = None
        self.ref = None
        self.x = None
        self.u = None
        self.problem = None
        self.observer_gain = None
        self.x_hat = None

    def _extract_measurement(self, y_current):
        if y_current is None:
            return None, None
        if isinstance(y_current, dict):
            y = np.asarray(y_current.get("output"), dtype=float).reshape(-1)
            mask = np.asarray(y_current.get("output_mask", np.ones_like(y)), dtype=float).reshape(-1)
            return y, mask > 0.5
        y = np.asarray(y_current, dtype=float).reshape(-1)
        return y, np.ones_like(y, dtype=bool)

    def _measurement_to_state_guess(self, y_meas):
        x_guess = np.asarray(self.system.x_eq, dtype=float).reshape(-1).copy()
        for output_row, state_idx in enumerate(self.system.output_indices):
            x_guess[state_idx] = y_meas[output_row]
        return x_guess

    def _update_state_estimate(self, y_current):
        y_meas, mask = self._extract_measurement(y_current)
        if y_meas is None:
            if self.x_hat is None:
                self.x_hat = np.asarray(self.system.x_eq, dtype=float).reshape(-1).copy()
            return self.x_hat.copy()

        if self.x_hat is None:
            self.x_hat = self._measurement_to_state_guess(y_meas)
            return self.x_hat.copy()

        x_pred = (
            np.asarray(self.system.x_eq, dtype=float).reshape(-1)
            + self._identified_system.Ad @ (self.x_hat - np.asarray(self.system.x_eq, dtype=float).reshape(-1))
            + self._identified_system.Bd @ (self.last_u - np.asarray(self.system.u_eq, dtype=float).reshape(-1))
        )

        if np.all(mask):
            innovation = y_meas - self._identified_system.C @ x_pred
            self.x_hat = x_pred + self.observer_gain @ innovation
            return self.x_hat.copy()

        observed_rows = np.flatnonzero(mask)
        if observed_rows.size == 0:
            self.x_hat = x_pred
            return self.x_hat.copy()

        c_obs = self._identified_system.C[observed_rows, :]
        l_obs = self.observer_gain[:, observed_rows]
        innovation = y_meas[observed_rows] - c_obs @ x_pred
        self.x_hat = x_pred + l_obs @ innovation
        return self.x_hat.copy()

    def _reference_window(self, start_step, horizon):
        ref = self.full_output_reference[:, start_step:]
        ref = self.trajectory.extend_reference(ref, horizon)
        return ref[:, :horizon]

    def _current_reference(self):
        return self._reference_window(self.current_step, 1)[:, 0]

    def _record_transition(self, x_current):
        x_current = np.asarray(x_current, dtype=float).reshape(-1)
        if self._pending_x is not None and self._pending_u is not None:
            self.x_hist.append((self._pending_x.copy(), x_current.copy()))
            self.u_hist.append(self._pending_u.copy())
        self._pending_x = x_current

    def _maybe_identify_model(self):
        if self.identified:
            return
        if len(self.x_hist) < self.id_data_length:
            return

        x_prev = np.column_stack([pair[0] for pair in self.x_hist])
        x_next = np.column_stack([pair[1] for pair in self.x_hist])
        u_prev = np.column_stack(self.u_hist)
        x_prev_centered = x_prev - np.asarray(self.system.x_eq, dtype=float).reshape(-1, 1)
        x_next_centered = x_next - np.asarray(self.system.x_eq, dtype=float).reshape(-1, 1)
        u_prev_centered = u_prev - np.asarray(self.system.u_eq, dtype=float).reshape(-1, 1)
        xu = np.vstack((x_prev_centered, u_prev_centered))

        gram = xu @ xu.T
        if self.ridge > 0.0:
            gram = gram + self.ridge * np.eye(gram.shape[0])
        theta = x_next_centered @ xu.T @ np.linalg.pinv(gram)
        n = self.system.n
        self._identified_system = SimpleNamespace(
            Ad=np.asarray(theta[:, :n], dtype=float),
            Bd=np.asarray(theta[:, n:], dtype=float),
            C=self.system.C,
            n=self.system.n,
            m=self.system.m,
            p=self.system.p,
            x_eq=self.system.x_eq,
            u_eq=self.system.u_eq,
            F=self.system.F,
            f=self.system.f,
            G=self.system.G,
            g=self.system.g,
            Q=self.system.Q,
            R=self.system.R,
            ridge=self.ridge,
        )
        measurement_noise = np.asarray(self.system.measurement_config["noise_std"], dtype=float).reshape(-1)
        measurement_noise = np.maximum(measurement_noise, 1.0e-3)
        qn = 1.0e-3 * np.eye(self.system.n)
        rn = np.diag(np.square(measurement_noise))
        self.observer_gain, _, _ = control.dlqe(
            self._identified_system.Ad,
            np.eye(self.system.n),
            self._identified_system.C,
            qn,
            rn,
        )
        self.x_hat = np.asarray(self.system.x_eq, dtype=float).reshape(-1).copy()

        self.x0 = cp.Parameter(self.system.n)
        self.ref = cp.Parameter((self.system.p, self.N))
        self.x = cp.Variable((self.system.n, self.N + 1))
        self.u = cp.Variable((self.system.m, self.N))
        self.problem = cp.Problem(cp.Minimize(self._build_cost()), self._build_constraints())
        self.identified = True

    def _build_cost(self):
        cost = 0
        for k in range(self.N):
            y_k = self._identified_system.C @ self.x[:, k]
            cost += cp.quad_form(y_k - self.ref[:, k], self._identified_system.Q)
            cost += cp.quad_form(self.u[:, k] - self._identified_system.u_eq, self._identified_system.R)
        terminal_error = self._identified_system.C @ self.x[:, self.N] - self.ref[:, self.N - 1]
        cost += cp.quad_form(terminal_error, self._identified_system.Q)
        return cost

    def _build_constraints(self):
        constraints = [self.x[:, 0] == self.x0]
        for k in range(self.N):
            constraints += [
                self.x[:, k + 1] - self._identified_system.x_eq
                == self._identified_system.Ad @ (self.x[:, k] - self._identified_system.x_eq)
                + self._identified_system.Bd @ (self.u[:, k] - self._identified_system.u_eq)
            ]
            constraints += [
                self._identified_system.F @ (self._identified_system.C @ self.x[:, k]) <= self._identified_system.f,
                self._identified_system.G @ self.u[:, k] <= self._identified_system.g,
            ]
        constraints += [self._identified_system.F @ (self._identified_system.C @ self.x[:, self.N]) <= self._identified_system.f]
        return constraints

    def compute_input(self, x_current, y_current=None):
        x_current = np.asarray(x_current, dtype=float).reshape(-1)
        self._record_transition(x_current)
        self._maybe_identify_model()

        if not self.identified:
            ref = self._current_reference()
            u_optimal = np.asarray(self.initial_controller.compute_input(x_current, ref), dtype=float).reshape(-1)
            self._pending_u = u_optimal.copy()
            self.last_u = u_optimal.copy()
            self.current_step += 1
            return u_optimal

        x_est = self._update_state_estimate(y_current)
        self.x0.value = x_est
        self.ref.value = self._reference_window(self.current_step, self.N)
        solve_kwargs = {"verbose": False, "ignore_dpp": True}
        if self.solver is not None:
            solve_kwargs["solver"] = self.solver
        self.problem.solve(**solve_kwargs)

        if self.u[:, 0].value is None:
            u_optimal = self.last_u.copy()
        else:
            u_optimal = np.asarray(self.u[:, 0].value).reshape(-1)
            self.last_u = u_optimal.copy()

        self._pending_u = u_optimal.copy()
        self.current_step += 1
        return u_optimal
