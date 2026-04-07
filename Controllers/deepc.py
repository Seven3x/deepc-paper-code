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
        regularization_mode="uniform",
        sigma_y_group_weights=None,
        residual_weight_floor=1.0e-3,
        residual_weight_min=0.1,
        residual_weight_max=2.0,
        block_lambda_roll_pitch=None,
        block_lambda_yaw=None,
        block_lambda_position=None,
        data_length_extra=0,
        history_alignment="naive",
        iv_projection_lag=1,
        consistency_gate_lambda=3.0,
        consistency_gate_clip=3.0,
        consistency_gate_eps=1.0e-6,
        controller_health_mode="nominal",
        bank_selection_mode="fixed",
        bank_transfer_mode="none",
        bank_transfer_interval_steps=10,
    ):
        '''Initialize DeePC controller'''

        self.N = prediction_horizon # Prediction horizon
        
        self.system = system
        self.n = system.n
        self.m = system.m
        self.p = system.p
        self.bank_mode = "dual_bank"
        self.controller_health_mode = str(controller_health_mode)
        if self.controller_health_mode not in {"nominal", "degraded"}:
            raise ValueError("controller_health_mode must be nominal or degraded")
        self.bank_selection_mode = str(bank_selection_mode)
        if self.bank_selection_mode not in {"fixed", "oracle_minimal"}:
            raise ValueError("bank_selection_mode must be fixed or oracle_minimal")
        self.bank_transfer_mode = str(bank_transfer_mode)
        if self.bank_transfer_mode not in {"none", "warm_start_only", "adapt_only", "warm_start_adapt"}:
            raise ValueError(
                "bank_transfer_mode must be one of: none, warm_start_only, adapt_only, warm_start_adapt"
            )
        self.bank_transfer_interval_steps = max(int(bank_transfer_interval_steps), 1)
        self.health_mode = self.controller_health_mode
        self.plant_health_mode = "nominal"
        self.requested_bank_name = self.controller_health_mode
        self.control_bank_name = self.controller_health_mode
        self.training_bank_name = self.controller_health_mode
        self.candidate_bank_scores = {}
        self.degraded_bank_bootstrapped = False
        self.degraded_bank_adaptation_steps = 0
        self.yaw_output_rows = [row for row, idx in enumerate(system.output_indices) if idx == 2]
        self.non_yaw_output_rows = [row for row in range(self.p) if row not in self.yaw_output_rows]

        self.initial_controller = initial_controller
        self.data_is_persistently_exciting = False

        self.T_ini = t_ini # >= l(B) (how many past datapoints we need for indirect initial state estimation, consider observability matrix)
        data_length_from_outputs = (self.T_ini + self.N) * (1 + self.m + self.p) - 1
        persistence_order = self.T_ini + self.N + self.n
        min_data_length_for_inputs = (self.m + 1) * persistence_order - 1
        self.T = max(data_length_from_outputs, min_data_length_for_inputs) + data_length_extra
        self.solver = solver
        self.regularization_mode = regularization_mode
        self.residual_weight_floor = residual_weight_floor
        self.residual_weight_min = residual_weight_min
        self.residual_weight_max = residual_weight_max
        self.residual_stat_summary = None
        self.consistency_gate_summary = None
        self.history_alignment = history_alignment
        self.iv_projection_lag = max(int(iv_projection_lag), 1)
        self.iv_projection_summary = None
        self.consistency_gate_lambda = consistency_gate_lambda
        self.consistency_gate_clip = consistency_gate_clip
        self.consistency_gate_eps = consistency_gate_eps
        self.block_lambda_roll_pitch = block_lambda_roll_pitch if block_lambda_roll_pitch is not None else lambda_y
        self.block_lambda_yaw = block_lambda_yaw if block_lambda_yaw is not None else lambda_y
        self.block_lambda_position = block_lambda_position if block_lambda_position is not None else lambda_y

        self.trajectory = trajectory
        if self.trajectory.has_initial_ref:
            self.trajectory.output_reference = np.hstack((self.trajectory.initial_reference(self.T),
                                                        self.trajectory.output_reference))
        self.full_output_reference = self.trajectory.output_reference.copy()
        self.current_reference_step = 0
        self.remaining_output_reference = self._reference_window(self.current_reference_step, self.N)

        # Set up QP optimization
        self.y_ini = cp.Parameter((self.p, self.T_ini))
        self.u_ini = cp.Parameter((self.m, self.T_ini))
        self.y_ini_mask = cp.Parameter((self.p, self.T_ini), nonneg=True)
        self.y_ini_source_steps = cp.Parameter((self.p, self.T_ini))
        self.y_ini_timestamps = cp.Parameter((self.p, self.T_ini))
        self.y_ini.value = np.zeros((self.p, self.T_ini))
        self.u_ini.value = np.zeros((self.m, self.T_ini))
        self.y_ini_mask.value = np.ones((self.p, self.T_ini))
        self.y_ini_source_steps.value = -np.ones((self.p, self.T_ini))
        self.y_ini_timestamps.value = -self.system.h * np.ones((self.p, self.T_ini))
        self.U_p = cp.Parameter((self.T_ini*self.m, self.T - self.T_ini - self.N + 1))
        self.Y_p = cp.Parameter((self.T_ini*self.p, self.T - self.T_ini - self.N + 1))
        self.U_f = cp.Parameter((self.N*self.m, self.T - self.T_ini - self.N + 1))
        self.Y_f = cp.Parameter((self.N*self.p, self.T - self.T_ini - self.N + 1))
        self.ref = cp.Parameter((self.p, self.N))
        self.ref.value = self.remaining_output_reference[:,:self.N]
        self.g_r = cp.Parameter((self.T - self.T_ini - self.N + 1, 1))
        self.consistency_column_weights = cp.Parameter((self.T - self.T_ini - self.N + 1, 1), nonneg=True)
        self.consistency_column_weights.value = np.zeros((self.T - self.T_ini - self.N + 1, 1))
        self.U_p.value = np.zeros((self.T_ini * self.m, self.T - self.T_ini - self.N + 1))
        self.Y_p.value = np.zeros((self.T_ini * self.p, self.T - self.T_ini - self.N + 1))
        self.U_f.value = np.zeros((self.N * self.m, self.T - self.T_ini - self.N + 1))
        self.Y_f.value = np.zeros((self.N * self.p, self.T - self.T_ini - self.N + 1))

        self.y = cp.Variable((self.p, self.N))
        self.u = cp.Variable((self.m, self.N))
        self.g = cp.Variable((self.T - self.T_ini - self.N + 1, 1))
 
        # Slack variable and cost parameters
        self.is_regularized = is_regularized
        if is_regularized:
            self.sigma_y = cp.Variable((self.p, self.T_ini))
            self.lambda_y = lambda_y
            self.lambda_g = lambda_g
            self.sigma_y_group_weights = cp.Parameter((self.p, 1), nonneg=True)
            self.sigma_y_group_weights.value = self._normalize_sigma_y_group_weights(sigma_y_group_weights)

        self.con = self.setup_constraints()
        self.cost = self.setup_cost()

        self.problem = cp.Problem(cp.Minimize(self.cost), self.con)

        # Empty data arrays
        self.u_d = []
        self.y_d = []
        self.measurement_residual_d = []
        self.applied_inputs_by_step = {}
        self.received_history = []
        self.aligned_history = []
        self.async_source_slots = {}
        self.next_async_training_source_step = 0
        self.latest_measurement_metadata = None
        self.current_delay_steps = 0
        self.Hpinv = None
        self.bank_mode = "dual_bank"
        self.bank_states = {
            "nominal": self._capture_bank_state(),
            "degraded": self._capture_bank_state(),
        }

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
        if self.history_alignment == "iv_projected":
            U_d, Y_d = self._apply_iv_projection(U_d, Y_d)

        self.U_p.value = U_d[:self.m*self.T_ini,:]
        self.U_f.value = U_d[self.m*self.T_ini:,:]
        self.Y_p.value = Y_d[:self.p*self.T_ini,:]
        self.Y_f.value = Y_d[self.p*self.T_ini:,:]

        self.Hpinv = np.linalg.pinv(np.vstack((self.U_p.value, self.Y_p.value, self.U_f.value, self.Y_f.value)))

        # Residual-statistics modes all consume the same calibration bank that
        # was collected before DeePC became persistently exciting.
        if self.is_regularized and self.regularization_mode == "residual_stats":
            self.sigma_y_group_weights.value = self._compute_residual_stat_weights()
        elif self.is_regularized and self.regularization_mode == "residual_variance":
            self.sigma_y_group_weights.value = self._compute_residual_variance_weights()
        elif self.is_regularized and self.regularization_mode == "residual_bias_variance":
            self.sigma_y_group_weights.value = self._compute_residual_bias_variance_weights()
        elif self.is_regularized and self.regularization_mode == "robust_residual_stats":
            self.sigma_y_group_weights.value = self._compute_robust_residual_stat_weights()

        print("Hankel matrices created and partitioned.")

    def _apply_iv_projection(self, U_d, Y_d):
        lag = self.iv_projection_lag
        order = self.T_ini + self.N
        u_lagged = self._lagged_signal_matrix(self.u_d, lag)
        y_lagged = self._lagged_signal_matrix(self.y_d, lag)
        Z_u = self.construct_hankel_matrix(u_lagged, order)
        Z_y = self.construct_hankel_matrix(y_lagged, order)
        Z = np.vstack((Z_u, Z_y))
        gram = Z @ Z.T
        projection = Z.T @ np.linalg.pinv(gram) @ Z
        self.iv_projection_summary = {
            "lag": int(lag),
            "instrument_rows": int(Z.shape[0]),
            "instrument_rank": int(np.linalg.matrix_rank(Z)),
            "projection_rank": int(np.linalg.matrix_rank(projection)),
            "num_columns": int(U_d.shape[1]),
        }
        return U_d @ projection, Y_d @ projection

    def _lagged_signal_matrix(self, data, lag):
        array = np.asarray(data, dtype=float)
        if array.ndim != 2:
            raise ValueError(f"Expected a 2D signal matrix, got shape {array.shape}")
        if lag <= 0:
            return array
        prefix = np.repeat(array[:, :1], lag, axis=1)
        suffix = array[:, :-lag] if array.shape[1] > lag else np.empty((array.shape[0], 0))
        return np.hstack((prefix, suffix))

    def setup_constraints(self):
        # Equality constraint
        u_ini = np.reshape(self.u_ini, (self.m*self.T_ini, 1), order='F')
        y_ini = np.reshape(self.y_ini, (self.p*self.T_ini, 1), order='F')
        u = np.reshape(self.u, (self.m*self.N, 1), order='F')
        y = np.reshape(self.y, (self.p*self.N, 1), order='F')

        if self.is_regularized:
            sigma_y = np.reshape(self.sigma_y, (self.p*self.T_ini, 1), order='F')
        
        eq_con = [self.U_p @ self.g == u_ini]
        if self.history_alignment == "async_masked":
            Y_pg = cp.reshape(self.Y_p @ self.g, (self.p, self.T_ini), order="F")
            if self.is_regularized:
                eq_con += [cp.multiply(self.y_ini_mask, Y_pg - self.y_ini - self.sigma_y) == 0]
                eq_con += [cp.multiply(1.0 - self.y_ini_mask, self.sigma_y) == 0]
            else:
                eq_con += [cp.multiply(self.y_ini_mask, Y_pg - self.y_ini) == 0]
        elif self.regularization_mode == "drop_yaw_past" and self.yaw_output_rows:
            Y_pg = cp.reshape(self.Y_p @ self.g, (self.p, self.T_ini), order="F")
            for row in self.non_yaw_output_rows:
                eq_con += [Y_pg[row, :] == self.y_ini[row, :]]
        elif self.is_regularized and self.regularization_mode == "yaw_selective_slack" and self.yaw_output_rows:
            Y_pg = cp.reshape(self.Y_p @ self.g, (self.p, self.T_ini), order="F")
            for row in self.non_yaw_output_rows:
                eq_con += [Y_pg[row, :] == self.y_ini[row, :]]
            for row in self.yaw_output_rows:
                eq_con += [Y_pg[row, :] == self.y_ini[row, :] + self.sigma_y[row, :]]
            for row in self.non_yaw_output_rows:
                eq_con += [self.sigma_y[row, :] == 0]
        else:
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
            cost += self.lambda_g*cp.norm2(self.g-self.g_r)
            if self.regularization_mode == "block_l2":
                cost += self._block_sigma_y_cost()
            elif self.regularization_mode == "yaw_selective_slack":
                cost += self._yaw_selective_sigma_y_cost()
            else:
                weighted_sigma_y = cp.multiply(self.sigma_y_group_weights, self.sigma_y)
                cost += self.lambda_y*cp.norm2(weighted_sigma_y)

        if self.history_alignment == "consistency_gated_time_aligned":
            cost += self.consistency_gate_lambda * cp.sum(
                cp.multiply(self.consistency_column_weights, cp.square(self.g))
            )

        return cost

    def _block_sigma_y_cost(self):
        cost = 0
        block_rows = self._sigma_y_blocks()
        if block_rows["roll_pitch"]:
            cost += self.block_lambda_roll_pitch * cp.norm2(self.sigma_y[block_rows["roll_pitch"], :])
        if block_rows["yaw"]:
            cost += self.block_lambda_yaw * cp.norm2(self.sigma_y[block_rows["yaw"], :])
        if block_rows["position"]:
            cost += self.block_lambda_position * cp.norm2(self.sigma_y[block_rows["position"], :])
        return cost

    def _yaw_selective_sigma_y_cost(self):
        if not self.yaw_output_rows:
            return 0
        return self.lambda_y * cp.norm2(self.sigma_y[self.yaw_output_rows, :])

    def _sigma_y_blocks(self):
        blocks = {
            "roll_pitch": [],
            "yaw": [],
            "position": [],
        }
        for row, state_index in enumerate(self.system.output_indices):
            if state_index in (0, 1):
                blocks["roll_pitch"].append(row)
            elif state_index == 2:
                blocks["yaw"].append(row)
            elif state_index in (9, 10, 11):
                blocks["position"].append(row)
        return blocks

    def _normalize_sigma_y_group_weights(self, sigma_y_group_weights):
        if sigma_y_group_weights is None:
            return np.ones((self.p, 1))

        weights = np.asarray(sigma_y_group_weights, dtype=float)
        if weights.shape == (self.p,):
            weights = weights.reshape(self.p, 1)
        if weights.shape != (self.p, 1):
            raise ValueError(f"sigma_y_group_weights must have shape ({self.p},) or ({self.p}, 1), got {weights.shape}")
        if np.any(weights <= 0):
            raise ValueError("sigma_y_group_weights must be strictly positive")
        return weights

    def _compute_residual_stat_weights(self):
        residuals = self._measurement_residual_matrix()
        residual_rms = np.sqrt(np.mean(np.square(residuals), axis=1))
        return self._scale_to_weights(residual_rms, residuals, residual_rms)

    def _compute_residual_variance_weights(self):
        residuals = self._measurement_residual_matrix()
        residual_variance = np.var(residuals, axis=1)
        residual_std = np.sqrt(np.maximum(residual_variance, 0.0))
        return self._scale_to_weights(residual_std, residuals, residual_std)

    def _compute_residual_bias_variance_weights(self):
        residuals = self._measurement_residual_matrix()
        residual_mean = np.mean(residuals, axis=1)
        residual_variance = np.var(residuals, axis=1)
        combined_scale = np.sqrt(np.square(residual_mean) + residual_variance)
        return self._scale_to_weights(combined_scale, residuals, combined_scale)

    def _compute_robust_residual_stat_weights(self):
        residuals = self._measurement_residual_matrix()
        residual_median = np.median(residuals, axis=1)
        centered = residuals - residual_median[:, None]
        mad = np.median(np.abs(centered), axis=1)
        robust_scale = 1.4826 * mad
        combined_scale = np.sqrt(np.square(residual_median) + np.square(robust_scale))
        return self._scale_to_weights(combined_scale, residuals, combined_scale)

    def _measurement_residual_matrix(self):
        residuals = np.asarray(self.measurement_residual_d, dtype=float)
        if residuals.ndim != 2:
            raise ValueError(
                "measurement_residual_d must be a 2D array with shape (p, T); "
                f"got {residuals.ndim}D"
            )
        if residuals.shape[0] != self.p:
            raise ValueError(
                "measurement_residual_d has the wrong number of output rows; "
                f"expected {self.p}, got {residuals.shape[0]}"
            )
        return residuals

    def _scale_to_weights(self, scales, residuals=None, residual_scale=None):
        scales = np.asarray(scales, dtype=float).reshape(self.p)
        effective_scales = np.maximum(scales, self.residual_weight_floor)
        inv_scales = 1.0 / effective_scales
        median_inv_scale = np.median(inv_scales)
        weights = inv_scales / median_inv_scale
        weights = np.clip(weights, self.residual_weight_min, self.residual_weight_max)
        if residuals is not None:
            residual_array = np.asarray(residuals, dtype=float)
            self.residual_stat_summary = {
                "mode": self.regularization_mode,
                "num_samples": int(residual_array.shape[1]),
                "mean_residual": np.mean(residual_array, axis=1).tolist(),
                "std_residual": np.std(residual_array, axis=1).tolist(),
                "residual_scale": np.asarray(residual_scale, dtype=float).reshape(self.p).tolist(),
                "effective_scale": effective_scales.tolist(),
                "weights": weights.tolist(),
            }
        return weights.reshape(self.p, 1)

    def compute_input(self, x_current, y_current=None):
        self.plant_health_mode = self._plant_health_mode()
        measurement_packet = self._normalize_measurement_packet(x_current, y_current)
        delivered_y = measurement_packet["output"]
        measurement_residual = measurement_packet["output_mask"] * (
            delivered_y - measurement_packet["true_output"]
        )
        desired_health_mode = self._desired_health_mode()
        self._bootstrap_degraded_bank_if_needed(desired_health_mode)
        control_bank = self._select_control_bank(
            desired_health_mode,
            x_current=x_current,
            measurement_packet=measurement_packet,
            measurement_residual=measurement_residual,
        )
        self.requested_bank_name = desired_health_mode
        self.control_bank_name = control_bank
        self._restore_bank_state(control_bank)
        u_optimal, _ = self._compute_control_candidate_for_active_bank(
            x_current,
            measurement_packet,
            measurement_residual,
        )

        self._restore_bank_state(desired_health_mode)
        self.health_mode = desired_health_mode
        self.training_bank_name = desired_health_mode
        self.collect_and_update(measurement_packet, u_optimal, measurement_residual)
        self._capture_bank_state_to(desired_health_mode)

        return u_optimal
    
    def compute_optimal_control(self, control_offset=0):
        solve_kwargs = {
            "verbose": False,
            "ignore_dpp": True,
        }
        if self.solver is not None:
            solve_kwargs["solver"] = self.solver

        self.problem.solve(**solve_kwargs)
        control_index = min(max(int(control_offset), 0), self.N - 1)
        u_optimal = self.u[:, control_index].value

        return u_optimal

    def _prepare_active_bank_for_measurement(self, measurement_packet, measurement_residual):
        if (
            self.history_alignment in {"time_aligned", "suffix_aligned", "consistency_gated_time_aligned"}
            and self.data_is_persistently_exciting
            and measurement_packet["source_step"] < measurement_packet["delivered_step"]
            and measurement_packet["source_step"] in self.applied_inputs_by_step
        ):
            aligned_pair = self._build_history_pair(
                pair_input=self.applied_inputs_by_step[measurement_packet["source_step"]],
                pair_step=measurement_packet["source_step"],
                measurement_packet=measurement_packet,
                measurement_residual=measurement_residual,
            )
            self.aligned_history.append(aligned_pair)
            self._refresh_ini_windows()
        elif (
            self.history_alignment == "async_masked"
            and self.data_is_persistently_exciting
            and self._effective_reference_step(measurement_packet) < measurement_packet["delivered_step"]
        ):
            self._update_async_source_slots(measurement_packet, measurement_residual)
            self._refresh_ini_windows()

        if self.history_alignment in {"time_aligned", "delay_ref_only", "suffix_aligned", "consistency_gated_time_aligned"}:
            optimization_reference_step = measurement_packet["source_step"]
            control_offset = measurement_packet["delay_steps"]
        elif self.history_alignment == "async_masked":
            optimization_reference_step = self._effective_reference_step(measurement_packet)
            control_offset = max(0, measurement_packet["delivered_step"] - optimization_reference_step)
        else:
            optimization_reference_step = measurement_packet["delivered_step"]
            control_offset = 0

        self.ref.value = self._reference_window(optimization_reference_step, self.N)
        self._update_consistency_gate_weights()
        return control_offset

    def _compute_control_candidate_for_active_bank(self, x_current, measurement_packet, measurement_residual):
        control_offset = self._prepare_active_bank_for_measurement(measurement_packet, measurement_residual)
        if self.data_is_persistently_exciting:
            u_optimal = self.compute_optimal_control(control_offset=control_offset)
            score = float(self.problem.value) if self.problem.value is not None else float("inf")
        else:
            u_optimal = self.initial_controller.compute_input(x_current, self.ref.value[:, 0])
            score = float("inf")
        return u_optimal, score

    def collect_and_update(self, measurement_packet, u_optimal, measurement_residual):
        delivered_step = int(measurement_packet["delivered_step"])
        source_step = int(measurement_packet["source_step"])
        self.applied_inputs_by_step[delivered_step] = np.asarray(u_optimal, dtype=float).reshape(-1)
        received_pair = self._build_history_pair(
            pair_input=np.asarray(u_optimal, dtype=float).reshape(-1),
            pair_step=delivered_step,
            measurement_packet=measurement_packet,
            measurement_residual=measurement_residual,
        )
        self.received_history.append(received_pair)

        training_pair = None
        if self.history_alignment in {"time_aligned", "consistency_gated_time_aligned"}:
            if source_step in self.applied_inputs_by_step:
                training_pair = self._build_history_pair(
                    pair_input=self.applied_inputs_by_step[source_step],
                    pair_step=source_step,
                    measurement_packet=measurement_packet,
                    measurement_residual=measurement_residual,
                )
        elif self.history_alignment == "async_masked":
            self._update_async_source_slots(measurement_packet, measurement_residual)
            self._update_async_source_slot_inputs(
                source_step=source_step,
                fallback_input=np.asarray(u_optimal, dtype=float).reshape(-1),
            )
            self._append_async_training_pairs()
        elif self.history_alignment == "suffix_aligned":
            training_pair = received_pair
        else:
            training_pair = received_pair

        self._append_training_pair(training_pair)
        self._refresh_ini_windows()
        # Update the reference trajectory.
        self.shift_reference()
        
        if self.data_is_persistently_exciting:
            # Update steady state solution g_r 
            uy_r = np.hstack((np.tile(self.system.u_eq, self.T_ini),
                            np.tile(self.ref.value[:,0], self.T_ini),
                            np.tile(self.system.u_eq, self.N),
                            np.tile(self.ref.value[:,0], self.N),
                            )).reshape(-1, 1)
            
            self.g_r.value = self.Hpinv @ uy_r

    def shift_reference(self):
        self.current_reference_step += 1
        self.remaining_output_reference = self._reference_window(self.current_reference_step, self.N)
        self.ref.value = self.remaining_output_reference[:, :self.N]

    def _reference_window(self, start_step, horizon):
        start_step = max(int(start_step), 0)
        ref = self.full_output_reference[:, start_step:start_step + horizon]
        return self.trajectory.extend_reference(ref, horizon)

    def _build_history_pair(self, pair_input, pair_step, measurement_packet, measurement_residual):
        return {
            "u": np.asarray(pair_input, dtype=float).reshape(self.m),
            "y": np.asarray(measurement_packet["output"], dtype=float).reshape(self.p),
            "mask": np.asarray(measurement_packet["output_mask"], dtype=float).reshape(self.p),
            "source_steps": np.asarray(measurement_packet["output_source_steps"], dtype=float).reshape(self.p),
            "timestamps": np.asarray(measurement_packet["output_timestamps"], dtype=float).reshape(self.p),
            "measurement_residual": np.asarray(measurement_residual, dtype=float).reshape(self.p),
            "pair_step": int(pair_step),
            "source_step": int(measurement_packet["source_step"]),
            "delivered_step": int(measurement_packet["delivered_step"]),
        }

    def _normalize_measurement_packet(self, x_current, y_current):
        if y_current is None:
            delivered_y = self.system.measure_output(x_current)
            true_y = self.system.C @ x_current
            packet = {
                "output": delivered_y,
                "true_output": true_y,
                "source_step": len(self.applied_inputs_by_step),
                "delivered_step": len(self.applied_inputs_by_step),
                "delay_steps": 0,
            }
        elif isinstance(y_current, dict):
            packet = dict(y_current)
        else:
            delivered_y = np.asarray(y_current, dtype=float)
            true_y = self.system.C @ x_current
            packet = {
                "output": delivered_y,
                "true_output": true_y,
                "source_step": len(self.applied_inputs_by_step),
                "delivered_step": len(self.applied_inputs_by_step),
                "delay_steps": 0,
            }

        packet["output"] = np.asarray(packet["output"], dtype=float).reshape(self.p)
        packet["true_output"] = np.asarray(packet["true_output"], dtype=float).reshape(self.p)
        packet["source_step"] = int(packet.get("source_step", packet.get("delivered_step", len(self.applied_inputs_by_step))))
        packet["target_source_step"] = int(packet.get("target_source_step", packet["source_step"]))
        packet["delivered_step"] = int(packet.get("delivered_step", packet["source_step"]))
        packet["delay_steps"] = int(packet.get("delay_steps", max(0, packet["delivered_step"] - packet["source_step"])))
        packet["output_mask"] = np.asarray(packet.get("output_mask", np.ones(self.p)), dtype=float).reshape(self.p)
        packet["output_source_steps"] = np.asarray(
            packet.get("output_source_steps", np.full(self.p, packet["source_step"])),
            dtype=float,
        ).reshape(self.p)
        packet["output_timestamps"] = np.asarray(
            packet.get("output_timestamps", packet["output_source_steps"] * self.system.h),
            dtype=float,
        ).reshape(self.p)
        self.current_delay_steps = packet["delay_steps"]
        self.latest_measurement_metadata = {
            "source_step": packet["source_step"],
            "target_source_step": packet["target_source_step"],
            "delivered_step": packet["delivered_step"],
            "delay_steps": packet["delay_steps"],
            "output_mask": packet["output_mask"].tolist(),
            "output_source_steps": packet["output_source_steps"].astype(int).tolist(),
            "output_timestamps": packet["output_timestamps"].tolist(),
            "effective_reference_step": int(self._effective_reference_step(packet)),
        }
        return packet

    def _refresh_ini_windows(self):
        self.y_ini.value[:, :] = 0.0
        self.u_ini.value[:, :] = 0.0
        self.y_ini_mask.value[:, :] = 0.0
        self.y_ini_source_steps.value[:, :] = -1.0
        self.y_ini_timestamps.value[:, :] = -self.system.h
        history_window = self._history_window()
        if not history_window:
            return

        start_col = self.T_ini - len(history_window)
        for col, pair in enumerate(history_window, start=start_col):
            self.u_ini.value[:, col] = pair["u"]
            self.y_ini.value[:, col] = pair["y"]
            self.y_ini_mask.value[:, col] = pair["mask"]
            self.y_ini_source_steps.value[:, col] = pair["source_steps"]
            self.y_ini_timestamps.value[:, col] = pair["timestamps"]

    def _history_window(self):
        if self.history_alignment == "async_masked":
            if not self.async_source_slots:
                return []
            latest_source_step = max(self.async_source_slots)
            start_step = max(0, latest_source_step - self.T_ini + 1)
            return [
                self.async_source_slots[source_step]
                for source_step in range(start_step, latest_source_step + 1)
                if source_step in self.async_source_slots
            ]

        if self.history_alignment in {"time_aligned", "consistency_gated_time_aligned"}:
            return self.aligned_history[-self.T_ini:]

        if self.history_alignment == "suffix_aligned":
            delay_steps = min(max(int(self.current_delay_steps), 0), self.T_ini)
            prefix_len = self.T_ini - delay_steps
            if delay_steps == 0:
                return self.received_history[-self.T_ini:]

            prefix = self.received_history[-(prefix_len + delay_steps):-delay_steps]
            suffix = self.aligned_history[-delay_steps:]
            return prefix + suffix

        return self.received_history[-self.T_ini:]

    def _effective_reference_step(self, measurement_packet):
        if self.history_alignment != "async_masked":
            return int(measurement_packet["source_step"])

        observed = np.asarray(measurement_packet["output_mask"], dtype=float).reshape(self.p) > 0.5
        source_steps = np.asarray(measurement_packet["output_source_steps"], dtype=float).reshape(self.p)
        if np.any(observed):
            return int(np.max(source_steps[observed]))
        return int(np.max(source_steps))

    def _update_consistency_gate_weights(self):
        self.consistency_column_weights.value[:, :] = 0.0
        if self.history_alignment != "consistency_gated_time_aligned":
            return
        if self.Y_p.value is None:
            return

        y_ini_vector = self.y_ini.value.reshape(self.p * self.T_ini, 1, order="F")
        diffs = self.Y_p.value - y_ini_vector
        scores = np.linalg.norm(diffs, axis=0)
        score_median = float(np.median(scores))
        mad = float(np.median(np.abs(scores - score_median)))
        normalized_scores = (scores - score_median) / (mad + self.consistency_gate_eps)
        weights = np.clip(normalized_scores, 0.0, self.consistency_gate_clip)
        self.consistency_column_weights.value = weights.reshape(-1, 1)
        self.consistency_gate_summary = {
            "mode": self.history_alignment,
            "lambda_c": float(self.consistency_gate_lambda),
            "clip": float(self.consistency_gate_clip),
            "eps": float(self.consistency_gate_eps),
            "score_median": score_median,
            "score_mad": mad,
            "weight_min": float(np.min(weights)) if weights.size else 0.0,
            "weight_max": float(np.max(weights)) if weights.size else 0.0,
            "weight_mean": float(np.mean(weights)) if weights.size else 0.0,
        }

    def _append_training_pair(self, pair):
        if pair is None:
            return

        if self.data_is_persistently_exciting:
            if self._should_refresh_mature_bank():
                self._append_training_pair_to_mature_bank(pair)
            return

        self.u_d.append(pair["u"])
        self.y_d.append(pair["y"])
        self.measurement_residual_d.append(pair["measurement_residual"])

        if len(self.u_d) == self.T:
            self.u_d = np.array(self.u_d).T
            self.y_d = np.array(self.y_d).T
            self.measurement_residual_d = np.array(self.measurement_residual_d).T
            self.create_and_partition_hankel_matrices()
            self.data_is_persistently_exciting = self.is_persistently_excited_of_order_L(self.u_d, self.T_ini + self.N + self.n)

    def _should_refresh_mature_bank(self):
        if self.health_mode != "degraded":
            return False
        if self.bank_transfer_mode == "warm_start_adapt":
            return self.degraded_bank_bootstrapped
        if self.bank_transfer_mode == "adapt_only":
            return True
        return False

    def _append_training_pair_to_mature_bank(self, pair):
        u_hist = self._as_history_list(self.u_d)
        y_hist = self._as_history_list(self.y_d)
        residual_hist = self._as_history_list(self.measurement_residual_d)

        u_hist.append(np.asarray(pair["u"], dtype=float).reshape(self.m))
        y_hist.append(np.asarray(pair["y"], dtype=float).reshape(self.p))
        residual_hist.append(np.asarray(pair["measurement_residual"], dtype=float).reshape(self.p))

        if len(u_hist) > self.T:
            u_hist = u_hist[-self.T:]
            y_hist = y_hist[-self.T:]
            residual_hist = residual_hist[-self.T:]

        self.u_d = np.array(u_hist, dtype=float).T
        self.y_d = np.array(y_hist, dtype=float).T
        self.measurement_residual_d = np.array(residual_hist, dtype=float).T
        self.degraded_bank_adaptation_steps += 1
        if (
            self.degraded_bank_adaptation_steps == 1
            or self.degraded_bank_adaptation_steps % self.bank_transfer_interval_steps == 0
        ):
            self.create_and_partition_hankel_matrices()
        self.data_is_persistently_exciting = True

    def _active_bank_name(self):
        requested = getattr(self, "health_mode", "nominal")
        return requested if requested in self.bank_states else "nominal"

    def _plant_health_mode(self):
        if hasattr(self.system, "current_health_mode"):
            return self.system.current_health_mode()
        return "nominal"

    def _desired_health_mode(self):
        if self.controller_health_mode == "nominal":
            return "nominal"
        return self._plant_health_mode()

    def _select_control_bank(self, desired_health_mode, x_current=None, measurement_packet=None, measurement_residual=None):
        desired_bank = self.bank_states[desired_health_mode]
        if desired_bank["data_is_persistently_exciting"]:
            fallback_bank = desired_health_mode
        elif desired_health_mode == "degraded" and self.bank_states["nominal"]["data_is_persistently_exciting"]:
            fallback_bank = "nominal"
        else:
            fallback_bank = desired_health_mode

        self.candidate_bank_scores = {}
        if (
            self.bank_selection_mode != "oracle_minimal"
            or desired_health_mode != "degraded"
            or measurement_packet is None
            or measurement_residual is None
            or x_current is None
        ):
            return fallback_bank

        candidate_banks = [
            bank_name
            for bank_name in ("nominal", "degraded")
            if self.bank_states[bank_name]["data_is_persistently_exciting"]
        ]
        if not candidate_banks:
            return fallback_bank

        best_bank = fallback_bank
        best_score = float("inf")
        for bank_name in candidate_banks:
            self._restore_bank_state(bank_name)
            _, score = self._compute_control_candidate_for_active_bank(
                x_current,
                measurement_packet,
                measurement_residual,
            )
            self.candidate_bank_scores[bank_name] = score
            if score < best_score:
                best_score = score
                best_bank = bank_name
        return best_bank

    def _bootstrap_degraded_bank_if_needed(self, desired_health_mode):
        if self.bank_transfer_mode == "adapt_only":
            return
        if desired_health_mode != "degraded":
            return
        if self.bank_states["degraded"]["data_is_persistently_exciting"]:
            return
        if not self.bank_states["nominal"]["data_is_persistently_exciting"]:
            return
        self.bank_states["degraded"] = self._copy_bank_state(self.bank_states["nominal"])
        self.degraded_bank_bootstrapped = True
        self.bank_states["degraded"]["degraded_bank_adaptation_steps"] = 0

    def _capture_bank_state(self):
        return {
            "data_is_persistently_exciting": bool(self.data_is_persistently_exciting),
            "u_d": self._copy_bank_array_or_list(self.u_d),
            "y_d": self._copy_bank_array_or_list(self.y_d),
            "measurement_residual_d": self._copy_bank_array_or_list(self.measurement_residual_d),
            "applied_inputs_by_step": {
                int(step): np.asarray(value, dtype=float).copy()
                for step, value in self.applied_inputs_by_step.items()
            },
            "received_history": self._copy_history(self.received_history),
            "aligned_history": self._copy_history(self.aligned_history),
            "async_source_slots": {
                int(step): self._copy_history_pair(pair)
                for step, pair in self.async_source_slots.items()
            },
            "next_async_training_source_step": int(self.next_async_training_source_step),
            "latest_measurement_metadata": None if self.latest_measurement_metadata is None else dict(self.latest_measurement_metadata),
            "current_delay_steps": int(self.current_delay_steps),
            "degraded_bank_adaptation_steps": int(self.degraded_bank_adaptation_steps),
            "Hpinv": None if self.Hpinv is None else np.asarray(self.Hpinv, dtype=float).copy(),
            "U_p": np.asarray(self.U_p.value, dtype=float).copy(),
            "Y_p": np.asarray(self.Y_p.value, dtype=float).copy(),
            "U_f": np.asarray(self.U_f.value, dtype=float).copy(),
            "Y_f": np.asarray(self.Y_f.value, dtype=float).copy(),
        }

    def _as_history_list(self, value):
        if isinstance(value, list):
            return [np.asarray(item, dtype=float).copy() for item in value]
        array = np.asarray(value, dtype=float)
        if array.ndim != 2:
            raise ValueError(f"Expected 2D bank history array, got shape {array.shape}")
        return [array[:, idx].copy() for idx in range(array.shape[1])]

    def _restore_active_bank_state(self):
        self._restore_bank_state(self._active_bank_name())

    def _restore_bank_state(self, bank_name):
        bank = self.bank_states[bank_name]
        self.health_mode = bank_name
        self.data_is_persistently_exciting = bool(bank["data_is_persistently_exciting"])
        self.u_d = self._copy_bank_array_or_list(bank["u_d"])
        self.y_d = self._copy_bank_array_or_list(bank["y_d"])
        self.measurement_residual_d = self._copy_bank_array_or_list(bank["measurement_residual_d"])
        self.applied_inputs_by_step = {
            int(step): np.asarray(value, dtype=float).copy()
            for step, value in bank["applied_inputs_by_step"].items()
        }
        self.received_history = self._copy_history(bank["received_history"])
        self.aligned_history = self._copy_history(bank["aligned_history"])
        self.async_source_slots = {
            int(step): self._copy_history_pair(pair)
            for step, pair in bank["async_source_slots"].items()
        }
        self.next_async_training_source_step = int(bank["next_async_training_source_step"])
        self.latest_measurement_metadata = None if bank["latest_measurement_metadata"] is None else dict(bank["latest_measurement_metadata"])
        self.current_delay_steps = int(bank["current_delay_steps"])
        self.degraded_bank_adaptation_steps = int(bank.get("degraded_bank_adaptation_steps", 0))
        self.Hpinv = None if bank["Hpinv"] is None else np.asarray(bank["Hpinv"], dtype=float).copy()
        self.U_p.value = np.asarray(bank["U_p"], dtype=float).copy()
        self.Y_p.value = np.asarray(bank["Y_p"], dtype=float).copy()
        self.U_f.value = np.asarray(bank["U_f"], dtype=float).copy()
        self.Y_f.value = np.asarray(bank["Y_f"], dtype=float).copy()

    def _capture_active_bank_state(self):
        self._capture_bank_state_to(self._active_bank_name())

    def _capture_bank_state_to(self, bank_name):
        self.bank_states[bank_name] = self._capture_bank_state()

    def _copy_bank_array_or_list(self, value):
        if isinstance(value, list):
            return [np.asarray(item, dtype=float).copy() for item in value]
        return np.asarray(value, dtype=float).copy()

    def _copy_history_pair(self, pair):
        copied = {}
        for key, value in pair.items():
            if isinstance(value, np.ndarray):
                copied[key] = value.copy()
            else:
                copied[key] = value
        return copied

    def _copy_history(self, history):
        return [self._copy_history_pair(pair) for pair in history]

    def _copy_bank_state(self, bank_state):
        return {
            "data_is_persistently_exciting": bool(bank_state["data_is_persistently_exciting"]),
            "u_d": self._copy_bank_array_or_list(bank_state["u_d"]),
            "y_d": self._copy_bank_array_or_list(bank_state["y_d"]),
            "measurement_residual_d": self._copy_bank_array_or_list(bank_state["measurement_residual_d"]),
            "applied_inputs_by_step": {
                int(step): np.asarray(value, dtype=float).copy()
                for step, value in bank_state["applied_inputs_by_step"].items()
            },
            "received_history": self._copy_history(bank_state["received_history"]),
            "aligned_history": self._copy_history(bank_state["aligned_history"]),
            "async_source_slots": {
                int(step): self._copy_history_pair(pair)
                for step, pair in bank_state["async_source_slots"].items()
            },
            "next_async_training_source_step": int(bank_state["next_async_training_source_step"]),
            "latest_measurement_metadata": None
            if bank_state["latest_measurement_metadata"] is None
            else dict(bank_state["latest_measurement_metadata"]),
            "current_delay_steps": int(bank_state["current_delay_steps"]),
            "degraded_bank_adaptation_steps": int(bank_state.get("degraded_bank_adaptation_steps", 0)),
            "Hpinv": None if bank_state["Hpinv"] is None else np.asarray(bank_state["Hpinv"], dtype=float).copy(),
            "U_p": np.asarray(bank_state["U_p"], dtype=float).copy(),
            "Y_p": np.asarray(bank_state["Y_p"], dtype=float).copy(),
            "U_f": np.asarray(bank_state["U_f"], dtype=float).copy(),
            "Y_f": np.asarray(bank_state["Y_f"], dtype=float).copy(),
        }

    def _make_async_source_slot(self, source_step):
        previous_slot = self.async_source_slots.get(source_step - 1)
        if previous_slot is None:
            y = np.zeros(self.p)
            source_steps = -np.ones(self.p)
            timestamps = -self.system.h * np.ones(self.p)
        else:
            y = previous_slot["y"].copy()
            source_steps = previous_slot["source_steps"].copy()
            timestamps = previous_slot["timestamps"].copy()

        return {
            "u": None,
            "y": y,
            "mask": np.zeros(self.p),
            "source_steps": source_steps,
            "timestamps": timestamps,
            "measurement_residual": np.zeros(self.p),
            "pair_step": int(source_step),
            "source_step": int(source_step),
            "delivered_step": int(source_step),
        }

    def _update_async_source_slots(self, measurement_packet, measurement_residual):
        output_source_steps = np.asarray(measurement_packet["output_source_steps"], dtype=int).reshape(self.p)
        output_mask = np.asarray(measurement_packet["output_mask"], dtype=float).reshape(self.p)
        output_timestamps = np.asarray(measurement_packet["output_timestamps"], dtype=float).reshape(self.p)
        delivered_output = np.asarray(measurement_packet["output"], dtype=float).reshape(self.p)
        measurement_residual = np.asarray(measurement_residual, dtype=float).reshape(self.p)
        delivered_step = int(measurement_packet["delivered_step"])
        target_source_step = int(measurement_packet.get("target_source_step", measurement_packet["source_step"]))

        unique_source_steps = sorted({int(step) for step in output_source_steps if step >= 0})
        if target_source_step >= 0:
            unique_source_steps = sorted(set(unique_source_steps + [target_source_step]))
        for source_step in unique_source_steps:
            if source_step not in self.async_source_slots:
                self.async_source_slots[source_step] = self._make_async_source_slot(source_step)

            slot = self.async_source_slots[source_step]
            slot["delivered_step"] = delivered_step
            for row in range(self.p):
                if output_mask[row] < 0.5:
                    continue
                if int(output_source_steps[row]) != source_step:
                    continue
                slot["y"][row] = delivered_output[row]
                slot["mask"][row] = 1.0
                slot["source_steps"][row] = float(output_source_steps[row])
                slot["timestamps"][row] = output_timestamps[row]
                slot["measurement_residual"][row] = measurement_residual[row]

        self._update_async_source_slot_inputs()

    def _update_async_source_slot_inputs(self, source_step=None, fallback_input=None):
        if source_step is not None and source_step in self.async_source_slots:
            slot = self.async_source_slots[source_step]
            if source_step in self.applied_inputs_by_step:
                slot["u"] = np.asarray(self.applied_inputs_by_step[source_step], dtype=float).reshape(self.m)
            elif fallback_input is not None:
                slot["u"] = np.asarray(fallback_input, dtype=float).reshape(self.m)

        for tracked_source_step, slot in self.async_source_slots.items():
            if slot["u"] is None and tracked_source_step in self.applied_inputs_by_step:
                slot["u"] = np.asarray(self.applied_inputs_by_step[tracked_source_step], dtype=float).reshape(self.m)

    def _append_async_training_pairs(self):
        while not self.data_is_persistently_exciting:
            source_step = self.next_async_training_source_step
            if source_step not in self.async_source_slots:
                break
            slot = self.async_source_slots[source_step]
            if slot["u"] is None:
                break
            self._append_training_pair(slot)
            self.next_async_training_source_step += 1
        
    def is_persistently_excited_of_order_L(self, x, L):
        H = self.construct_hankel_matrix(x, L)
        
        if np.linalg.matrix_rank(H) == L*self.m:
            print(f"Data is persistently exciting of order {L}!")
            return True
        else:
            raise ValueError(f"Data is not persistently exciting, increase the amount of datapoints used in Hankelmatrix.")
