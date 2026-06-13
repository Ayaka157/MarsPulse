"""
MarsPulse — 物理引擎
=====================
模块说明: 火星熔岩管基地核心物理仿真。
         实现能量平衡、氧气质量平衡、热动力学、太阳电力输入(比尔-朗伯消光定律)、
         随机故障及生存稳定度指数S(t)的计算。
         采用预计算常量优化，步进耗时约42微秒/步。
作者: 曾皓轩
开发时间: 2026年3–5月
所属项目: MarsPulse — 基于MARL的火星地下熔岩管基地资源智能调配与生存极限仿真
"""

import numpy as np
from .config import MarsBaseConfig


class MarsPhysics:
    """Core physical simulation of the Mars lava tube base."""

    def __init__(self, config: MarsBaseConfig = None, rng: np.random.Generator = None):
        self.cfg = config or MarsBaseConfig()
        self.rng = rng or np.random.default_rng()

        # --- Pre-computed constants (avoids repeated arithmetic in hot loop) ---
        self._precompute()

        # --- State variables ---
        self.time: float = 0.0
        self.energy_stored: float = 0.0
        self.oxygen_stored: float = 0.0
        self.habitat_temp: float = 0.0
        self.power_in: float = 0.0
        self.tau: float = 0.0
        self.power_consumed: float = 0.0
        self.O2_production_rate: float = 0.0
        self.heater_power: float = 0.0
        self.crew_alive: bool = True
        self.S: float = 1.0

        self.active_failures: dict = {}

        self.reset()

    def _precompute(self):
        """Pre-compute derived constants for fast stepping."""
        c = self.cfg
        self._dt_hours = c.dt * 24.65           # sols → hours
        self._dt_thermal = c.dt * 24.65 * 3600.0 / c.habitat_heat_capacity  # °C per kW
        self._P_solar_max = c.P_max * c.solar_efficiency  # max solar kW
        self._cos_theta = max(np.cos(c.theta_avg), 0.05)
        self._tau_delta = c.tau_storm - c.tau_clear

    def reset(self, rng: np.random.Generator = None):
        if rng is not None:
            self.rng = rng
        self.time = 0.0
        self.energy_stored = self.cfg.battery_capacity * self.cfg.battery_initial_frac
        self.oxygen_stored = self.cfg.O2_storage_capacity * self.cfg.O2_initial_frac
        self.habitat_temp = self.cfg.target_habitat_temp
        self.tau = self.cfg.tau_clear
        self.active_failures = {k: False for k in self.cfg.failure_modes}
        self.crew_alive = True
        self.S = 1.0
        self._update_power_in()
        self.power_consumed = 0.0

    # ------------------------------------------------------------------
    # Solar power model: P_in(t) = P_max * exp(-tau(t)) * cos(theta)
    # ------------------------------------------------------------------
    def _update_power_in(self):
        theta = self.cfg.theta_avg  # simplified daily average
        self.tau = self._get_storm_tau()

        # Solar contribution (using pre-computed constants)
        solar_power = self._P_solar_max * np.exp(-self.tau) * self._cos_theta
        self.power_in = solar_power + self.cfg.P_rtg

    def is_storm_active(self) -> bool:
        """Check if ANY storm is currently active.

        Uses cfg.storm_list: iterable of (start_day, duration, tau_peak) tuples.
        Add more storms to storm_list in config.py — no code changes needed here.
        """
        for start_day, duration, _ in self.cfg.storm_list:
            if start_day <= self.time < start_day + duration:
                return True
        return False

    def _get_storm_tau(self) -> float:
        """Compute tau for the currently active storm (or clear sky).

        Uses cfg.storm_list with a sin^0.8 profile for realistic onset/decay.
        If multiple storms overlap, returns the tau of the first active one.
        """
        for start_day, duration, tau_peak in self.cfg.storm_list:
            if start_day <= self.time < start_day + duration:
                elapsed = self.time - start_day
                phase = elapsed / duration
                storm_factor = np.sin(np.pi * phase) ** 0.8
                return self.cfg.tau_clear + storm_factor * (tau_peak - self.cfg.tau_clear)

        return self.cfg.tau_clear

    # ------------------------------------------------------------------
    # Energy step
    # ------------------------------------------------------------------
    def step_energy(self, power_draw: float) -> float:
        """Step the energy system. Returns actual power supplied (may be curtailed)."""
        self._update_power_in()

        # Check failures
        curtailed = power_draw
        if self.active_failures.get("battery_cell_failure", False):
            usable_capacity = self.cfg.battery_capacity * 0.85
        else:
            usable_capacity = self.cfg.battery_capacity

        # Net power
        net = self.power_in - power_draw  # kW
        # Energy change per timestep
        dE = net * self.cfg.dt  # kWh (since dt is in sols and power is kW, need scaling)
        # Convert: 1 sol = 24.65 hours, so kW * sol = kWh * 24.65
        # Actually: power_draw is in kW, dt is in sols.
        # 1 sol ≈ 24.65 hours, so energy in kWh = power_kW * dt_sol * 24.65
        dE = net * self._dt_hours

        self.energy_stored += dE

        # Clamp to [0, capacity]
        self.energy_stored = np.clip(self.energy_stored, 0.0, usable_capacity)

        # If energy hits zero, curtail non-essential loads
        if self.energy_stored <= 0.0:
            curtailed = max(power_draw * 0.3, self.power_in)  # only what's coming in
            self.energy_stored = 0.0

        self.power_consumed = curtailed
        return curtailed

    # ------------------------------------------------------------------
    # Oxygen step
    # ------------------------------------------------------------------
    def step_oxygen(self, O2_gen_power: float):
        """Step the oxygen system.
        O2_gen_power: kW allocated to electrolysis.
        """
        # Crew consumption
        O2_consumed = self.cfg.O2_consumption_per_person * self.cfg.crew_size * self.cfg.dt

        # Electrolysis production
        eff = self.cfg.O2_electrolysis_efficiency
        if self.active_failures.get("electrolysis_degraded", False):
            eff *= 0.50
        O2_produced = O2_gen_power * self._dt_hours * eff
        self.O2_production_rate = O2_produced / max(self.cfg.dt, 1e-9)

        self.oxygen_stored += O2_produced - O2_consumed
        self.oxygen_stored = np.clip(self.oxygen_stored, 0.0, self.cfg.O2_storage_capacity)

    # ------------------------------------------------------------------
    # Thermal step
    # ------------------------------------------------------------------
    def step_thermal(self, heater_power: float):
        """Step the thermal system.
        heater_power: kW allocated to heaters.
        """
        eff = self.cfg.heater_efficiency
        if self.active_failures.get("heater_fault", False):
            eff *= 0.40

        heat_loss_coeff = self.cfg.habitat_heat_loss_coeff
        if self.active_failures.get("habitat_leak", False):
            heat_loss_coeff *= 2.0

        # Heat gain from heater + internal equipment
        Q_in = heater_power * eff + self.power_consumed * 0.15  # 15% of consumed power → waste heat
        # Heat loss to lava tube rock
        Q_out = heat_loss_coeff * (self.habitat_temp - self.cfg.lava_tube_temp)

        dT = (Q_in - Q_out) * self._dt_thermal
        self.habitat_temp += dT
        self.heater_power = heater_power

    # ------------------------------------------------------------------
    # Survival Stability Index
    # ------------------------------------------------------------------
    def compute_S(self) -> float:
        """S(t) = w1*E(t) + w2*O(t) + w3*T(t) — each term in [0, 1]."""
        # Energy term
        E_norm = np.clip(self.energy_stored / self.cfg.battery_capacity, 0.0, 1.0)

        # Oxygen term
        O_norm = np.clip(self.oxygen_stored / self.cfg.O2_storage_capacity, 0.0, 1.0)

        # Thermal term: 1.0 at target, drops to 0 at survivable limits
        if self.habitat_temp >= self.cfg.target_habitat_temp:
            T_dev = (self.habitat_temp - self.cfg.target_habitat_temp) / (
                self.cfg.temp_max_survivable - self.cfg.target_habitat_temp
            )
        else:
            T_dev = (self.cfg.target_habitat_temp - self.habitat_temp) / (
                self.cfg.target_habitat_temp - self.cfg.temp_min_survivable
            )
        T_norm = np.clip(1.0 - T_dev, 0.0, 1.0)

        self.S = (
            self.cfg.w_energy * E_norm
            + self.cfg.w_oxygen * O_norm
            + self.cfg.w_thermal * T_norm
        )
        return self.S

    # ------------------------------------------------------------------
    # Random failures
    # ------------------------------------------------------------------
    def check_random_failures(self):
        """Randomly trigger component failures each timestep.
        Failure probability increases 3x during dust storms (compound risk).
        """
        storm_mult = 3.0 if self.is_storm_active() else 1.0
        prob_per_step = self.cfg.failure_prob_per_sol * self.cfg.dt * storm_mult
        for mode in self.cfg.failure_modes:
            if self.rng.random() < prob_per_step:
                self.active_failures[mode] = True

            # Small chance of auto-recovery (crew repair)
            if self.active_failures[mode] and self.rng.random() < prob_per_step * 0.5:
                self.active_failures[mode] = False

    # ------------------------------------------------------------------
    # Full step
    # ------------------------------------------------------------------
    def step(self, power_draw: float, O2_gen_power: float, heater_power: float) -> dict:
        """Advance the simulation by one timestep."""
        self.check_random_failures()
        actual_power = self.step_energy(power_draw)
        self.step_oxygen(O2_gen_power)
        self.step_thermal(heater_power)
        self.compute_S()

        self.time += self.cfg.dt

        # Check fatality
        if self.S <= self.cfg.S_fatal and self.crew_alive:
            # S below fatal threshold for sustained period — check if resources are truly exhausted
            if self.energy_stored <= 0.0 or self.oxygen_stored <= 0.0:
                self.crew_alive = False
        if (self.habitat_temp < self.cfg.temp_min_survivable
                or self.habitat_temp > self.cfg.temp_max_survivable):
            self.crew_alive = False

        return self.get_state()

    def get_state(self) -> dict:
        return {
            "time": self.time,
            "energy_stored": self.energy_stored,
            "oxygen_stored": self.oxygen_stored,
            "habitat_temp": self.habitat_temp,
            "power_in": self.power_in,
            "tau": self.tau,
            "S": self.S,
            "crew_alive": self.crew_alive,
            "storm_active": self.is_storm_active(),
            "failures": dict(self.active_failures),
        }
