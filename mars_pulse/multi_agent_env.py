"""
MarsPulse — 多智能体协作环境
=============================
模块说明: 三智能体MARL环境封装。
         电力管理Agent(生命保障功率)、热控管理Agent(供暖功率)、
         科学管理Agent(实验功率)共享协作奖励。
         提供中心化评论家所需的全局状态和联合动作接口。
作者: 曾皓轩
开发时间: 2026年4–5月
所属项目: MarsPulse — 基于MARL的火星地下熔岩管基地资源智能调配与生存极限仿真
"""

from typing import Tuple, Dict, List, Optional
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .config import MarsBaseConfig
from .physics import MarsPhysics


class MultiAgentMarsEnv:
    """Multi-agent wrapper around the Mars base physics simulation.

    Each agent has its own observation and action space.
    The centralized critic sees the concatenated global state.
    """

    def __init__(self, config: MarsBaseConfig = None, seed: int = None):
        self.cfg = config or MarsBaseConfig()
        self.rng = np.random.default_rng(seed)
        self.physics = MarsPhysics(self.cfg, self.rng)
        self.max_steps = int(self.cfg.episode_length / self.cfg.dt)
        self.step_count = 0
        self.n_agents = 3

        # --- Agent definitions ---
        # Agent 0: Power Manager — controls life_support_power (kW)
        # Agent 1: Thermal Controller — controls heater_power (kW)
        # Agent 2: Science Manager — controls experiment_power (kW)

        n_failures = len(self.cfg.failure_modes)

        # Each agent gets a tailored observation
        self.obs_dims = [
            4 + n_failures,  # Power: energy, O2, storm, tau + failures
            3 + n_failures,  # Thermal: temp, energy, storm + failures
            3 + n_failures,  # Science: energy, storm, O2 + failures
        ]

        self.observation_spaces = [
            spaces.Box(low=-1, high=2, shape=(d,), dtype=np.float32)
            for d in self.obs_dims
        ]

        # Action spaces (scalar per agent)
        self.action_spaces = [
            spaces.Box(low=np.array([0.0]), high=np.array([25.0]), dtype=np.float32),  # life support
            spaces.Box(low=np.array([0.0]), high=np.array([10.0]), dtype=np.float32),  # heater
            spaces.Box(low=np.array([0.0]), high=np.array([20.0]), dtype=np.float32),  # experiments
        ]

        # Global state for critic: all obs concatenated + all actions
        self.global_obs_dim = sum(self.obs_dims)
        self.global_act_dim = self.n_agents  # each agent has scalar action

        # Action bounds for each agent (used by MAPPO / illegal-action penalty)
        self.action_bounds = [
            (0.0, 30.0),   # Power / Life Support (kW)
            (0.0, 20.0),   # Thermal / Heating (kW)
            (0.0, 25.0),   # Science / Experiments (kW)
        ]

    def reset(self, seed: int = None) -> Tuple[List[np.ndarray], dict]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.physics.reset(self.rng)
        self.step_count = 0
        obs = self._get_observations()
        return obs, {}

    def step(self, actions: List[np.ndarray]) -> Tuple[
        List[np.ndarray], List[float], bool, bool, dict
    ]:
        """Execute one step with actions from all agents."""
        life_support_pwr = float(actions[0][0])
        heater_pwr = float(actions[1][0])
        experiment_pwr = float(actions[2][0])

        # Total power draw
        total_draw = (
            life_support_pwr
            + heater_pwr
            + experiment_pwr
            + self.cfg.power_lighting_misc
            + self.cfg.power_communication
        )

        state = self.physics.step(
            power_draw=total_draw,
            O2_gen_power=life_support_pwr * 0.6,
            heater_power=heater_pwr,
        )
        self.step_count += 1

        terminated = not state["crew_alive"]
        truncated = self.step_count >= self.max_steps

        # Shared cooperative reward
        reward = self._compute_reward(state, terminated, actions)

        obs = self._get_observations()
        rewards = [reward] * self.n_agents  # all agents share the same reward

        info = {
            "S": state["S"],
            "energy_stored": state["energy_stored"],
            "oxygen_stored": state["oxygen_stored"],
            "habitat_temp": state["habitat_temp"],
            "survival_sols": state["time"],
            "storm_active": state["storm_active"],
            "failures": state["failures"],
            "actions": [life_support_pwr, heater_pwr, experiment_pwr],
        }

        return obs, rewards, terminated, truncated, info

    def _get_observations(self) -> List[np.ndarray]:
        """Build per-agent observations."""
        s = self.physics
        cfg = self.cfg
        failures = [float(s.active_failures.get(m, False)) for m in cfg.failure_modes]

        # Agent 0: Power Manager — sees energy, O2, storm status, failures
        obs0 = np.array([
            s.energy_stored / cfg.battery_capacity,
            s.oxygen_stored / cfg.O2_storage_capacity,
            float(s.is_storm_active()),
            s.tau / cfg.tau_storm,
        ] + failures, dtype=np.float32)

        # Agent 1: Thermal Controller — sees temp, energy, storm, failures
        obs1 = np.array([
            (s.habitat_temp - cfg.temp_min_survivable)
            / (cfg.temp_max_survivable - cfg.temp_min_survivable),
            s.energy_stored / cfg.battery_capacity,
            float(s.is_storm_active()),
        ] + failures, dtype=np.float32)

        # Agent 2: Science Manager — sees energy, storm, O2, failures
        obs2 = np.array([
            s.energy_stored / cfg.battery_capacity,
            float(s.is_storm_active()),
            s.oxygen_stored / cfg.O2_storage_capacity,
        ] + failures, dtype=np.float32)

        return [obs0, obs1, obs2]

    def get_global_obs(self) -> np.ndarray:
        """Concatenated observation for centralized critic."""
        obs_list = self._get_observations()
        return np.concatenate(obs_list).astype(np.float32)

    def _compute_reward(
        self, state: dict, terminated: bool, actions: List[np.ndarray]
    ) -> float:
        """Shared cooperative reward with illegal-action penalty.

        Penalises agents for attempting actions outside their legal range
        (before clamping), encouraging natural boundary compliance.
        """
        if terminated:
            return -500.0  # heavy penalty for crew fatality

        reward = 1.0  # base survival reward per step

        # S(t) bonus — healthy resource margins
        reward += state["S"] * 2.0

        # --- Illegal action penalty ---
        # Agent 0 (Power/Life Support): legal range [0, 30] kW
        # Agent 1 (Thermal/Heating): legal range [0, 20] kW
        # Agent 2 (Science/Experiments): legal range [0, 25] kW
        action_bounds = [(0.0, 30.0), (0.0, 20.0), (0.0, 25.0)]
        for i, (a, (lo, hi)) in enumerate(zip(actions, action_bounds)):
            val = float(a[0])
            if val < lo:
                reward -= 2.0 * (lo - val)  # penalty proportional to violation
            elif val > hi:
                reward -= 2.0 * (val - hi)

        # Storm-specific incentives
        if state["storm_active"]:
            # Reward power conservation during storm
            total_pwr = sum(max(0.0, float(a[0])) for a in actions)
            cfg_pwr = (self.cfg.power_life_support_base +
                       self.cfg.power_lighting_misc +
                       self.cfg.power_communication)
            if total_pwr < cfg_pwr + 5.0:  # experiments fully shed
                reward += 2.0
            # Penalise energy depletion rate
            E_frac = state["energy_stored"] / self.cfg.battery_capacity
            if E_frac < 0.30:
                reward -= 1.0

        # Critical resource penalties
        E_frac = state["energy_stored"] / self.cfg.battery_capacity
        O_frac = state["oxygen_stored"] / self.cfg.O2_storage_capacity
        if E_frac < 0.15:
            reward -= 2.0
        if O_frac < 0.15:
            reward -= 2.0

        # Thermal safety margin
        T = state["habitat_temp"]
        if T < self.cfg.temp_min_survivable + 3 or T > self.cfg.temp_max_survivable - 3:
            reward -= 2.0

        return reward
