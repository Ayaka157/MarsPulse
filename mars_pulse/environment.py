"""
MarsPulse — 单智能体Gymnasium环境
==================================
模块说明: 符合Gymnasium接口的强化学习环境。
         将物理引擎封装为标准RL环境，提供单智能体观察/动作空间。
         同时提供get_multi_agent_observations()方法供MARL评估使用。
作者: 曾皓轩
开发时间: 2026年4–5月
所属项目: MarsPulse — 基于MARL的火星地下熔岩管基地资源智能调配与生存极限仿真
"""

from typing import Optional, Tuple, Dict, Any
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .config import MarsBaseConfig
from .physics import MarsPhysics


class MarsBaseEnv(gym.Env):
    """Mars lava tube base survival environment.

    Observation space: [energy_frac, O2_frac, temp_norm, storm_intensity, failure_flags...]
    Action space: continuous — [power_life_support, power_heating, power_experiments] in kW

    The agent controls power allocation across subsystems to maximise survival time.
    """

    metadata = {"render_modes": ["human", "none"]}

    def __init__(self, config: MarsBaseConfig = None, render_mode: str = "none",
                 seed: int = None):
        super().__init__()
        self.cfg = config or MarsBaseConfig()
        self.rng = np.random.default_rng(seed)
        self.physics = MarsPhysics(self.cfg, self.rng)
        self.render_mode = render_mode
        self.step_count = 0
        self.max_steps = int(self.cfg.episode_length / self.cfg.dt)

        # Observation: 3 main resources + storm info + 4 failure flags
        obs_dim = 3 + 2 + len(self.cfg.failure_modes)
        self.observation_space = spaces.Box(
            low=-1.0, high=2.0, shape=(obs_dim,), dtype=np.float32
        )

        # Action: [power_life_support, power_heating, power_experiments] in kW
        self.action_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([30.0, 30.0, 25.0], dtype=np.float32),
            dtype=np.float32,
        )

    def reset(self, seed: int = None, **kwargs) -> Tuple[np.ndarray, dict]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.physics.reset(self.rng)
        self.step_count = 0
        return self._get_obs(), {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        life_support_pwr, heater_pwr, experiment_pwr = action

        total_draw = (
            life_support_pwr
            + heater_pwr
            + experiment_pwr
            + self.cfg.power_lighting_misc
            + self.cfg.power_communication
        )

        state = self.physics.step(
            power_draw=total_draw,
            O2_gen_power=life_support_pwr * 0.6,  # 60% of LS power → electrolysis
            heater_power=heater_pwr,
        )
        self.step_count += 1

        terminated = not state["crew_alive"]
        truncated = self.step_count >= self.max_steps

        reward = self._compute_reward(state, terminated, action)

        info = {
            "S": state["S"],
            "energy_stored": state["energy_stored"],
            "oxygen_stored": state["oxygen_stored"],
            "habitat_temp": state["habitat_temp"],
            "survival_sols": state["time"],
            "storm_active": state["storm_active"],
            "failures": state["failures"],
        }

        return self._get_obs(), reward, terminated, truncated, info

    def get_multi_agent_observations(self) -> list:
        """Build per-agent observations compatible with MultiAgentMarsEnv.

        Returns three observations for: Power Manager, Thermal Controller,
        and Science Manager agents. This enables MAPPO evaluation directly
        from a single-agent environment wrapper.
        """
        s = self.physics
        cfg = self.cfg
        failures = [float(s.active_failures.get(m, False)) for m in cfg.failure_modes]

        # Agent 0 (Power Manager): energy, O2, storm, tau + failures
        obs0 = np.array([
            s.energy_stored / cfg.battery_capacity,
            s.oxygen_stored / cfg.O2_storage_capacity,
            float(s.is_storm_active()),
            s.tau / cfg.tau_storm,
        ] + failures, dtype=np.float32)

        # Agent 1 (Thermal Controller): temp, energy, storm + failures
        obs1 = np.array([
            (s.habitat_temp - cfg.temp_min_survivable)
            / (cfg.temp_max_survivable - cfg.temp_min_survivable),
            s.energy_stored / cfg.battery_capacity,
            float(s.is_storm_active()),
        ] + failures, dtype=np.float32)

        # Agent 2 (Science Manager): energy, storm, O2 + failures
        obs2 = np.array([
            s.energy_stored / cfg.battery_capacity,
            float(s.is_storm_active()),
            s.oxygen_stored / cfg.O2_storage_capacity,
        ] + failures, dtype=np.float32)

        return [obs0, obs1, obs2]

    def _get_obs(self) -> np.ndarray:
        s = self.physics
        cfg = self.cfg
        obs = np.array([
            s.energy_stored / cfg.battery_capacity,
            s.oxygen_stored / cfg.O2_storage_capacity,
            (s.habitat_temp - cfg.temp_min_survivable)
            / (cfg.temp_max_survivable - cfg.temp_min_survivable),
            float(s.is_storm_active()),
            s.tau / cfg.tau_storm,  # storm intensity normalised
        ] + [float(s.active_failures.get(m, False)) for m in cfg.failure_modes],
            dtype=np.float32,
        )
        return obs

    def _compute_reward(self, state: dict, terminated: bool,
                        action: np.ndarray = None) -> float:
        """Reward with illegal-action penalty for robust training."""
        if terminated:
            return -500.0

        reward = 1.0 + state["S"] * 2.0

        # --- Illegal action penalty ---
        if action is not None:
            bounds = [
                (0.0, 30.0),   # life support (kW)
                (0.0, 20.0),   # heating (kW)
                (0.0, 25.0),   # experiments (kW)
            ]
            for i, (lo, hi) in enumerate(bounds):
                val = float(action[i])
                if val < lo:
                    reward -= 2.0 * (lo - val)
                elif val > hi:
                    reward -= 2.0 * (val - hi)

        # Storm incentives
        if state["storm_active"]:
            cfg_pwr = (self.cfg.power_life_support_base +
                       self.cfg.power_lighting_misc +
                       self.cfg.power_communication)
            if action is not None:
                total_pwr = sum(max(0.0, float(action[i])) for i in range(3))
                if total_pwr < cfg_pwr + 5.0:
                    reward += 2.0

        # Critical resource penalties
        E_frac = state["energy_stored"] / self.cfg.battery_capacity
        O_frac = state["oxygen_stored"] / self.cfg.O2_storage_capacity
        if E_frac < 0.15:
            reward -= 2.0
        if O_frac < 0.15:
            reward -= 2.0

        T = state["habitat_temp"]
        if T < self.cfg.temp_min_survivable + 3 or T > self.cfg.temp_max_survivable - 3:
            reward -= 2.0

        return reward
