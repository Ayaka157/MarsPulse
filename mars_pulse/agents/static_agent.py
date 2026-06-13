"""
MarsPulse — 静态规则基线智能体
===============================
模块说明: 模拟传统航天器资源管理的固定优先级规则预案。
         含StaticRuleAgent(标准静态规则)和StaticRuleAgentConservative(保守变体)。
         优先级: 生命保障 > 热控 > 实验。含过热保护(T>26°C/T>30°C减载)。
         作为MARL方法的对比基线，验证AI自主决策的优势。
作者: 曾皓轩
开发时间: 2026年4月
所属项目: MarsPulse — 基于MARL的火星地下熔岩管基地资源智能调配与生存极限仿真
"""

import numpy as np
from ..environment import MarsBaseEnv


def _current_temp(obs: np.ndarray, cfg) -> float:
    """Decode habitat temperature from normalised observation."""
    return float(obs[2]) * (cfg.temp_max_survivable - cfg.temp_min_survivable) + cfg.temp_min_survivable


def _thermostat(obs: np.ndarray, cfg, max_power: float) -> float:
    """Simple proportional thermostat: heat only when below target."""
    T = _current_temp(obs, cfg)
    error = cfg.target_habitat_temp - T
    if error <= 0:
        return 0.0
    # P-controller: ~0.5 kW per °C below target, clamped
    heating = min(error * 0.5, max_power)
    return max(heating, 0.0)


class StaticRuleAgent:
    """Rule-based resource manager with fixed allocation priorities.

    Priority order: life support > heating > experiments
    During storm: shed experiments first, then reduce heating to minimum safe level.
    """

    def __init__(self, env: MarsBaseEnv):
        self.env = env
        self.cfg = env.cfg

    def select_action(self, obs: np.ndarray) -> np.ndarray:
        energy_frac = float(obs[0])
        O2_frac = float(obs[1])
        storm_active = bool(obs[3] > 0.5)

        # Thermostat-controlled heating
        heating = _thermostat(obs, self.cfg, self.cfg.heater_max_power)

        if storm_active:
            life_support = self.cfg.power_life_support_base * 1.1
            experiments = 2.0
        else:
            life_support = self.cfg.power_life_support_base
            experiments = self.cfg.power_experiments

        # Emergency curtailment if energy critically low
        if energy_frac < 0.15:
            experiments = 0.0
            life_support = self.cfg.power_life_support_base * 0.85
        if energy_frac < 0.08:
            life_support = self.cfg.power_life_support_base * 0.60

        # If O2 is low, boost life support
        if O2_frac < 0.20:
            life_support = min(life_support * 1.5, 25.0)
            experiments = 0.0

        # Overheating protection: shed heat-generating loads
        T = _current_temp(obs, self.cfg)
        if T > 30.0:
            experiments = 0.0
            life_support = min(life_support, self.cfg.power_life_support_base * 0.7)
        elif T > 26.0:
            experiments = min(experiments, 2.0)

        return np.array([life_support, heating, experiments], dtype=np.float32)


class StaticRuleAgentConservative:
    """More conservative variant — maintains higher resource margins, sheds load early."""

    def __init__(self, env: MarsBaseEnv):
        self.env = env
        self.cfg = env.cfg

    def select_action(self, obs: np.ndarray) -> np.ndarray:
        energy_frac = float(obs[0])
        O2_frac = float(obs[1])
        storm_active = bool(obs[3] > 0.5)
        any_failure = np.any(obs[5:] > 0.5)

        heating = _thermostat(obs, self.cfg, self.cfg.heater_max_power)

        if storm_active or any_failure:
            life_support = self.cfg.power_life_support_base * 1.05
            experiments = 0.0
        else:
            life_support = self.cfg.power_life_support_base
            experiments = 8.0

        # Graduated curtailment based on energy margin
        if energy_frac < 0.30:
            experiments = 0.0
        if energy_frac < 0.15:
            life_support = self.cfg.power_life_support_base * 0.80
        if energy_frac < 0.08:
            life_support = self.cfg.power_life_support_base * 0.55

        # O2 boost
        if O2_frac < 0.25:
            life_support = min(life_support * 1.5, 22.0)
            experiments = 0.0

        # Overheating protection
        T = _current_temp(obs, self.cfg)
        if T > 30.0:
            experiments = 0.0
            life_support = min(life_support, self.cfg.power_life_support_base * 0.65)
        elif T > 26.0:
            experiments = 0.0

        return np.array([life_support, heating, experiments], dtype=np.float32)
