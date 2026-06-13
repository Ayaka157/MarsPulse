"""
MarsPulse — 火星基地仿真配置参数
====================================
模块说明: 定义火星熔岩管基地的所有物理和场景参数。
         所有物理常数采用SI单位制，参数取值依据NASA标准及已发表文献。
作者: 曾皓轩
开发时间: 2026年3–5月
所属项目: MarsPulse — 基于MARL的火星地下熔岩管基地资源智能调配与生存极限仿真
"""

from dataclasses import dataclass, field
from typing import Tuple, List


@dataclass
class MarsBaseConfig:
    """Configuration for a realistic medium-scale Mars lava tube base."""

    # --- Time ---
    dt: float = 0.05              # simulation timestep (Mars days), ~1.2 Earth hours
    episode_length: int = 300     # max episode length in Mars days (sols) — ~10 months

    # --- Solar Power (150 kW array, medium-scale base) ---
    P_max: float = 150.0          # max solar panel output at Mars surface (kW)
    solar_efficiency: float = 0.85  # panel efficiency factor
    tau_clear: float = 0.15       # optical depth during clear sky
    tau_storm: float = 2.5        # optical depth during severe dust storm (→ 8.2% transmission)
    theta_avg: float = 0.0        # average solar incidence angle (radians)
    P_rtg: float = 10.0           # RTG baseline power (kW, always available)

    # --- Battery / Energy Storage (MWh scale) ---
    battery_capacity: float = 3000.0   # kWh (Mars base battery bank — sized for storm challenge)
    battery_max_charge_rate: float = 80.0   # kW
    battery_max_discharge_rate: float = 80.0  # kW
    battery_initial_frac: float = 0.80  # start at 80% charge
    battery_min_safe_frac: float = 0.05

    # --- Oxygen System ---
    O2_storage_capacity: float = 500.0   # kg of O2 (scaled for larger base)
    O2_initial_frac: float = 0.90
    O2_consumption_per_person: float = 0.84  # kg/sol/person (NASA baseline)
    O2_electrolysis_efficiency: float = 0.15  # kg O2 per kWh
    O2_min_safe_frac: float = 0.05
    crew_size: int = 6

    # --- Thermal System ---
    lava_tube_temp: float = -20.0    # °C, ambient rock temperature
    target_habitat_temp: float = 20.0  # °C
    habitat_heat_capacity: float = 50000.0  # kJ/°C
    habitat_heat_loss_coeff: float = 0.18  # kW/°C (with radiator-assisted heat rejection)
    heater_max_power: float = 15.0   # kW
    heater_efficiency: float = 0.90
    temp_min_survivable: float = 5.0
    temp_max_survivable: float = 35.0

    # --- Equipment Power (nominal, scaled for medium base) ---
    power_life_support_base: float = 12.0    # kW (O2 gen, water recycling, CO2 scrub)
    power_experiments: float = 18.0          # kW (science instruments)
    power_lighting_misc: float = 4.0         # kW
    power_communication: float = 3.0         # kW

    # --- Dust Storm Scenario (extended major storm) ---
    storm_start_day: float = 60.0    # sol when major dust storm begins
    storm_duration: float = 60.0     # sols — severe 2-month storm
    storm_tau_peak: float = 3.5      # peak optical depth (config reference)
    storm_power_frac: float = 0.12   # power drops to ~12% at peak (config reference)

    # --- Second storm (minor, later in year) ---
    storm2_start_day: float = 200.0
    storm2_duration: float = 35.0
    storm2_tau: float = 2.2          # moderate-severe optical depth (→ 11.1% transmission)

    # --- Storm definitions list (structured, extensible) ---
    # Each storm: (start_day, duration, tau_peak)
    # This is the canonical storm list used by physics.py.
    # Add more tuples here for additional storms; no code changes needed elsewhere.
    storm_list: Tuple[Tuple[float, float, float], ...] = field(default_factory=lambda: (
        (60.0, 60.0, 2.5),     # Storm 1: major, sol 60-120
        (200.0, 35.0, 2.2),    # Storm 2: moderate, sol 200-235
    ))

    # --- Survival Stability Weights ---
    w_energy: float = 0.40
    w_oxygen: float = 0.35
    w_thermal: float = 0.25
    S_critical: float = 0.15
    S_fatal: float = 0.05

    # --- Random Failure Events ---
    failure_prob_per_sol: float = 0.01   # 1% per sol base rate
    failure_modes: Tuple[str, ...] = (
        "electrolysis_degraded",   # O2 generation drops 50%
        "heater_fault",            # heater output drops 60%
        "battery_cell_failure",    # capacity drops 15%
        "habitat_leak",            # heat loss doubles
    )
