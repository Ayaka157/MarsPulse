"""
MarsPulse — 步骤1：静态基线测试
===============================
模块说明: 运行蒙特卡洛模拟对比随机智能体、静态规则和静态保守三种基线。
         输出step1_results.json(聚合统计)和step1_detail.json(详细轨迹)。
         评估指标: 生存时间、死亡率、完整存活率、S(t)均值。
作者: 曾皓轩
开发时间: 2026年4月
"""

import sys
import os
import json
import time
from typing import List, Dict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mars_pulse.config import MarsBaseConfig
from mars_pulse.environment import MarsBaseEnv
from mars_pulse.agents.static_agent import StaticRuleAgent, StaticRuleAgentConservative


# ─── Simulation runner ──────────────────────────────────────────────────

def run_episode(env: MarsBaseEnv, agent, max_steps: int = None) -> Dict:
    """Run a single episode and return results."""
    if max_steps is None:
        max_steps = env.max_steps

    obs, _ = env.reset()
    total_reward = 0.0
    S_history = []
    energy_history = []
    O2_history = []
    temp_history = []
    actions_history = []

    for step_i in range(max_steps):
        action = agent.select_action(obs)
        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        S_history.append(info["S"])
        energy_history.append(info["energy_stored"])
        O2_history.append(info["oxygen_stored"])
        temp_history.append(info["habitat_temp"])
        actions_history.append(action.copy())

        if terminated or truncated:
            break

    return {
        "survival_sols": float(round(info["survival_sols"], 2)),
        "total_reward": float(round(total_reward, 2)),
        "final_S": float(round(S_history[-1], 4)) if S_history else 0.0,
        "final_energy": float(round(energy_history[-1], 2)) if energy_history else 0.0,
        "final_O2": float(round(O2_history[-1], 2)) if O2_history else 0.0,
        "final_temp": float(round(temp_history[-1], 2)) if temp_history else 0.0,
        "min_energy": float(round(min(energy_history), 2)),
        "min_O2": float(round(min(O2_history), 2)),
        "min_temp": float(round(min(temp_history), 2)),
        "max_temp": float(round(max(temp_history), 2)),
        "S_min": float(round(min(S_history), 4)),
        "S_mean": float(round(np.mean(S_history), 4)),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "actions": [[float(round(float(a[i]), 2)) for i in range(3)] for a in actions_history],
    }


def run_monte_carlo(
    env: MarsBaseEnv,
    agent_factory,
    n_episodes: int = 100,
    verbose: bool = True,
) -> List[Dict]:
    """Run multiple episodes with different seeds, return aggregated results."""
    results = []
    for ep in range(n_episodes):
        seed = ep * 137 + 42
        env.reset(seed=seed)
        result = run_episode(env, agent_factory(env))
        results.append(result)
        if verbose and (ep + 1) % 20 == 0:
            print(f"  ... {ep + 1}/{n_episodes} episodes complete")
    return results


def aggregate_results(results: List[Dict], label: str) -> Dict:
    """Compute aggregate statistics."""
    survivals = [r["survival_sols"] for r in results]
    S_finals = [r["final_S"] for r in results]
    S_mins = [r["S_min"] for r in results]
    S_means = [r["S_mean"] for r in results]
    rewards = [r["total_reward"] for r in results]
    n_died = sum(1 for r in results if r["terminated"])
    n_survived = sum(1 for r in results if r["truncated"] and not r["terminated"])

    return {
        "agent": label,
        "n_episodes": len(results),
        "survival_sols_mean": round(np.mean(survivals), 2),
        "survival_sols_std": round(np.std(survivals), 2),
        "survival_sols_min": round(np.min(survivals), 2),
        "survival_sols_max": round(np.max(survivals), 2),
        "S_final_mean": round(np.mean(S_finals), 4),
        "S_final_std": round(np.std(S_finals), 4),
        "S_min_mean": round(np.mean(S_mins), 4),
        "S_min_worst": round(np.min(S_mins), 4),
        "S_mean_of_means": round(np.mean(S_means), 4),
        "total_reward_mean": round(np.mean(rewards), 2),
        "total_reward_std": round(np.std(rewards), 2),
        "fatality_rate": round(n_died / len(results), 4),
        "full_survival_rate": round(n_survived / len(results), 4),
    }


# ─── Random agent (lower bound) ────────────────────────────────────────

class RandomAgent:
    def __init__(self, env: MarsBaseEnv):
        self.env = env
        self.rng = np.random.default_rng()

    def select_action(self, obs: np.ndarray) -> np.ndarray:
        return self.env.action_space.sample()


# ─── Main ────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  MarsPulse Step 1 — Preliminary Simulation & Baseline Comparison")
    print("=" * 70)
    print()

    cfg = MarsBaseConfig()
    print("Configuration summary:")
    print(f"  Episode length: {cfg.episode_length} sols (dt={cfg.dt} sol/step)")
    print(f"  Storm: starts day {cfg.storm_start_day}, duration {cfg.storm_duration} sols")
    print(f"  Storm peak tau: {cfg.tau_storm} (power ~{cfg.storm_power_frac*100:.0f}%)")
    print(f"  Battery: {cfg.battery_capacity} kWh, O2: {cfg.O2_storage_capacity} kg")
    print(f"  Crew: {cfg.crew_size}, Target temp: {cfg.target_habitat_temp}°C")
    print(f"  S weights: E={cfg.w_energy}, O2={cfg.w_oxygen}, T={cfg.w_thermal}")
    print()

    N_EPISODES = 50  # matches step1_results.json; increase to 200+ for publication

    agents_to_test = [
        ("Random (lower bound)", RandomAgent),
        ("Static Rules", StaticRuleAgent),
        ("Static Conservative", StaticRuleAgentConservative),
    ]

    all_aggregates = []

    for label, agent_cls in agents_to_test:
        print(f"Running {label} agent ({N_EPISODES} episodes)...")
        env = MarsBaseEnv(config=cfg)
        t0 = time.time()
        results = run_monte_carlo(env, agent_cls, n_episodes=N_EPISODES)
        elapsed = time.time() - t0
        agg = aggregate_results(results, label)
        all_aggregates.append(agg)
        print(f"  Done in {elapsed:.1f}s")
        print(f"  Mean survival: {agg['survival_sols_mean']} ± {agg['survival_sols_std']} sols")
        print(f"  Fatality rate: {agg['fatality_rate']:.1%}")
        print(f"  Full survival rate: {agg['full_survival_rate']:.1%}")
        print(f"  S mean: {agg['S_mean_of_means']:.4f}, S min worst: {agg['S_min_worst']:.4f}")
        print()

    # ─── Comparison table ───────────────────────────────────────────
    print("=" * 70)
    print("  Comparison Summary")
    print("=" * 70)
    header = f"{'Agent':<28} {'Survival(sol)':<18} {'Fatality%':<12} {'FullSurv%':<12} {'S_mean':<10} {'Reward':<10}"
    print(header)
    print("-" * len(header))
    for agg in all_aggregates:
        print(
            f"{agg['agent']:<28} "
            f"{agg['survival_sols_mean']:>6.1f} ± {agg['survival_sols_std']:>5.1f}  "
            f"{agg['fatality_rate']:>8.1%}   "
            f"{agg['full_survival_rate']:>8.1%}   "
            f"{agg['S_mean_of_means']:>7.4f}  "
            f"{agg['total_reward_mean']:>7.1f}"
        )

    print()
    print("Key finding from Step 1:")
    baseline = all_aggregates[1]["survival_sols_mean"]  # Static Rules
    conservative = all_aggregates[2]["survival_sols_mean"]  # Static Conservative
    print(f"  Static Rules baseline:        {baseline:.1f} sols")
    print(f"  Static Conservative baseline: {conservative:.1f} sols")
    print(f"  These baselines will serve as comparison targets for MARL in Step 2.")

    # ─── Save raw data ──────────────────────────────────────────────
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output")
    os.makedirs(output_dir, exist_ok=True)

    data_path = os.path.join(output_dir, "step1_results.json")
    with open(data_path, "w") as f:
        json.dump(all_aggregates, f, indent=2)
    print(f"\nResults saved to {data_path}")

    # Also save detailed episode data for one representative run of each agent
    detail_path = os.path.join(output_dir, "step1_detail.json")
    detail_results = {}
    for label, agent_cls in agents_to_test:
        env = MarsBaseEnv(config=cfg)
        env.reset(seed=42)
        result = run_episode(env, agent_cls(env))
        detail_results[label] = result
    with open(detail_path, "w") as f:
        json.dump(detail_results, f, indent=2)
    print(f"Detail traces saved to {detail_path}")

    return all_aggregates


if __name__ == "__main__":
    main()
