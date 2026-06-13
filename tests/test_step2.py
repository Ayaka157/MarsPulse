"""
MarsPulse — 步骤2：MARL评估对比
===============================
模块说明: 加载训练好的MAPPO模型，运行蒙特卡洛评估对比静态基线。
         使用MAPPOWrapper将多智能体策略映射至单智能体环境。
         输出step2_results.json。
作者: 曾皓轩
开发时间: 2026年5月
"""

import sys
import os
import json
import time
from typing import List, Dict

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mars_pulse.config import MarsBaseConfig
from mars_pulse.environment import MarsBaseEnv
from mars_pulse.multi_agent_env import MultiAgentMarsEnv
from mars_pulse.agents.static_agent import StaticRuleAgent, StaticRuleAgentConservative
from mars_pulse.agents.mappo import MAPPO

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output")


# ─── Multi-agent to single-agent wrapper ─────────────────────────────

class MAPPOWrapper:
    """Wraps MAPPO multi-agent policy for use with single-agent MarsBaseEnv.

    Builds per-agent observations from the flattened single-agent observation
    vector (consistent with the observation layout in plot_step2.py), so the
    wrapper never references a stale environment instance.
    """

    def __init__(self, mappo: MAPPO, cfg: MarsBaseConfig):
        self.mappo = mappo
        self.cfg = cfg
        self.n_failures = len(cfg.failure_modes)

    def select_action(self, obs: np.ndarray) -> np.ndarray:
        """Decompose flat obs into three per-agent views for MAPPO."""
        failures = [float(obs[5 + i]) for i in range(self.n_failures)]

        # Agent 0 (Power Manager): energy, O2, storm, tau + failures
        obs0 = np.array([obs[0], obs[1], obs[3], obs[4]] + failures, dtype=np.float32)
        # Agent 1 (Thermal Controller): temp, energy, storm + failures
        obs1 = np.array([obs[2], obs[0], obs[3]] + failures, dtype=np.float32)
        # Agent 2 (Science Manager): energy, storm, O2 + failures
        obs2 = np.array([obs[0], obs[3], obs[1]] + failures, dtype=np.float32)

        actions, _, _, _ = self.mappo.select_actions([obs0, obs1, obs2], deterministic=True)

        life_support = float(actions[0][0])
        heater = float(actions[1][0])
        experiments = float(actions[2][0])

        return np.array([life_support, heater, experiments], dtype=np.float32)


def run_single_episode(env: MarsBaseEnv, agent, max_steps: int = None) -> Dict:
    """Run a single episode with single-agent env."""
    if max_steps is None:
        max_steps = env.max_steps
    obs, _ = env.reset()
    total_reward = 0.0
    S_history = []
    energy_history = []
    O2_history = []

    for _ in range(max_steps):
        action = agent.select_action(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        S_history.append(info["S"])
        energy_history.append(info["energy_stored"])
        O2_history.append(info["oxygen_stored"])
        if terminated or truncated:
            break

    return {
        "survival_sols": float(round(info["survival_sols"], 2)),
        "total_reward": float(round(total_reward, 2)),
        "final_S": float(round(S_history[-1], 4)) if S_history else 0.0,
        "final_energy": float(round(energy_history[-1], 2)) if energy_history else 0.0,
        "final_O2": float(round(O2_history[-1], 2)) if O2_history else 0.0,
        "S_min": float(round(min(S_history), 4)),
        "S_mean": float(round(np.mean(S_history), 4)),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
    }


def run_monte_carlo(env, agent_factory, n_episodes: int = 100, label: str = "") -> List[Dict]:
    results = []
    for ep in range(n_episodes):
        seed = ep * 251 + 17
        env.reset(seed=seed)
        result = run_single_episode(env, agent_factory())
        results.append(result)
    return results


def aggregate(results: List[Dict], label: str) -> Dict:
    survivals = [r["survival_sols"] for r in results]
    n_died = sum(1 for r in results if r["terminated"])
    n_survived = sum(1 for r in results if r["truncated"] and not r["terminated"])

    return {
        "agent": label,
        "n_episodes": len(results),
        "survival_sols_mean": round(np.mean(survivals), 2),
        "survival_sols_std": round(np.std(survivals), 2),
        "survival_sols_min": round(np.min(survivals), 2),
        "survival_sols_max": round(np.max(survivals), 2),
        "S_final_mean": round(np.mean([r["final_S"] for r in results]), 4),
        "S_min_mean": round(np.mean([r["S_min"] for r in results]), 4),
        "S_mean_of_means": round(np.mean([r["S_mean"] for r in results]), 4),
        "total_reward_mean": round(np.mean([r["total_reward"] for r in results]), 2),
        "total_reward_std": round(np.std([r["total_reward"] for r in results]), 2),
        "fatality_rate": round(n_died / len(results), 4),
        "full_survival_rate": round(n_survived / len(results), 4),
    }


def main():
    print("=" * 70)
    print("  MarsPulse Step 2 — MARL Evaluation & Comparison")
    print("=" * 70)
    print()

    cfg = MarsBaseConfig()
    N_EPISODES = 100

    # Load trained MAPPO model
    model_path = os.path.join(OUTPUT_DIR, "mappo_model.pt")
    if not os.path.exists(model_path):
        print(f"ERROR: No trained model found at {model_path}")
        print("Run train_step2.py first to train the MAPPO agents.")
        sys.exit(1)

    print(f"Loading trained MAPPO model from {model_path}...")
    multi_env = MultiAgentMarsEnv(config=cfg)
    act_dims = [1, 1, 1]
    mappo = MAPPO(
        obs_dims=multi_env.obs_dims,
        act_dims=act_dims,
        global_obs_dim=multi_env.global_obs_dim,
    )
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    mappo.actors.load_state_dict(checkpoint["actors"])
    mappo.critic.load_state_dict(checkpoint["critic"])
    print(f"  Loaded model from epoch {checkpoint['epoch']}, "
          f"best survival={checkpoint['best_survival']:.1f} sols")
    print()

    # Create agent wrappers
    def make_mappo_agent():
        return MAPPOWrapper(mappo, cfg)

    def make_static_rules():
        env = MarsBaseEnv(config=cfg)
        return StaticRuleAgent(env)

    def make_static_conservative():
        env = MarsBaseEnv(config=cfg)
        return StaticRuleAgentConservative(env)

    agents_to_test = [
        ("Static Rules (Step1 baseline)", make_static_rules),
        ("Static Conservative", make_static_conservative),
        ("MAPPO MARL (Step2)", make_mappo_agent),
    ]

    all_aggregates = []

    for label, factory in agents_to_test:
        print(f"Evaluating {label} ({N_EPISODES} episodes)...")
        env = MarsBaseEnv(config=cfg)
        t0 = time.time()
        results = run_monte_carlo(env, factory, n_episodes=N_EPISODES, label=label)
        elapsed = time.time() - t0
        agg = aggregate(results, label)
        all_aggregates.append(agg)
        print(f"  Done in {elapsed:.1f}s")
        print(f"  Mean survival: {agg['survival_sols_mean']:.1f} ± {agg['survival_sols_std']:.1f} sols")
        print(f"  Fatality: {agg['fatality_rate']:.1%} | Full survival: {agg['full_survival_rate']:.1%}")
        print(f"  S mean: {agg['S_mean_of_means']:.4f} | Reward: {agg['total_reward_mean']:.1f}")
        print()

    # ─── Comparison ─────────────────────────────────────────────────
    print("=" * 70)
    print("  Step 2 Comparison Summary")
    print("=" * 70)
    header = f"{'Agent':<32} {'Survival(sol)':<18} {'Fatality%':<12} {'FullSurv%':<12} {'S_mean':<10} {'Reward':<10}"
    print(header)
    print("-" * len(header))
    for agg in all_aggregates:
        print(
            f"{agg['agent']:<32} "
            f"{agg['survival_sols_mean']:>6.1f} ± {agg['survival_sols_std']:>5.1f}  "
            f"{agg['fatality_rate']:>8.1%}   "
            f"{agg['full_survival_rate']:>8.1%}   "
            f"{agg['S_mean_of_means']:>7.4f}  "
            f"{agg['total_reward_mean']:>7.1f}"
        )

    # Improvement over best static baseline
    static_best = max(a["survival_sols_mean"] for a in all_aggregates if "Static" in a["agent"])
    mappo_result = next(a for a in all_aggregates if "MAPPO" in a["agent"])
    improvement = (mappo_result["survival_sols_mean"] - static_best) / static_best * 100

    print()
    print(f"  Best static baseline: {static_best:.1f} sols")
    print(f"  MAPPO MARL:           {mappo_result['survival_sols_mean']:.1f} sols")
    print(f"  Improvement:          {improvement:+.1f}%")
    print()

    # Save
    save_path = os.path.join(OUTPUT_DIR, "step2_results.json")
    with open(save_path, "w") as f:
        json.dump(all_aggregates, f, indent=2)
    print(f"Results saved to {save_path}")


if __name__ == "__main__":
    main()
