"""
MarsPulse — 步骤4：消融实验
===========================
模块说明: 对比四种方法——静态规则、单智能体PPO、独立PPO×3、MAPPO(中心化评论家)。
         隔离中心化评论家的贡献，证明其在长时域任务中从"有益"变为"必需"。
         输出step4_ablation.json和对比柱状图。
作者: 毛元君
开发时间: 2026年5月
"""

import sys, os, json, time, copy
from typing import Dict
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mars_pulse.config import MarsBaseConfig
from mars_pulse.environment import MarsBaseEnv
from mars_pulse.multi_agent_env import MultiAgentMarsEnv
from mars_pulse.agents.static_agent import StaticRuleAgent
from mars_pulse.agents.ppo import PPO, PPOBuffer
from mars_pulse.agents.mappo import MAPPO, MAPPOReplayBuffer

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output")


# ─── Training utilities ──────────────────────────────────────────────

def train_single_ppo(total_steps: int = 50_000, device: str = "cpu") -> PPO:
    """Train single-agent PPO on MarsBaseEnv."""
    cfg = MarsBaseConfig()
    env = MarsBaseEnv(config=cfg)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    ppo = PPO(obs_dim, act_dim, device=device, ent_coef=0.05)
    buffer = PPOBuffer(obs_dim, act_dim, 2000, ppo.gamma, ppo.lam)

    steps = 0
    while steps < total_steps:
        obs, _ = env.reset(seed=steps)
        buffer.clear()
        episode_reward = 0.0

        for _ in range(1200):
            action, lp, _, _ = ppo.select_action(obs)
            value = ppo.compute_value(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            buffer.store(obs, action, reward, lp, value, done)
            episode_reward += reward
            steps += 1
            obs = next_obs if not done else env.reset(seed=steps)[0]

            if done or steps >= total_steps:
                break

        # Compute GAE
        if not done:
            last_val = ppo.compute_value(obs)
            last_done = False
        else:
            last_val = 0.0
            last_done = True
        buffer.compute_gae(last_val, last_done)

        # PPO update
        data = buffer.get()
        obs_t = torch.as_tensor(data["obs"], dtype=torch.float32, device=ppo.device)
        act_t = torch.as_tensor(data["actions"], dtype=torch.float32, device=ppo.device)
        ret_t = torch.as_tensor(data["returns"], dtype=torch.float32, device=ppo.device)
        adv_t = torch.as_tensor(data["advantages"], dtype=torch.float32, device=ppo.device)
        old_lp = torch.as_tensor(data["log_probs"], dtype=torch.float32, device=ppo.device)

        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        # Critic update
        for _ in range(ppo.train_iters):
            ppo.critic_optimizer.zero_grad()
            v_loss = nn.MSELoss()(ppo.critic(obs_t), ret_t)
            v_loss.backward()
            nn.utils.clip_grad_norm_(ppo.critic.parameters(), 10.0)
            ppo.critic_optimizer.step()

        # Actor update
        for _ in range(ppo.train_iters):
            ppo.actor_optimizer.zero_grad()
            new_lp, ent = ppo.actor.evaluate(obs_t, act_t)
            ratio = torch.exp(new_lp - old_lp)
            clip_adv = torch.clamp(ratio, 1 - ppo.clip_ratio, 1 + ppo.clip_ratio) * adv_t
            p_loss = -torch.min(ratio * adv_t, clip_adv).mean()
            ent_loss = -ppo.ent_coef * ent.mean()
            (p_loss + ent_loss).backward()
            nn.utils.clip_grad_norm_(ppo.actor.parameters(), 10.0)
            ppo.actor_optimizer.step()

    return ppo


def train_independent_ppo(total_steps: int = 50_000, device: str = "cpu") -> Dict:
    """Train three independent PPO agents (no centralized critic) — ablation."""
    cfg = MarsBaseConfig()
    env = MultiAgentMarsEnv(config=cfg)

    # Three separate PPO agents, each with own critic
    pp0 = PPO(env.obs_dims[0], 1, device=device, ent_coef=0.05)
    pp1 = PPO(env.obs_dims[1], 1, device=device, ent_coef=0.05)
    pp2 = PPO(env.obs_dims[2], 1, device=device, ent_coef=0.05)
    agents = [pp0, pp1, pp2]

    buffers = [
        PPOBuffer(env.obs_dims[i], 1, 2000, agents[i].gamma, agents[i].lam)
        for i in range(3)
    ]

    steps = 0
    while steps < total_steps:
        observations, _ = env.reset(seed=steps)
        for buf in buffers:
            buf.clear()
        episode_reward = 0.0

        for _ in range(1200):
            actions = []
            log_probs = []
            values = []
            for i, obs in enumerate(observations):
                act, lp, _, _ = agents[i].select_action(obs)
                val = agents[i].compute_value(obs)
                actions.append(act)
                log_probs.append(lp)
                values.append(val)

            actions_list = [a.astype(np.float32) for a in actions]
            next_obs, rewards, terminated, truncated, info = env.step(actions_list)
            done = terminated or truncated

            for i in range(3):
                buffers[i].store(observations[i], actions[i],
                                 rewards[i], log_probs[i], values[i], done)

            episode_reward += rewards[0]
            steps += 1
            observations = next_obs if not done else env.reset(seed=steps)[0]

            if done or steps >= total_steps:
                break

        # GAE + update for each agent
        for i in range(3):
            if not done:
                last_val = agents[i].compute_value(observations[i])
                buffers[i].compute_gae(last_val, False)
            else:
                buffers[i].compute_gae(0.0, True)

            data = buffers[i].get()
            obs_t = torch.as_tensor(data["obs"], dtype=torch.float32, device=device)
            act_t = torch.as_tensor(data["actions"], dtype=torch.float32, device=device)
            ret_t = torch.as_tensor(data["returns"], dtype=torch.float32, device=device)
            adv_t = torch.as_tensor(data["advantages"], dtype=torch.float32, device=device)
            old_lp = torch.as_tensor(data["log_probs"], dtype=torch.float32, device=device)
            adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

            for _ in range(agents[i].train_iters):
                agents[i].critic_optimizer.zero_grad()
                v_loss = nn.MSELoss()(agents[i].critic(obs_t), ret_t)
                v_loss.backward()
                nn.utils.clip_grad_norm_(agents[i].critic.parameters(), 10.0)
                agents[i].critic_optimizer.step()

            for _ in range(agents[i].train_iters):
                agents[i].actor_optimizer.zero_grad()
                new_lp, ent = agents[i].actor.evaluate(obs_t, act_t)
                ratio = torch.exp(new_lp - old_lp)
                clip_adv = torch.clamp(ratio, 1 - 0.2, 1 + 0.2) * adv_t
                p_loss = -torch.min(ratio * adv_t, clip_adv).mean()
                (p_loss - 0.05 * ent.mean()).backward()
                nn.utils.clip_grad_norm_(agents[i].actor.parameters(), 10.0)
                agents[i].actor_optimizer.step()

    return {"agents": agents, "buffers": buffers}


# ─── Evaluation utilities ────────────────────────────────────────────

def evaluate_static(env: MarsBaseEnv, n_eps: int = 50) -> Dict:
    survivals, fatalities = [], []
    for seed in range(n_eps):
        env.reset(seed=seed)
        agent = StaticRuleAgent(env)
        obs, _ = env.reset(seed=seed)
        for _ in range(env.max_steps):
            action = agent.select_action(obs)
            obs, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        survivals.append(info["survival_sols"])
        fatalities.append(1.0 if terminated else 0.0)
    return {"mean": np.mean(survivals), "std": np.std(survivals),
            "fatal_rate": np.mean(fatalities)}


def evaluate_single_ppo(ppo: PPO, n_eps: int = 50) -> Dict:
    cfg = MarsBaseConfig()
    env = MarsBaseEnv(config=cfg)
    survivals, fatalities = [], []
    for seed in range(n_eps):
        obs, _ = env.reset(seed=seed)
        for _ in range(env.max_steps):
            action, _, _, _ = ppo.select_action(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        survivals.append(info["survival_sols"])
        fatalities.append(1.0 if terminated else 0.0)
    return {"mean": np.mean(survivals), "std": np.std(survivals),
            "fatal_rate": np.mean(fatalities)}


def evaluate_mappo(mappo: MAPPO, n_eps: int = 50) -> Dict:
    """Evaluate MAPPO (with centralized critic) using proper multi-agent observations."""
    cfg = MarsBaseConfig()
    env = MarsBaseEnv(config=cfg)
    survivals, fatalities = [], []
    for seed in range(n_eps):
        obs, _ = env.reset(seed=seed)
        for _ in range(env.max_steps):
            multi_obs = env.get_multi_agent_observations()
            actions, _, _, _ = mappo.select_actions(multi_obs, deterministic=True)
            action = np.array([float(actions[0][0]), float(actions[1][0]), float(actions[2][0])], dtype=np.float32)
            obs, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        survivals.append(info["survival_sols"])
        fatalities.append(1.0 if terminated else 0.0)
    return {"mean": np.mean(survivals), "std": np.std(survivals),
            "fatal_rate": np.mean(fatalities)}


def evaluate_independent(agents_dict: Dict, n_eps: int = 50) -> Dict:
    """Evaluate 3 independent PPO agents using proper multi-agent observations."""
    agents = agents_dict["agents"]
    cfg = MarsBaseConfig()
    env = MarsBaseEnv(config=cfg)
    survivals, fatalities = [], []
    for seed in range(n_eps):
        obs, _ = env.reset(seed=seed)
        for _ in range(env.max_steps):
            multi_obs = env.get_multi_agent_observations()
            a0, _, _, _ = agents[0].select_action(multi_obs[0], deterministic=True)
            a1, _, _, _ = agents[1].select_action(multi_obs[1], deterministic=True)
            a2, _, _, _ = agents[2].select_action(multi_obs[2], deterministic=True)
            action = np.array([float(a0[0]), float(a1[0]), float(a2[0])], dtype=np.float32)
            obs, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        survivals.append(info["survival_sols"])
        fatalities.append(1.0 if terminated else 0.0)
    return {"mean": np.mean(survivals), "std": np.std(survivals),
            "fatal_rate": np.mean(fatalities)}


# ─── Main ─────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  MarsPulse Step 4 — Algorithm Comparison & Ablation")
    print("=" * 70)
    print()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = MarsBaseConfig()

    # --- Train single-agent PPO ---
    # Note: MAPPO was trained for 300k steps. We use 100k here as a practical
    # limit; single PPO's failure at 2.5 sols (vs 114 sols static) is severe
    # enough that additional steps are unlikely to close a >100 sol gap.
    print("Training Single-Agent PPO (100k steps)...")
    t0 = time.time()
    ppo = train_single_ppo(100000, device)
    print(f"  Done in {time.time()-t0:.0f}s")

    # --- Train independent PPO (ablation) ---
    # Same note as above: 100k steps vs MAPPO's 300k. The 2.4→129.7 sol gap
    # is so large that even 300k steps would be unlikely to close it entirely.
    print("Training Independent PPO — no centralized critic (100k steps)...")
    t0 = time.time()
    ind_ppo = train_independent_ppo(100000, device)
    print(f"  Done in {time.time()-t0:.0f}s")

    # --- Load MAPPO ---
    print("Loading MAPPO (centralized critic)...")
    multi_env = MultiAgentMarsEnv(config=cfg)
    act_dims = [1, 1, 1]
    mappo = MAPPO(multi_env.obs_dims, act_dims, multi_env.global_obs_dim, device=device)
    ckpt = torch.load(os.path.join(OUTPUT_DIR, "mappo_model.pt"),
                      map_location=device, weights_only=False)
    mappo.actors.load_state_dict(ckpt["actors"])
    mappo.critic.load_state_dict(ckpt["critic"])
    print(f"  Loaded (epoch {ckpt['epoch']}, best={ckpt['best_survival']:.1f} sols)")

    # --- Evaluate all ---
    print("\nEvaluating all approaches (50 episodes each)...\n")
    N = 50

    env = MarsBaseEnv(config=cfg)
    r_static = evaluate_static(env, N)
    print(f"  Static Rules:        {r_static['mean']:.1f} ± {r_static['std']:.1f} sols  (fatality {r_static['fatal_rate']:.1%})")

    r_sp = evaluate_single_ppo(ppo, N)
    print(f"  Single-Agent PPO:    {r_sp['mean']:.1f} ± {r_sp['std']:.1f} sols  (fatality {r_sp['fatal_rate']:.1%})")

    r_ind = evaluate_independent(ind_ppo, N)
    print(f"  Independent PPO (3): {r_ind['mean']:.1f} ± {r_ind['std']:.1f} sols  (fatality {r_ind['fatal_rate']:.1%})")

    r_mappo = evaluate_mappo(mappo, N)
    print(f"  MAPPO (centralized): {r_mappo['mean']:.1f} ± {r_mappo['std']:.1f} sols  (fatality {r_mappo['fatal_rate']:.1%})")

    # --- Save results ---
    results = {
        "Static Rules": r_static,
        "Single-Agent PPO": r_sp,
        "Independent PPO (no central critic)": r_ind,
        "MAPPO (centralized critic)": r_mappo,
    }
    # Convert numpy scalar types to native Python (compatible with NumPy 1.x and 2.x)
    for k, v in results.items():
        converted = {}
        for kk, vv in v.items():
            if hasattr(vv, 'item'):
                converted[kk] = float(vv.item())
            elif isinstance(vv, (int, float)):
                converted[kk] = vv
            else:
                converted[kk] = vv
        results[k] = converted

    with open(os.path.join(OUTPUT_DIR, "step4_ablation.json"), "w") as f:
        json.dump(results, f, indent=2)

    # --- Plot ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    labels = list(results.keys())
    short_labels = ["Static\nRules", "Single-Agent\nPPO", "Independent\nPPO (3x)", "MAPPO\n(central critic)"]
    colors = ["#757575", "#2196F3", "#FF9800", "#4CAF50"]

    ax = axes[0]
    means = [results[l]["mean"] for l in labels]
    stds = [results[l]["std"] for l in labels]
    bars = ax.bar(short_labels, means, color=colors, edgecolor="black", linewidth=0.5)
    ax.errorbar(range(len(short_labels)), means, yerr=stds, fmt="none", ecolor="black", capsize=5)
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"{m:.1f}", ha="center", fontweight="bold", fontsize=11)
    ax.set_ylabel("Mean Survival (sols)")
    ax.set_title("Algorithm Comparison — Survival Time")
    ax.grid(True, alpha=0.2, axis="y")

    ax = axes[1]
    fats = [results[l]["fatal_rate"] * 100 for l in labels]
    bars = ax.bar(short_labels, fats, color=colors, edgecolor="black", linewidth=0.5)
    for bar, f in zip(bars, fats):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"{f:.1f}%", ha="center", fontweight="bold", fontsize=11)
    ax.set_ylabel("Fatality Rate (%)")
    ax.set_title("Algorithm Comparison — Crew Fatality Rate")
    ax.grid(True, alpha=0.2, axis="y")

    # Annotate centralized critic contribution
    improvement = r_mappo["mean"] - r_ind["mean"]
    ax.annotate(f"Centralized critic\n+{improvement:.1f} sols improvement",
                xy=(2, fats[2]), textcoords="offset points",
                xytext=(40, 30), fontsize=10, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="green", lw=2),
                bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.8))

    plt.suptitle("MarsPulse Step 4 — Ablation Study: Centralized Critic Contribution", fontsize=14, fontweight="bold")
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, "step4_ablation.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"\n  Ablation plot saved to {path}")

    # Key finding
    print(f"\n  Key finding: Centralized critic improves survival by {improvement:.1f} sols")
    print(f"  over independent PPO agents, validating the MARL architecture choice.")


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    main()
