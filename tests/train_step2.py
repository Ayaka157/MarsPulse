"""
MarsPulse — 步骤2：MAPPO训练脚本
================================
模块说明: 使用MultiAgentMarsEnv训练三智能体MAPPO策略，300,000环境步。
         每15轮评估一次，自动保存最佳模型至output/mappo_model.pt。
         支持CUDA加速，含训练日志记录。
作者: 曾皓轩
开发时间: 2026年4–5月
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
from mars_pulse.multi_agent_env import MultiAgentMarsEnv
from mars_pulse.agents.mappo import MAPPO, MAPPOReplayBuffer

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output")


def compute_value(mappo: MAPPO, global_obs: np.ndarray, actions_flat: np.ndarray) -> float:
    """Compute critic value for a given global state and joint action."""
    go_t = torch.as_tensor(global_obs, dtype=torch.float32, device=mappo.device).unsqueeze(0)
    acts_t = torch.as_tensor(actions_flat, dtype=torch.float32, device=mappo.device).unsqueeze(0)
    with torch.no_grad():
        val = mappo.critic(go_t, acts_t)
    return val.item()


def evaluate_agent(mappo: MAPPO, env: MultiAgentMarsEnv, n_episodes: int = 10) -> Dict:
    """Evaluate current policy (deterministic) over several episodes."""
    survivals = []
    S_finals = []
    for ep in range(n_episodes):
        observations, _ = env.reset(seed=ep * 1234)
        done = False
        while not done:
            actions, _, _, _ = mappo.select_actions(observations, deterministic=True)
            observations, rewards, terminated, truncated, info = env.step(actions)
            done = terminated or truncated
        survivals.append(info["survival_sols"])
        S_finals.append(info["S"])

    return {
        "survival_mean": np.mean(survivals),
        "survival_std": np.std(survivals),
        "S_final_mean": np.mean(S_finals),
    }


def train(
    total_steps: int = 300_000,
    steps_per_epoch: int = 3000,
    max_episode_steps: int = 6000,
    eval_interval: int = 15,
    save_path: str = None,
):
    """Main training loop with bounded actions and improved stability."""
    cfg = MarsBaseConfig()
    env = MultiAgentMarsEnv(config=cfg)

    # MAPPO setup with action bounds from environment
    act_dims = [1, 1, 1]
    mappo = MAPPO(
        obs_dims=env.obs_dims,
        act_dims=act_dims,
        global_obs_dim=env.global_obs_dim,
        action_bounds=env.action_bounds,
        lr_actor=3e-4,
        lr_critic=8e-4,
        gamma=0.995,         # higher discount for long horizon (300 sols)
        lam=0.95,
        clip_ratio=0.2,
        target_kl=0.01,      # tighter KL for stability
        ent_coef=0.05,       # higher entropy for exploration
        ppo_epochs=8,        # mini-batch PPO epochs
        mini_batch_size=512, # mini-batch size for efficient GPU usage
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    print(f"MAPPO device: {mappo.device}")
    print(f"Config: episode={cfg.episode_length} sols, battery={cfg.battery_capacity} kWh, "
          f"solar={cfg.P_max} kW, RTG={cfg.P_rtg} kW")
    print(f"Storm: sol {cfg.storm_start_day}-{cfg.storm_start_day+cfg.storm_duration} "
          f"(tau={cfg.tau_storm}), Storm2: sol {cfg.storm2_start_day}-"
          f"{cfg.storm2_start_day+cfg.storm2_duration} (tau={cfg.storm2_tau})")
    print(f"Training for {total_steps} steps...")
    print()

    buffer = MAPPOReplayBuffer(
        obs_dims=env.obs_dims,
        act_dims=act_dims,
        global_obs_dim=env.global_obs_dim,
        buffer_size=steps_per_epoch * 2,
        n_agents=env.n_agents,
        gamma=mappo.gamma,
        lam=mappo.lam,
    )

    global_step = 0
    epoch = 0
    best_survival = 0.0
    log_history = []

    while global_step < total_steps:
        epoch += 1
        observations, _ = env.reset(seed=global_step)
        episode_rewards = np.zeros(mappo.n_agents)

        buffer.clear()

        for _ in range(steps_per_epoch):
            global_obs = env.get_global_obs()
            actions_list, log_probs, _, means = mappo.select_actions(observations)

            actions_flat = np.concatenate([a.astype(np.float32) for a in actions_list])
            value = compute_value(mappo, global_obs, actions_flat)

            next_observations, rewards, terminated, truncated, info = env.step(actions_list)
            done = terminated or truncated

            buffer.store(
                global_obs=global_obs,
                observations=observations,
                actions=actions_flat,
                agent_actions=[a.astype(np.float32).reshape(-1) for a in actions_list],
                rewards=np.array(rewards, dtype=np.float32),
                log_probs=log_probs,
                value=value,
                done=done,
            )

            episode_rewards += np.array(rewards, dtype=np.float32)
            global_step += 1
            observations = next_observations

            if done:
                observations, _ = env.reset(seed=global_step)
                # Don't break — continue collecting for stable batch size

        # GAE
        if not done:
            global_obs = env.get_global_obs()
            actions_list, _, _, _ = mappo.select_actions(observations)
            actions_flat = np.concatenate([a.astype(np.float32) for a in actions_list])
            last_value = compute_value(mappo, global_obs, actions_flat)
            last_done = False
        else:
            last_value = 0.0
            last_done = True

        buffer.compute_gae(last_value, last_done)
        stats = mappo.update(buffer)

        mean_reward = episode_rewards.mean()
        log_history.append({
            "epoch": epoch,
            "global_step": global_step,
            "mean_reward": float(mean_reward),
            "critic_loss": float(stats["critic_loss"]),
        })

        if epoch % 5 == 0:
            print(f"Epoch {epoch:4d} | steps {global_step:7d} | "
                  f"reward {mean_reward:8.1f} | c_loss {stats['critic_loss']:.4f} | "
                  f"a0_kl {stats['actor_0_kl']:.4f} | a1_kl {stats['actor_1_kl']:.4f} | "
                  f"a2_kl {stats['actor_2_kl']:.4f}")

        if epoch % eval_interval == 0:
            eval_env = MultiAgentMarsEnv(config=cfg)
            metrics = evaluate_agent(mappo, eval_env, n_episodes=5)
            print(f"  >>> EVAL | survival {metrics['survival_mean']:.1f} ± "
                  f"{metrics['survival_std']:.1f} sols | "
                  f"S_final {metrics['S_final_mean']:.4f}")

            if metrics["survival_mean"] > best_survival:
                best_survival = metrics["survival_mean"]
                if save_path:
                    torch.save({
                        "actors": mappo.actors.state_dict(),
                        "critic": mappo.critic.state_dict(),
                        "epoch": epoch,
                        "global_step": global_step,
                        "best_survival": best_survival,
                    }, save_path)
                    print(f"  >>> Saved best model (survival={best_survival:.1f} sols)")

    log_path = os.path.join(OUTPUT_DIR, "step2_training_log.json")
    with open(log_path, "w") as f:
        json.dump(log_history, f, indent=2)
    print(f"\nTraining log saved to {log_path}")

    return mappo, log_history


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model_path = os.path.join(OUTPUT_DIR, "mappo_model.pt")

    t0 = time.time()
    mappo, history = train(
        total_steps=300_000,
        steps_per_epoch=3000,
        eval_interval=15,
        save_path=model_path,
    )
    elapsed = time.time() - t0
    print(f"\nTraining completed in {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"Model saved to {model_path}")
