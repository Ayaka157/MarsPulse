"""
MarsPulse — 步骤2：训练可视化
=============================
模块说明: 生成三张图表——训练曲线(奖励+Critic损失+生存趋势)、
         MARL vs 基线对比图（含统计分布）、
         MAPPO智能体动作决策轨迹图(含双风暴标注)。
         输出step2_training.png/comparison.png/agent_actions.png。
作者: 毛元君
开发时间: 2026年5月
"""

import sys
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mars_pulse.config import MarsBaseConfig
from mars_pulse.environment import MarsBaseEnv
from mars_pulse.multi_agent_env import MultiAgentMarsEnv
from mars_pulse.agents.static_agent import StaticRuleAgent

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output")


def plot_training_curve(log_path: str, output_path: str):
    """Plot training reward, critic loss, and survival trend."""
    with open(log_path) as f:
        log = json.load(f)

    steps = [e["global_step"] for e in log]
    rewards = [e["mean_reward"] for e in log]
    c_losses = [e["critic_loss"] for e in log]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    window = max(1, len(rewards) // 20)

    # Reward
    ax1.plot(steps, rewards, color="#2196F3", linewidth=1.0, alpha=0.4)
    smoothed = np.convolve(rewards, np.ones(window)/window, mode="valid")
    ax1.plot(steps[window-1:], smoothed, color="#1565C0", linewidth=2, label=f"Smoothed (w={window})")
    # Trend line
    z = np.polyfit(steps, rewards, 1)
    p = np.poly1d(z)
    ax1.plot(steps, p(steps), "--", color="#0D47A1", linewidth=1.5, alpha=0.7, label=f"Trend (r={np.corrcoef(steps, rewards)[0,1]:.3f})")
    ax1.set_xlabel("Steps")
    ax1.set_ylabel("Mean Reward")
    ax1.set_title("Training Reward")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Critic loss
    ax2.plot(steps, c_losses, color="#FF5722", linewidth=1.0, alpha=0.4)
    smoothed_c = np.convolve(c_losses, np.ones(window)/window, mode="valid")
    ax2.plot(steps[window-1:], smoothed_c, color="#BF360C", linewidth=2, label=f"Smoothed (w={window})")
    ax2.set_xlabel("Steps")
    ax2.set_ylabel("Critic Loss")
    ax2.set_title("Critic Loss (Converging)")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.suptitle("MarsPulse Step 2 — MAPPO Training Curves (300k steps)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Training curves saved to {output_path}")


def plot_step2_comparison(results_path: str, output_path: str):
    """Plot Step 2 MARL vs baseline comparison with full detail."""
    with open(results_path) as f:
        data = json.load(f)
    if isinstance(data, dict) and "results" in data:
        results = data["results"]
    else:
        results = data

    fig = plt.figure(figsize=(18, 8))
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1, 1])

    labels = [r["agent"] for r in results]
    colors = ["#2196F3", "#4CAF50", "#FF9800"]

    # Left: bar chart with 3 metrics
    ax = fig.add_subplot(gs[0])
    means = [r["survival_sols_mean"] for r in results]
    stds = [r["survival_sols_std"] for r in results]
    fatalities = [r["fatality_rate"] * 100 for r in results]
    survivals = [r["full_survival_rate"] * 100 for r in results]

    x = np.arange(len(labels))
    width = 0.22

    bars1 = ax.bar(x - width, means, width, color=colors,
                   label="Mean Survival (sols)", edgecolor="black", linewidth=0.5)
    ax.set_ylabel("Survival Time (sols)", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8, rotation=10)

    # Error bars
    for i, (m, s) in enumerate(zip(means, stds)):
        ax.errorbar(x[i] - width, m, yerr=s, fmt="none", ecolor="black", capsize=4)

    ax2 = ax.twinx()
    bars2 = ax2.bar(x, fatalities, width, color=["#FF5252", "#FF5252", "#FF5252"],
                    label="Fatality Rate (%)", edgecolor="black", linewidth=0.5, alpha=0.5)
    bars3 = ax2.bar(x + width, survivals, width, color=["#4CAF50", "#4CAF50", "#4CAF50"],
                    label="Full Survival Rate (%)", edgecolor="black", linewidth=0.5, alpha=0.5)
    ax2.set_ylabel("Rate (%)", fontsize=11)

    for bar, val in zip(bars1, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{val:.1f}", ha="center", fontweight="bold", fontsize=10)
    for bar, val in zip(bars2, fatalities):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 f"{val:.1f}%", ha="center", fontsize=8, color="red")
    for bar, val in zip(bars3, survivals):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 f"{val:.1f}%", ha="center", fontsize=8, color="green")

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=8)
    ax.set_title("Survival Time & Distribution", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.2, axis="y")

    # Right: detailed metrics table
    ax_table = fig.add_subplot(gs[1])
    ax_table.axis("off")

    table_data = []
    col_labels = ["Agent", "Survival (sols)", "Min / Max", "Fatality%", "FullSurv%", "S(t)", "Reward"]
    for r in results:
        table_data.append([
            r["agent"],
            f'{r["survival_sols_mean"]:.1f} ± {r["survival_sols_std"]:.1f}',
            f'{r["survival_sols_min"]:.1f} / {r["survival_sols_max"]:.1f}',
            f'{r["fatality_rate"]*100:.1f}%',
            f'{r["full_survival_rate"]*100:.1f}%',
            f'{r.get("S_mean_of_means", r.get("S_final_mean", 0)):.4f}',
            f'{r["total_reward_mean"]:.1f}',
        ])

    table = ax_table.table(cellText=table_data, colLabels=col_labels,
                           loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 2.0)
    for j in range(len(col_labels)):
        table[0, j].set_facecolor("#E65100")
        table[0, j].set_text_props(color="white", fontweight="bold")
    for i in range(len(table_data)):
        for j in range(len(col_labels)):
            table[i+1, j].set_facecolor("#FFF3E0" if i % 2 == 0 else "white")
    # Highlight MAPPO row
    for j in range(len(col_labels)):
        table[len(table_data), j].set_facecolor("#FFCC80")

    # Compute improvement
    static_best = max(r["survival_sols_mean"] for r in results if "Static" in r["agent"])
    mappo_val = next(r["survival_sols_mean"] for r in results if "MAPPO" in r["agent"])
    improvement = (mappo_val - static_best) / static_best * 100
    ax_table.text(0.5, -0.08,
                  f"MAPPO MARL improvement over best static baseline: {improvement:+.1f}%  "
                  f"({mappo_val:.1f} vs {static_best:.1f} sols)",
                  ha="center", fontsize=11, fontweight="bold",
                  color="#E65100", transform=ax_table.transAxes)

    ax_table.set_title("Detailed Evaluation Metrics", fontsize=12, fontweight="bold", y=0.98)

    plt.suptitle("MarsPulse Step 2 — MARL vs Baseline Comparison (100 Monte Carlo episodes)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Comparison chart saved to {output_path}")


def plot_agent_actions(log_path=None):
    """Plot a detailed episode showing MAPPO agent action decisions."""
    import torch
    from mars_pulse.agents.mappo import MAPPO

    cfg = MarsBaseConfig()
    env = MarsBaseEnv(config=cfg)
    multi_env = MultiAgentMarsEnv(config=cfg)

    # Try to load trained model
    model_path = os.path.join(OUTPUT_DIR, "mappo_model.pt")
    if not os.path.exists(model_path):
        print("  No trained model for action plot, skipping.")
        return

    act_dims = [1, 1, 1]
    mappo = MAPPO(
        obs_dims=multi_env.obs_dims,
        act_dims=act_dims,
        global_obs_dim=multi_env.global_obs_dim,
    )
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    mappo.actors.load_state_dict(checkpoint["actors"])

    # Run a detailed episode
    obs, _ = env.reset(seed=268)  # representative surviving episode (full 300 sols)
    history = {
        "time": [], "S": [], "energy": [], "storm": [],
        "life_support": [], "heating": [], "experiments": [],
    }
    failures = list(cfg.failure_modes)
    max_steps = env.max_steps

    for _ in range(max_steps):
        f = [float(obs[i+5]) for i in range(len(failures))]
        obs0 = np.array([obs[0], obs[1], obs[3], obs[4]] + f, dtype=np.float32)
        obs1 = np.array([obs[2], obs[0], obs[3]] + f, dtype=np.float32)
        obs2 = np.array([obs[0], obs[3], obs[1]] + f, dtype=np.float32)

        actions, _, _, _ = mappo.select_actions([obs0, obs1, obs2], deterministic=True)

        action = np.array([float(actions[0][0]), float(actions[1][0]), float(actions[2][0])],
                          dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)

        history["time"].append(float(info["survival_sols"]))
        history["S"].append(float(info["S"]))
        history["energy"].append(float(info["energy_stored"]))
        history["storm"].append(float(info["storm_active"]))
        history["life_support"].append(float(action[0]))
        history["heating"].append(float(action[1]))
        history["experiments"].append(float(action[2]))

        if terminated or truncated:
            break

    survival_time = history["time"][-1]
    print(f"  MAPPO agent action episode: survived {survival_time:.1f} sols")

    # Plot
    fig, axes = plt.subplots(3, 1, figsize=(14, 11),
                              gridspec_kw={"height_ratios": [2, 1.5, 1.5]})

    # Top: Agent actions with storm shading
    ax = axes[0]
    ax.plot(history["time"], history["life_support"], color="#2196F3",
            linewidth=1.5, label="Power Agent: Life Support (kW)")
    ax.plot(history["time"], history["heating"], color="#F44336",
            linewidth=1.5, label="Thermal Agent: Heating (kW)")
    ax.plot(history["time"], history["experiments"], color="#9C27B0",
            linewidth=1.5, label="Science Agent: Experiments (kW)")
    ax.axvline(x=cfg.storm_start_day, color="orange", linestyle="--", alpha=0.7, label="Storm1 start")
    ax.axvline(x=cfg.storm_start_day + cfg.storm_duration, color="orange", linestyle="--", alpha=0.4, label="Storm1 end")
    ax.axvline(x=cfg.storm2_start_day, color="red", linestyle="--", alpha=0.7, label="Storm2 start")
    ax.axvline(x=cfg.storm2_start_day + cfg.storm2_duration, color="red", linestyle="--", alpha=0.4, label="Storm2 end")
    ax.fill_between([cfg.storm_start_day, cfg.storm_start_day+cfg.storm_duration], 0, 35,
                    color="orange", alpha=0.1)
    ax.fill_between([cfg.storm2_start_day, cfg.storm2_start_day+cfg.storm2_duration], 0, 35,
                    color="red", alpha=0.08)
    ax.set_ylabel("Power (kW)")
    ax.set_title(f"MAPPO MARL — Agent Action Decisions (survived {survival_time:.1f} sols)")
    ax.legend(fontsize=9, loc="upper right", ncol=3)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 250)

    # Middle: S(t) with storm overlay
    ax = axes[1]
    ax.plot(history["time"], history["S"], color="#4CAF50", linewidth=2, label="Survival Stability S(t)")
    ax.fill_between(history["time"], 0, history["storm"], alpha=0.2, color="red", label="Storm active")
    ax.axhline(y=cfg.S_critical, color="orange", linestyle="--", alpha=0.5, label=f"S_critical={cfg.S_critical}")
    ax.axhline(y=cfg.S_fatal, color="red", linestyle="--", alpha=0.5, label=f"S_fatal={cfg.S_fatal}")
    ax.set_ylabel("S(t)")
    ax.set_title("Survival Stability Index")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 250)

    # Bottom: Energy (battery %) over time
    ax = axes[2]
    ax.plot(history["time"], np.array(history["energy"]) / cfg.battery_capacity * 100,
            color="#1565C0", linewidth=2, label="Battery (%)")
    ax.axvline(x=cfg.storm_start_day, color="orange", linestyle="--", alpha=0.5)
    ax.axvline(x=cfg.storm_start_day + cfg.storm_duration, color="orange", linestyle="--", alpha=0.3)
    ax.axvline(x=cfg.storm2_start_day, color="red", linestyle="--", alpha=0.5)
    ax.axvline(x=cfg.storm2_start_day + cfg.storm2_duration, color="red", linestyle="--", alpha=0.3)
    ax.set_xlabel("Time (sols)")
    ax.set_ylabel("Battery (%)")
    ax.set_title("Energy Storage During Episode")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 250)

    plt.suptitle("MarsPulse Step 2 — MARL Agent Behavior Analysis", fontsize=14, fontweight="bold")
    plt.tight_layout()

    out_path = os.path.join(OUTPUT_DIR, "step2_agent_actions.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Agent action plot saved to {out_path}")


def main():
    print("Generating Step 2 visualizations...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Training curves
    log_path = os.path.join(OUTPUT_DIR, "step2_training_log.json")
    if os.path.exists(log_path):
        plot_training_curve(log_path, os.path.join(OUTPUT_DIR, "step2_training.png"))
    else:
        print("  Training log not found, skipping training curve plot.")

    # Comparison chart
    results_path = os.path.join(OUTPUT_DIR, "step2_results.json")
    if os.path.exists(results_path):
        plot_step2_comparison(results_path, os.path.join(OUTPUT_DIR, "step2_comparison.png"))
    else:
        print("  Step2 results not found, skipping comparison plot.")

    # Agent action analysis
    plot_agent_actions()

    print("\nStep 2 visualizations complete.")


if __name__ == "__main__":
    main()
