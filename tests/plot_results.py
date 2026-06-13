"""
MarsPulse — 步骤1：基线可视化
=============================
模块说明: 生成资源轨迹图(能量/O2/温度/S(t)×时间，含双风暴标注)、
         功率平衡图(输入vs消耗)、基线对比柱状图和生存时间分布直方图。
         输出step1_trajectories.png/power_balance.png/comparison.png/histogram.png。
作者: 毛元君
开发时间: 2026年4–5月
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
from mars_pulse.agents.static_agent import StaticRuleAgent, StaticRuleAgentConservative

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output")


def run_detailed_episode(env, agent, max_steps=None):
    """Run an episode and record full state history."""
    if max_steps is None:
        max_steps = env.max_steps
    obs, _ = env.reset()
    history = {
        "time": [], "energy": [], "oxygen": [], "temp": [], "S": [],
        "power_in": [], "power_draw": [], "storm": [], "failures": [],
        "life_support": [], "heating": [], "experiments": [],
    }
    for _ in range(max_steps):
        action = agent.select_action(obs)
        obs, reward, terminated, truncated, info = env.step(action)

        history["time"].append(float(info["survival_sols"]))
        history["energy"].append(float(info["energy_stored"]))
        history["oxygen"].append(float(info["oxygen_stored"]))
        history["temp"].append(float(info["habitat_temp"]))
        history["S"].append(float(info["S"]))
        history["power_in"].append(float(env.physics.power_in))
        history["power_draw"].append(float(env.physics.power_consumed))
        history["storm"].append(float(info["storm_active"]))
        history["failures"].append(sum(info["failures"].values()))
        history["life_support"].append(float(action[0]))
        history["heating"].append(float(action[1]))
        history["experiments"].append(float(action[2]))

        if terminated or truncated:
            break
    return history


def add_storm_markers(ax, cfg):
    """Add storm shading and lines for two-storm scenario."""
    ax.axvline(x=cfg.storm_start_day, color="red", linestyle="--", alpha=0.6, linewidth=1.2, label="Storm1")
    ax.axvline(x=cfg.storm_start_day + cfg.storm_duration, color="red", linestyle="--", alpha=0.3, linewidth=1.0)
    ax.fill_between([cfg.storm_start_day, cfg.storm_start_day+cfg.storm_duration], *ax.get_ylim(),
                    color="orange", alpha=0.08)
    ax.axvline(x=cfg.storm2_start_day, color="darkred", linestyle="--", alpha=0.6, linewidth=1.2, label="Storm2")
    ax.axvline(x=cfg.storm2_start_day + cfg.storm2_duration, color="darkred", linestyle="--", alpha=0.3, linewidth=1.0)
    ax.fill_between([cfg.storm2_start_day, cfg.storm2_start_day+cfg.storm2_duration], *ax.get_ylim(),
                    color="red", alpha=0.06)


def plot_resource_trajectories(histories: dict, cfg: MarsBaseConfig, path: str):
    """Plot energy, oxygen, temperature, and S(t) over time for each agent."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    colors = {"Static Rules": "#2196F3", "Static Conservative": "#4CAF50"}

    # Energy
    ax = axes[0, 0]
    for label, h in histories.items():
        t = np.array(h["time"])
        ax.plot(t, np.array(h["energy"]) / cfg.battery_capacity * 100,
                color=colors.get(label, "gray"), label=label, linewidth=1.5)
    add_storm_markers(ax, cfg)
    ax.set_xlabel("Time (sols)")
    ax.set_ylabel("Battery (%)")
    ax.set_title("Energy Storage")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 250)

    # Oxygen
    ax = axes[0, 1]
    for label, h in histories.items():
        ax.plot(h["time"], np.array(h["oxygen"]) / cfg.O2_storage_capacity * 100,
                color=colors.get(label, "gray"), label=label, linewidth=1.5)
    add_storm_markers(ax, cfg)
    ax.set_xlabel("Time (sols)")
    ax.set_ylabel("O2 Storage (%)")
    ax.set_title("Oxygen Storage")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 250)

    # Temperature
    ax = axes[1, 0]
    for label, h in histories.items():
        ax.plot(h["time"], h["temp"], color=colors.get(label, "gray"),
                label=label, linewidth=1.5)
    ax.axhline(y=cfg.target_habitat_temp, color="black", linestyle=":", alpha=0.5, label=f"Target ({cfg.target_habitat_temp}°C)")
    ax.axhline(y=cfg.temp_min_survivable, color="blue", linestyle="--", alpha=0.4, label=f"Min survivable")
    ax.axhline(y=cfg.temp_max_survivable, color="red", linestyle="--", alpha=0.4, label=f"Max survivable")
    add_storm_markers(ax, cfg)
    ax.set_xlabel("Time (sols)")
    ax.set_ylabel("Temperature (°C)")
    ax.set_title("Habitat Temperature")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 250)

    # Survival Stability Index
    ax = axes[1, 1]
    for label, h in histories.items():
        ax.plot(h["time"], h["S"], color=colors.get(label, "gray"),
                label=label, linewidth=1.5)
    ax.axhline(y=cfg.S_critical, color="orange", linestyle="--", alpha=0.5, label=f"S_critical ({cfg.S_critical})")
    ax.axhline(y=cfg.S_fatal, color="red", linestyle="--", alpha=0.5, label=f"S_fatal ({cfg.S_fatal})")
    add_storm_markers(ax, cfg)
    ax.set_xlabel("Time (sols)")
    ax.set_ylabel("S(t)")
    ax.set_title("Survival Stability Index")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 250)

    plt.suptitle("MarsPulse Step 1 — Resource Trajectories During Two-Storm Scenario", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Trajectory plot saved to {path}")


def plot_power_balance(histories: dict, cfg: MarsBaseConfig, path: str):
    """Plot power input vs consumption and agent actions."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    h = histories.get("Static Rules", list(histories.values())[0])

    # Power balance
    ax = axes[0]
    ax.fill_between(h["time"], 0, h["power_in"], alpha=0.3, color="green", label="Power In (Solar+RTG)")
    ax.plot(h["time"], h["power_draw"], color="red", linewidth=1.5, label="Power Draw")
    add_storm_markers(ax, cfg)
    ax.set_xlabel("Time (sols)")
    ax.set_ylabel("Power (kW)")
    ax.set_title("Power Balance — Static Rules Agent")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 250)

    # Agent actions
    ax = axes[1]
    ax.plot(h["time"], h["life_support"], color="blue", linewidth=1.2, label="Life Support (kW)")
    ax.plot(h["time"], h["heating"], color="red", linewidth=1.2, label="Heating (kW)")
    ax.plot(h["time"], h["experiments"], color="purple", linewidth=1.2, label="Experiments (kW)")
    add_storm_markers(ax, cfg)
    ax.set_xlabel("Time (sols)")
    ax.set_ylabel("Power (kW)")
    ax.set_title("Agent Action — Power Allocation")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 250)

    plt.suptitle("MarsPulse Step 1 — Power Allocation Analysis (Two-Storm Scenario)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Power balance plot saved to {path}")


def plot_survival_histogram(results_path: str, output_path: str):
    """Load Monte Carlo results and plot survival time comparison + distribution."""
    with open(results_path) as f:
        data = json.load(f)
    if isinstance(data, dict) and "results" in data:
        aggregates = data["results"]
    else:
        aggregates = data

    fig = plt.figure(figsize=(18, 7))
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1, 1.2])

    # Left: bar chart — survival time + fatality rate
    ax = fig.add_subplot(gs[0])
    labels = [a["agent"] for a in aggregates]
    means = [a["survival_sols_mean"] for a in aggregates]
    stds = [a["survival_sols_std"] for a in aggregates]
    fatalities = [a["fatality_rate"] * 100 for a in aggregates]
    survivals = [a["full_survival_rate"] * 100 for a in aggregates]

    # Only show Static agents, skip Random
    static_labels = [l for l in labels if "Random" not in l]
    static_idx = [i for i, l in enumerate(labels) if "Random" not in l]
    x = np.arange(len(static_labels))
    width = 0.2

    colors = ["#2196F3", "#4CAF50"]
    bar_colors = [colors[i % len(colors)] for i in range(len(static_labels))]

    bars1 = ax.bar(x - width, [means[i] for i in static_idx], width,
                   color=bar_colors, label="Mean Survival (sols)",
                   edgecolor="black", linewidth=0.5)
    ax.set_ylabel("Survival Time (sols)", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(static_labels, fontsize=10)

    ax2 = ax.twinx()
    bars2 = ax2.bar(x, [fatalities[i] for i in static_idx], width,
                    color=["#FF5252", "#FF5252"],
                    label="Fatality Rate (%)", edgecolor="black", linewidth=0.5, alpha=0.6)
    bars3 = ax2.bar(x + width, [survivals[i] for i in static_idx], width,
                    color=["#4CAF50", "#4CAF50"],
                    label="Full Survival (%)", edgecolor="black", linewidth=0.5, alpha=0.6)
    ax2.set_ylabel("Rate (%)", fontsize=11)

    # Add error bars manually
    for i, idx in enumerate(static_idx):
        ax.errorbar(x[i] - width, means[idx], yerr=stds[idx],
                    fmt="none", ecolor="black", capsize=4)

    for bar, idx in zip(bars1, static_idx):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{means[idx]:.1f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    for bar, idx in zip(bars2, static_idx):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 f"{fatalities[idx]:.1f}%", ha="center", va="bottom", fontsize=8, color="red")
    for bar, idx in zip(bars3, static_idx):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 f"{survivals[idx]:.1f}%", ha="center", va="bottom", fontsize=8, color="green")

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=8)
    ax.set_title("Step 1 — Baseline Agent Comparison", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.2, axis="y")

    # Right: summary table
    ax_table = fig.add_subplot(gs[1])
    ax_table.axis("off")

    table_data = []
    col_labels = ["Agent", "Survival (sols)", "Fatality %", "Full Surv %", "S(t) Mean", "Reward"]
    for a in aggregates:
        table_data.append([
            a["agent"],
            f'{a["survival_sols_mean"]:.1f} ± {a["survival_sols_std"]:.1f}',
            f'{a["fatality_rate"]*100:.1f}%',
            f'{a["full_survival_rate"]*100:.1f}%',
            f'{a["S_mean_of_means"]:.4f}',
            f'{a["total_reward_mean"]:.1f}',
        ])

    table = ax_table.table(cellText=table_data, colLabels=col_labels,
                           loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.8)
    # Style header
    for j in range(len(col_labels)):
        table[0, j].set_facecolor("#1976D2")
        table[0, j].set_text_props(color="white", fontweight="bold")
    for i in range(len(table_data)):
        for j in range(len(col_labels)):
            table[i+1, j].set_facecolor("#F5F5F5" if i % 2 == 0 else "white")
    ax_table.set_title("Detailed Metrics Summary", fontsize=12, fontweight="bold", y=0.98)

    plt.suptitle("MarsPulse Step 1 — Complete Baseline Comparison (50 Monte Carlo episodes)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Comparison chart saved to {output_path}")


def main():
    print("Generating Step 1 visualizations...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    cfg = MarsBaseConfig()

    # Run detailed episodes for trajectory plots with multiple seeds
    env = MarsBaseEnv(config=cfg)
    agents = {
        "Static Rules": StaticRuleAgent(env),
        "Static Conservative": StaticRuleAgentConservative(env),
    }

    # Use seed=1644 (full 300-sol survival for both agents) for complete-range trajectory
    histories = {}
    for label, agent in agents.items():
        env.reset(seed=1644)
        h = run_detailed_episode(env, agent)
        histories[label] = h
        print(f"  {label} (seed=1644): survived {h['time'][-1]:.1f} sols, "
              f"final S={h['S'][-1]:.4f}, "
              f"final energy={h['energy'][-1]:.1f} kWh")

    plot_resource_trajectories(histories, cfg,
                               os.path.join(OUTPUT_DIR, "step1_trajectories.png"))
    plot_power_balance(histories, cfg,
                       os.path.join(OUTPUT_DIR, "step1_power_balance.png"))
    plot_survival_histogram(os.path.join(OUTPUT_DIR, "step1_results.json"),
                            os.path.join(OUTPUT_DIR, "step1_comparison.png"))

    print("\nAll Step 1 plots generated in output/")


if __name__ == "__main__":
    main()
