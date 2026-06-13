"""
MarsPulse — 步骤3：敏感性分析与故障研究
=======================================
模块说明: 四项分析——(1)参数敏感性扫描(风暴tau/电池容量/RTG功率);
         (2)故障模式影响矩阵(4种单一+6种双重组合);
         (3)生存概率曲面(2D网格扫描); (4)相变分析与资源相关性。
         输出step3_sensitivity.json/failures.json/surface.json/phase.json及对应PNG。
作者: 曾皓轩, 毛元君
开发时间: 2026年5月
"""

import sys, os, json, time, itertools
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mars_pulse.config import MarsBaseConfig
from mars_pulse.environment import MarsBaseEnv
from mars_pulse.agents.static_agent import StaticRuleAgent, StaticRuleAgentConservative

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output")


# ─── Single episode runner ───────────────────────────────────────────

def run_episode_static(env: MarsBaseEnv, agent_cls, seed: int = 42):
    env.reset(seed=seed)
    agent = agent_cls(env)
    obs, _ = env.reset(seed=seed)
    S_hist, E_hist, O_hist, T_hist = [], [], [], []
    for _ in range(env.max_steps):
        action = agent.select_action(obs)
        obs, _, terminated, truncated, info = env.step(action)
        S_hist.append(info["S"])
        E_hist.append(info["energy_stored"])
        O_hist.append(info["oxygen_stored"])
        T_hist.append(info["habitat_temp"])
        if terminated or truncated:
            break
    return {
        "survival": info["survival_sols"],
        "terminated": terminated,
        "S_min": float(min(S_hist)),
        "S_mean": float(np.mean(S_hist)),
        "energy_min": float(min(E_hist)),
        "O2_min": float(min(O_hist)),
        "temp_min": float(min(T_hist)),
        "temp_max": float(max(T_hist)),
    }


# ─── 1. Parameter Sensitivity Analysis ───────────────────────────────

def sensitivity_analysis():
    """Sweep key parameters and measure survival time for static agents."""
    print("=" * 60)
    print("  1. Parameter Sensitivity Analysis")
    print("=" * 60)

    # Sweep ranges scaled for 300-sol / 3000 kWh baseline
    tau_storm_values = [1.5, 2.0, 2.5, 3.0, 3.5]
    battery_values = [1000, 2000, 3000, 4000, 5000]
    rtg_values = [4.0, 8.0, 10.0, 12.0, 15.0]

    results = {"tau_storm": {}, "battery": {}, "rtg": {}}

    # --- Storm severity sweep ---
    print("\nSweeping storm optical depth (tau_storm)...")
    for tau in tau_storm_values:
        cfg = MarsBaseConfig()
        cfg.tau_storm = tau
        env = MarsBaseEnv(config=cfg)
        survivals = []
        for seed in range(20):
            r = run_episode_static(env, StaticRuleAgent, seed)
            survivals.append(r["survival"])
        results["tau_storm"][tau] = {
            "mean": round(np.mean(survivals), 2),
            "std": round(np.std(survivals), 2),
            "min": round(np.min(survivals), 2),
        }
        print(f"  tau={tau:.1f}: survival {results['tau_storm'][tau]['mean']:.1f} ± {results['tau_storm'][tau]['std']:.1f} sols")

    # --- Battery capacity sweep ---
    print("\nSweeping battery capacity...")
    for bat in battery_values:
        cfg = MarsBaseConfig()
        cfg.battery_capacity = bat
        env = MarsBaseEnv(config=cfg)
        survivals = []
        for seed in range(20):
            r = run_episode_static(env, StaticRuleAgent, seed)
            survivals.append(r["survival"])
        results["battery"][bat] = {
            "mean": round(np.mean(survivals), 2),
            "std": round(np.std(survivals), 2),
        }
        print(f"  bat={bat} kWh: survival {results['battery'][bat]['mean']:.1f} ± {results['battery'][bat]['std']:.1f} sols")

    # --- RTG power sweep ---
    print("\nSweeping RTG baseline power...")
    for rtg in rtg_values:
        cfg = MarsBaseConfig()
        cfg.P_rtg = rtg
        env = MarsBaseEnv(config=cfg)
        survivals = []
        for seed in range(20):
            r = run_episode_static(env, StaticRuleAgent, seed)
            survivals.append(r["survival"])
        results["rtg"][rtg] = {
            "mean": round(np.mean(survivals), 2),
            "std": round(np.std(survivals), 2),
        }
        print(f"  RTG={rtg} kW: survival {results['rtg'][rtg]['mean']:.1f} ± {results['rtg'][rtg]['std']:.1f} sols")

    # Save
    with open(os.path.join(OUTPUT_DIR, "step3_sensitivity.json"), "w") as f:
        json.dump(results, f, indent=2)

    # --- Plot ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    ax = axes[0]
    xs = list(results["tau_storm"].keys())
    ys = [results["tau_storm"][x]["mean"] for x in xs]
    es = [results["tau_storm"][x]["std"] for x in xs]
    ax.errorbar(xs, ys, yerr=es, marker="o", color="#F44336", linewidth=2, capsize=5)
    ax.set_xlabel("Storm Optical Depth τ")
    ax.set_ylabel("Mean Survival (sols)")
    ax.set_title("Sensitivity: Storm Severity")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    xs = list(results["battery"].keys())
    ys = [results["battery"][x]["mean"] for x in xs]
    es = [results["battery"][x]["std"] for x in xs]
    ax.errorbar(xs, ys, yerr=es, marker="s", color="#2196F3", linewidth=2, capsize=5)
    ax.set_xlabel("Battery Capacity (kWh)")
    ax.set_ylabel("Mean Survival (sols)")
    ax.set_title("Sensitivity: Energy Storage")
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    xs = list(results["rtg"].keys())
    ys = [results["rtg"][x]["mean"] for x in xs]
    es = [results["rtg"][x]["std"] for x in xs]
    ax.errorbar(xs, ys, yerr=es, marker="^", color="#4CAF50", linewidth=2, capsize=5)
    ax.set_xlabel("RTG Power (kW)")
    ax.set_ylabel("Mean Survival (sols)")
    ax.set_title("Sensitivity: RTG Baseline")
    ax.grid(True, alpha=0.3)

    plt.suptitle("MarsPulse Step 3 — Parameter Sensitivity Analysis", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "step3_sensitivity.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"\n  Sensitivity plot saved to {path}")


# ─── 2. Failure Mode Impact Matrix ───────────────────────────────────

def failure_impact_analysis():
    """Test impact of each failure mode individually and in combination."""
    print("\n" + "=" * 60)
    print("  2. Failure Mode Impact Analysis")
    print("=" * 60)

    failure_modes = [
        "electrolysis_degraded",
        "heater_fault",
        "battery_cell_failure",
        "habitat_leak",
    ]
    short_names = ["Electrolysis\nDegraded", "Heater\nFault", "Battery Cell\nFailure", "Habitat\nLeak"]

    impacts = {}

    # Single failure impact
    print("\nSingle failure modes:")
    for mode in failure_modes:
        cfg = MarsBaseConfig()
        env = MarsBaseEnv(config=cfg)
        survivals = []
        for seed in range(20):
            env.reset(seed=seed)
            env.physics.active_failures[mode] = True  # force failure from start
            agent = StaticRuleAgent(env)
            obs, _ = env.reset(seed=seed)
            env.physics.active_failures[mode] = True
            for _ in range(env.max_steps):
                action = agent.select_action(obs)
                obs, _, terminated, truncated, info = env.step(action)
                if terminated or truncated:
                    break
            survivals.append(info["survival_sols"])
        impacts[mode] = {"mean": np.mean(survivals), "std": np.std(survivals)}
        print(f"  {mode}: {impacts[mode]['mean']:.1f} ± {impacts[mode]['std']:.1f} sols")

    # Double combinations
    print("\nDouble failure combinations:")
    combo_impacts = {}
    for m1, m2 in itertools.combinations(failure_modes, 2):
        key = f"{m1}+{m2}"
        survivals = []
        for seed in range(20):
            cfg = MarsBaseConfig()
            env = MarsBaseEnv(config=cfg)
            env.reset(seed=seed)
            env.physics.active_failures[m1] = True
            env.physics.active_failures[m2] = True
            agent = StaticRuleAgent(env)
            obs, _ = env.reset(seed=seed)
            env.physics.active_failures[m1] = True
            env.physics.active_failures[m2] = True
            for _ in range(env.max_steps):
                action = agent.select_action(obs)
                obs, _, terminated, truncated, info = env.step(action)
                if terminated or truncated:
                    break
            survivals.append(info["survival_sols"])
        combo_impacts[key] = {"mean": np.mean(survivals), "std": np.std(survivals)}
        print(f"  {key}: {combo_impacts[key]['mean']:.1f} ± {combo_impacts[key]['std']:.1f} sols")

    # Baseline (no failures)
    cfg = MarsBaseConfig()
    env = MarsBaseEnv(config=cfg)
    survivals = []
    for seed in range(20):
        r = run_episode_static(env, StaticRuleAgent, seed)
        survivals.append(r["survival"])
    baseline = np.mean(survivals)

    # Plot
    fig, ax = plt.subplots(figsize=(14, 7))

    # Single failures
    x_pos = list(range(len(failure_modes)))
    means = [impacts[m]["mean"] for m in failure_modes]
    stds = [impacts[m]["std"] for m in failure_modes]
    bars = ax.bar(x_pos, means, color=["#E53935", "#FB8C00", "#FDD835", "#43A047"],
                  edgecolor="black", linewidth=0.5)
    ax.errorbar(x_pos, means, yerr=stds, fmt="none", ecolor="black", capsize=5)
    ax.axhline(y=baseline, color="blue", linestyle="--", linewidth=1.5, label=f"No-failure baseline ({baseline:.1f} sols)")

    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{m:.1f}", ha="center", fontweight="bold", fontsize=10)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(short_names, fontsize=10)
    ax.set_ylabel("Mean Survival (sols)")
    ax.set_title("Failure Mode Impact Analysis — Single Failures")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2, axis="y")
    ax.set_ylim(0, max(means) * 1.2)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "step3_failure_impact.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"\n  Failure impact plot saved to {path}")

    # Save data
    with open(os.path.join(OUTPUT_DIR, "step3_failures.json"), "w") as f:
        json.dump({"single": impacts, "double": combo_impacts, "baseline": baseline}, f, indent=2)


# ─── 3. Survival Probability Surface ─────────────────────────────────

def survival_surface():
    """2D sweep: vary storm_severity × battery_capacity, map survival surface."""
    print("\n" + "=" * 60)
    print("  3. Survival Probability Surface")
    print("=" * 60)

    tau_vals = np.linspace(1.0, 3.5, 8)
    batt_vals = np.linspace(500, 5000, 8)
    surface = np.zeros((len(tau_vals), len(batt_vals)))

    print(f"Sweeping {len(tau_vals)}×{len(batt_vals)} grid...")
    for i, tau in enumerate(tau_vals):
        for j, bat in enumerate(batt_vals):
            cfg = MarsBaseConfig()
            cfg.tau_storm = float(tau)
            cfg.battery_capacity = float(bat)
            env = MarsBaseEnv(config=cfg)
            survivals = []
            for seed in range(8):
                r = run_episode_static(env, StaticRuleAgent, seed)
                survivals.append(min(r["survival"], cfg.episode_length))
            surface[i, j] = np.mean(survivals)
        print(f"  Row {i+1}/{len(tau_vals)}: tau={tau:.2f}, survivals={surface[i,:].round(1)}")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    X, Y = np.meshgrid(batt_vals, tau_vals)
    cmap = LinearSegmentedColormap.from_list("survival", ["#D32F2F", "#FFC107", "#4CAF50"], N=256)
    im = ax.pcolormesh(X, Y, surface, cmap=cmap, shading="auto")
    cs = ax.contour(X, Y, surface, levels=[30, 60, 90, 120, 150, 200, 250],
                     colors="black", linewidths=0.8)
    ax.clabel(cs, inline=True, fontsize=9, fmt="%.0f sols")

    # Mark baseline
    ax.plot(3000, 2.5, "o", color="white", markersize=12, markeredgecolor="black", markeredgewidth=2)
    ax.annotate("Baseline\n(3000 kWh, τ=2.5)", (3000, 2.5), textcoords="offset points",
                xytext=(15, 15), fontsize=10, color="white", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="white"))

    cbar = plt.colorbar(im, ax=ax, label="Mean Survival (sols)")
    ax.set_xlabel("Battery Capacity (kWh)")
    ax.set_ylabel("Storm Optical Depth τ")
    ax.set_title("Survival Probability Surface\nStatic Rules Agent — Storm Severity × Energy Storage", fontsize=13, fontweight="bold")

    path = os.path.join(OUTPUT_DIR, "step3_survival_surface.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"\n  Survival surface saved to {path}")

    # Save surface data
    surface_data = {
        "tau_vals": tau_vals.tolist(),
        "batt_vals": batt_vals.tolist(),
        "surface": surface.tolist(),
    }
    with open(os.path.join(OUTPUT_DIR, "step3_surface.json"), "w") as f:
        json.dump(surface_data, f, indent=2)


# ─── 4. Resource Correlation & Phase Transition ─────────────────────

def phase_transition_analysis():
    """Analyze the phase transition: when does the system shift from stable → collapse?"""
    print("\n" + "=" * 60)
    print("  4. Phase Transition & Resource Correlation")
    print("=" * 60)

    cfg = MarsBaseConfig()
    env = MarsBaseEnv(config=cfg)

    # Run many episodes with different seeds, collect full trajectory data
    all_trajectories = []
    for seed in range(50):
        env.reset(seed=seed)
        agent = StaticRuleAgent(env)
        obs, _ = env.reset(seed=seed)
        traj = {"seed": seed, "S": [], "E": [], "O2": [], "T": [], "time": [], "alive": True}
        for _ in range(env.max_steps):
            action = agent.select_action(obs)
            obs, _, terminated, truncated, info = env.step(action)
            traj["S"].append(float(info["S"]))
            traj["E"].append(float(info["energy_stored"]))
            traj["O2"].append(float(info["oxygen_stored"]))
            traj["T"].append(float(info["habitat_temp"]))
            traj["time"].append(float(info["survival_sols"]))
            if terminated:
                traj["alive"] = False
                break
        all_trajectories.append(traj)
    print(f"  Collected {len(all_trajectories)} trajectories")

    # Find phase transition point for each trajectory: when S(t) first drops below S_critical
    transitions = []
    for traj in all_trajectories:
        for t, S in enumerate(traj["S"]):
            if S < cfg.S_critical:
                transitions.append(traj["time"][t])
                break
        else:
            transitions.append(traj["time"][-1])  # never crossed

    # Correlation matrix: energy vs O2, energy vs temp, O2 vs temp
    all_E, all_O2, all_T = [], [], []
    for traj in all_trajectories:
        all_E.extend(traj["E"])
        all_O2.extend(traj["O2"])
        all_T.extend(traj["T"])

    corr_matrix = np.corrcoef([all_E, all_O2, all_T])

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Phase transition histogram
    ax = axes[0]
    ax.hist(transitions, bins=25, color="#FF9800", edgecolor="black", alpha=0.8)
    ax.axvline(x=cfg.storm_start_day, color="red", linestyle="--", linewidth=2, label=f"Storm onset (sol {cfg.storm_start_day})")
    ax.axvline(x=np.mean(transitions), color="blue", linestyle="--", linewidth=2,
               label=f"Mean transition ({np.mean(transitions):.1f} sols)")
    ax.set_xlabel("Time (sols)")
    ax.set_ylabel("Frequency")
    ax.set_title("Phase Transition: S(t) Drops Below Critical Threshold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Correlation heatmap
    ax = axes[1]
    im = ax.imshow(corr_matrix, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["Energy", "Oxygen", "Temperature"])
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["Energy", "Oxygen", "Temperature"])
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{corr_matrix[i, j]:.3f}", ha="center", va="center",
                    fontsize=12, fontweight="bold",
                    color="white" if abs(corr_matrix[i, j]) > 0.5 else "black")
    plt.colorbar(im, ax=ax, label="Pearson r")
    ax.set_title("Resource Correlation Matrix")

    plt.suptitle("MarsPulse Step 3 — Phase Transition & Resource Correlation", fontsize=14, fontweight="bold")
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, "step3_phase_transition.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Phase transition plot saved to {path}")

    # Save
    pt_data = {
        "transition_times": [float(t) for t in transitions],
        "mean_transition": float(np.mean(transitions)),
        "std_transition": float(np.std(transitions)),
        "correlation_matrix": corr_matrix.tolist(),
    }
    with open(os.path.join(OUTPUT_DIR, "step3_phase.json"), "w") as f:
        json.dump(pt_data, f, indent=2)


# ─── Main ─────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  MarsPulse Step 3 — Advanced Analysis & Sensitivity Studies")
    print("=" * 70)
    print()

    t0 = time.time()

    sensitivity_analysis()
    failure_impact_analysis()
    survival_surface()
    phase_transition_analysis()

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"  Step 3 complete in {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"{'='*70}")


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    main()
