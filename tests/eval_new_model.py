"""
MarsPulse — 新模型评估脚本
==========================
模块说明: 对最新训练的MAPPO模型与静态基线进行50回合对比评估。
         输出new_model_results.json，含生存时间、死亡率、改善百分比。
作者: 曾皓轩, 陈凯熙
开发时间: 2026年5月
"""

import sys, os, json, time, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mars_pulse.config import MarsBaseConfig
from mars_pulse.environment import MarsBaseEnv
from mars_pulse.multi_agent_env import MultiAgentMarsEnv
from mars_pulse.agents.static_agent import StaticRuleAgent
from mars_pulse.agents.mappo import MAPPO

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output")
cfg = MarsBaseConfig()
N_EP = 50

def eval_static(n=N_EP):
    env = MarsBaseEnv(config=cfg)
    surv, term = [], []
    for s in range(n):
        obs, _ = env.reset(seed=s*137+42)
        agent = StaticRuleAgent(env)
        for _ in range(env.max_steps):
            obs, _, terminated, truncated, info = env.step(agent.select_action(obs))
            if terminated or truncated:
                break
        surv.append(info["survival_sols"]); term.append(terminated)
    return {"mean": np.mean(surv), "std": np.std(surv), "fatal": np.mean(term),
            "min": np.min(surv), "max": np.max(surv)}

def eval_mappo(n=N_EP):
    env = MarsBaseEnv(config=cfg)
    multi_env = MultiAgentMarsEnv(config=cfg)
    mappo = MAPPO(multi_env.obs_dims, [1,1,1], multi_env.global_obs_dim,
                  action_bounds=multi_env.action_bounds)
    ckpt = torch.load(os.path.join(OUTPUT_DIR, "mappo_model.pt"),
                      map_location="cpu", weights_only=False)
    mappo.actors.load_state_dict(ckpt["actors"])

    surv, term = [], []
    for s in range(n):
        obs, _ = env.reset(seed=s*251+17)
        for _ in range(env.max_steps):
            # Use env.get_multi_agent_observations() for proper per-agent obs
            multi_obs = env.get_multi_agent_observations()
            acts, _, _, _ = mappo.select_actions(multi_obs, deterministic=True)
            action = np.array([float(acts[0][0]), float(acts[1][0]), float(acts[2][0])], dtype=np.float32)
            obs, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        surv.append(info["survival_sols"]); term.append(terminated)
    return {"mean": np.mean(surv), "std": np.std(surv), "fatal": np.mean(term),
            "min": np.min(surv), "max": np.max(surv)}

if __name__ == "__main__":
    print(f"Evaluating on {cfg.episode_length}-sol scenario ({N_EP} episodes each)...")
    print(f"Storms: sol {cfg.storm_start_day}-{cfg.storm_start_day+cfg.storm_duration} "
          f"(tau={cfg.tau_storm}), sol {cfg.storm2_start_day}-{cfg.storm2_start_day+cfg.storm2_duration} "
          f"(tau={cfg.storm2_tau})")
    print(f"Battery: {cfg.battery_capacity} kWh, Solar: {cfg.P_max} kW, RTG: {cfg.P_rtg} kW")
    print()

    t0 = time.time()
    r_static = eval_static()
    print(f"Static Rules:     {r_static['mean']:6.1f} ± {r_static['std']:5.1f} sols  "
          f"(fatal {r_static['fatal']:.1%}, min={r_static['min']:.0f}, max={r_static['max']:.0f})")

    r_mappo = eval_mappo()
    print(f"MAPPO MARL:       {r_mappo['mean']:6.1f} ± {r_mappo['std']:5.1f} sols  "
          f"(fatal {r_mappo['fatal']:.1%}, min={r_mappo['min']:.0f}, max={r_mappo['max']:.0f})")
    elapsed = time.time() - t0

    improvement = (r_mappo['mean'] - r_static['mean']) / r_static['mean'] * 100
    print(f"\nImprovement: {improvement:+.1f}%")
    print(f"Static → MAPPO: {r_static['mean']:.0f} → {r_mappo['mean']:.0f} sols")
    print(f"Elapsed: {elapsed:.0f}s")

    results = {"static": {k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                          for k, v in r_static.items()},
               "mappo": {k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                         for k, v in r_mappo.items()},
               "improvement_pct": round(improvement, 1),
               "config": {"episode": cfg.episode_length, "battery": cfg.battery_capacity,
                          "solar": cfg.P_max, "rtg": cfg.P_rtg}}
    with open(os.path.join(OUTPUT_DIR, "new_model_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to output/new_model_results.json")
