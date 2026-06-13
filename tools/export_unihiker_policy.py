"""
MarsPulse — 行空板策略导出工具
==============================
模块说明: 将桌面端训练的PyTorch MAPPO模型(mappo_model.pt)导出为NumPy格式(.npz)，
         供行空板(UniHiker)嵌入式平台在不依赖PyTorch的情况下进行推理。
         导出内容包括三个智能体的网络权重(weight/bias)及动作边界参数。
作者: 陈凯熙
开发时间: 2026年5月
"""

# -*- coding: utf-8 -*-
"""Export the trained MarsPulse MAPPO actors for UniHiker inference.

The UniHiker demo should not need PyTorch at runtime. This script loads the
desktop-trained ``output/mappo_model.pt`` checkpoint and writes a compact
NumPy ``.npz`` file containing only the deterministic actor weights.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = PROJECT_ROOT / "output" / "mappo_model.pt"
DEFAULT_OUTPUT = PROJECT_ROOT / "unihiker" / "mappo_policy_unihiker.npz"


def tensor_to_numpy(value):
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy().astype(np.float32)
    return np.asarray(value, dtype=np.float32)


def export_actor_state(checkpoint_path: Path, output_path: Path) -> dict:
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    actors = checkpoint["actors"]

    arrays = {}
    manifest = {
        "source_checkpoint": str(checkpoint_path),
        "epoch": int(checkpoint.get("epoch", -1)),
        "global_step": int(checkpoint.get("global_step", -1)),
        "best_survival": float(checkpoint.get("best_survival", 0.0)),
        "actors": [],
    }

    for agent_id in range(3):
        prefix = f"{agent_id}."
        actor_info = {
            "agent_id": agent_id,
            "obs_dim": int(actors[f"{prefix}net.0.weight"].shape[1]),
            "action_low": float(tensor_to_numpy(actors[f"{prefix}action_low"])),
            "action_high": float(tensor_to_numpy(actors[f"{prefix}action_high"])),
        }
        manifest["actors"].append(actor_info)

        for name in (
            "action_low",
            "action_high",
            "action_scale",
            "action_bias",
            "net.0.weight",
            "net.0.bias",
            "net.2.weight",
            "net.2.bias",
            "net.4.weight",
            "net.4.bias",
        ):
            arrays[f"a{agent_id}_{name.replace('.', '_')}"] = tensor_to_numpy(
                actors[f"{prefix}{name}"]
            )

    arrays["manifest_json"] = np.asarray(json.dumps(manifest), dtype=np.str_)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(output_path), **arrays)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    manifest = export_actor_state(args.checkpoint, args.output)
    print(f"Exported UniHiker policy to: {args.output}")
    print(
        "Checkpoint: epoch={epoch}, step={global_step}, best_survival={best_survival:.2f}".format(
            **manifest
        )
    )
    for actor in manifest["actors"]:
        print(
            "Agent {agent_id}: obs_dim={obs_dim}, action=[{action_low:.1f}, {action_high:.1f}]".format(
                **actor
            )
        )


if __name__ == "__main__":
    main()
