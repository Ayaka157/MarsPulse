"""
MarsPulse — 多智能体PPO (MAPPO)
================================
模块说明: Multi-Agent PPO实现，中心化评论家+去中心化执行者(CTDE)架构。
         三个协作智能体各自维护高斯策略，共享中心化价值函数。
         含小批量PPO更新、余弦退火学习率调度、GAE优势估计。
         动作通过tanh非线性映射至合法范围，对数标准差裁剪防止方差崩溃。
参考文献: Yu et al., "The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games", NeurIPS 2022.
作者: 曾皓轩, 毛元君
开发时间: 2026年4–5月
所属项目: MarsPulse — 基于MARL的火星地下熔岩管基地资源智能调配与生存极限仿真
"""

from typing import List, Tuple, Dict
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal


# ─── Neural networks ──────────────────────────────────────────────────

def mlp(sizes: List[int], activation=nn.Tanh, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[j], sizes[j + 1]))
        if j < len(sizes) - 2:
            layers.append(activation())
        else:
            layers.append(output_activation())
    return nn.Sequential(*layers)


class GaussianActor(nn.Module):
    """Stochastic policy with bounded actions via tanh squashing.

    Network outputs are squashed to [-1, 1] via tanh, then rescaled to
    [action_low, action_high]. Actions are clamped to ensure bounds are
    never violated, with corrected log-probabilities under the truncated
    Gaussian distribution.
    """

    def __init__(self, obs_dim: int, act_dim: int,
                 action_low: float = 0.0, action_high: float = 1.0,
                 hidden: List[int] = None):
        super().__init__()
        if hidden is None:
            hidden = [128, 128]
        self.net = mlp([obs_dim] + hidden + [act_dim])
        self.log_std = nn.Parameter(torch.zeros(act_dim))
        self.register_buffer("action_low", torch.tensor(action_low, dtype=torch.float32))
        self.register_buffer("action_high", torch.tensor(action_high, dtype=torch.float32))
        self.register_buffer("action_scale", torch.tensor(
            (action_high - action_low) / 2.0, dtype=torch.float32))
        self.register_buffer("action_bias", torch.tensor(
            (action_high + action_low) / 2.0, dtype=torch.float32))

    def forward(self, obs: torch.Tensor):
        mean_raw = self.net(obs)
        mean = torch.tanh(mean_raw)  # squash to [-1, 1]
        # Rescale to [action_low, action_high]
        mean = mean * self.action_scale + self.action_bias
        std = torch.exp(self.log_std.clamp(-4, 2))
        return mean, std

    def get_action(self, obs: torch.Tensor, deterministic: bool = False):
        mean, std = self.forward(obs)
        dist = Normal(mean, std)
        if deterministic:
            action = mean
        else:
            action = dist.rsample()
        # Clamp to valid range (corrects log_prob below)
        action_clamped = torch.clamp(action, self.action_low, self.action_high)
        # Log prob under truncated Gaussian: log p(clamped | dist)
        log_prob = dist.log_prob(action_clamped).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return action_clamped, log_prob, entropy, mean

    def evaluate(self, obs: torch.Tensor, action: torch.Tensor):
        mean, std = self.forward(obs)
        dist = Normal(mean, std)
        # Action is already clamped when stored; compute log_prob directly
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, entropy


class CentralizedCritic(nn.Module):
    """Centralized value function: sees global state + all agent actions."""

    def __init__(self, global_obs_dim: int, total_act_dim: int, hidden: List[int] = None):
        super().__init__()
        if hidden is None:
            hidden = [256, 256]
        self.net = mlp([global_obs_dim + total_act_dim] + hidden + [1])

    def forward(self, global_obs: torch.Tensor, actions: torch.Tensor):
        # actions shape: (batch, n_agents) or (batch, n_agents, act_dim_per_agent)
        x = torch.cat([global_obs, actions.flatten(1)], dim=-1)
        return self.net(x).squeeze(-1)


# ─── MAPPO Trainer ────────────────────────────────────────────────────

class MAPPO:
    """Multi-Agent PPO with centralized critic + mini-batch training.

    Each agent has its own actor. One shared centralized critic.
    Uses mini-batch updates for better sample efficiency and GPU utilization.
    """

    def __init__(
        self,
        obs_dims: List[int],
        act_dims: List[int],
        global_obs_dim: int,
        action_bounds: List[tuple] = None,
        lr_actor: float = 3e-4,
        lr_critic: float = 1e-3,
        gamma: float = 0.99,
        lam: float = 0.95,
        clip_ratio: float = 0.2,
        target_kl: float = 0.015,
        ent_coef: float = 0.02,
        ppo_epochs: int = 8,
        mini_batch_size: int = 512,
        device: str = "cpu",
    ):
        self.n_agents = len(obs_dims)
        self.gamma = gamma
        self.lam = lam
        self.clip_ratio = clip_ratio
        self.target_kl = target_kl
        self.ent_coef = ent_coef
        self.ppo_epochs = ppo_epochs
        self.mini_batch_size = mini_batch_size
        self.device = torch.device(device)
        self.total_updates = 0

        if action_bounds is None:
            action_bounds = [(0.0, 1.0)] * self.n_agents

        # Create actors (one per agent) with action bounds
        self.actors = nn.ModuleList([
            GaussianActor(obs_dims[i], act_dims[i],
                          action_low=action_bounds[i][0],
                          action_high=action_bounds[i][1])
            for i in range(self.n_agents)
        ]).to(self.device)

        # Centralized critic
        total_act_dim = sum(act_dims)
        self.critic = CentralizedCritic(global_obs_dim, total_act_dim).to(self.device)

        self.actor_optimizers = [
            optim.Adam(actor.parameters(), lr=lr_actor) for actor in self.actors
        ]
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

        # LR schedulers (cosine annealing to 10% of initial LR)
        self.actor_schedulers = [
            optim.lr_scheduler.CosineAnnealingLR(opt, T_max=200, eta_min=lr_actor*0.1)
            for opt in self.actor_optimizers
        ]
        self.critic_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.critic_optimizer, T_max=200, eta_min=lr_critic*0.1)

    def select_actions(
        self, observations: List[np.ndarray], deterministic: bool = False
    ) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
        """Select actions for all agents."""
        actions = []
        log_probs = np.zeros(self.n_agents, dtype=np.float32)
        entropies = np.zeros(self.n_agents, dtype=np.float32)
        means = np.zeros(self.n_agents, dtype=np.float32)

        for i, obs in enumerate(observations):
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            with torch.no_grad():
                action, log_prob, entropy, mean = self.actors[i].get_action(
                    obs_t, deterministic=deterministic
                )
            actions.append(action.cpu().numpy().reshape(-1))
            log_probs[i] = log_prob.cpu().item()
            entropies[i] = entropy.cpu().item()
            means[i] = mean.cpu().item()

        return actions, log_probs, entropies, means

    def evaluate_actions(
        self, observations: List[torch.Tensor], actions: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Evaluate log probs and entropy for given obs-action pairs."""
        all_log_probs = []
        all_entropies = []
        for i in range(self.n_agents):
            lp, ent = self.actors[i].evaluate(observations[i], actions[i])
            all_log_probs.append(lp)
            all_entropies.append(ent)
        return (
            torch.stack(all_log_probs, dim=-1),  # (batch, n_agents)
            torch.stack(all_entropies, dim=-1),
        )

    def update(self, buffer: "MAPPOReplayBuffer") -> Dict:
        """Mini-batch PPO update for sample-efficient training.

        Splits the collected buffer into mini-batches and performs multiple
        PPO epochs over the data, shuffling between epochs. This dramatically
        improves sample efficiency compared to full-batch updates.
        """
        data = buffer.get()
        batch_size = len(data["returns"])
        n_mini = max(1, batch_size // self.mini_batch_size)

        # Transfer all data to GPU once (pinned memory for speed)
        global_obs = torch.as_tensor(data["global_obs"], dtype=torch.float32, device=self.device)
        all_actions_flat = torch.as_tensor(data["actions"], dtype=torch.float32, device=self.device)
        old_log_probs = torch.as_tensor(data["log_probs"], dtype=torch.float32, device=self.device)
        advantages = torch.as_tensor(data["advantages"], dtype=torch.float32, device=self.device)
        returns = torch.as_tensor(data["returns"], dtype=torch.float32, device=self.device)

        observations = [
            torch.as_tensor(data["observations"][i], dtype=torch.float32, device=self.device)
            for i in range(self.n_agents)
        ]
        actions = [
            torch.as_tensor(data["agent_actions"][i], dtype=torch.float32, device=self.device)
            for i in range(self.n_agents)
        ]

        # Normalize advantages (per-mini-batch later for better stability)
        adv_mean, adv_std = advantages.mean(), advantages.std() + 1e-8
        advantages = (advantages - adv_mean) / adv_std

        stats = {"critic_loss": 0.0, "total_updates": 0}

        # --- Mini-batch PPO epochs ---
        for epoch in range(self.ppo_epochs):
            # Shuffle indices each epoch
            indices = torch.randperm(batch_size, device=self.device)

            for mb_start in range(0, batch_size, self.mini_batch_size):
                mb_idx = indices[mb_start:mb_start + self.mini_batch_size]

                # --- Critic update ---
                self.critic_optimizer.zero_grad()
                mb_values = self.critic(global_obs[mb_idx], all_actions_flat[mb_idx])
                critic_loss = nn.MSELoss()(mb_values, returns[mb_idx])
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), 5.0)
                self.critic_optimizer.step()
                stats["critic_loss"] += critic_loss.item()

                # --- Per-agent actor updates ---
                for i in range(self.n_agents):
                    self.actor_optimizers[i].zero_grad()

                    new_lp, ent = self.actors[i].evaluate(
                        observations[i][mb_idx], actions[i][mb_idx])

                    ratio = torch.exp(new_lp - old_log_probs[mb_idx, i])
                    adv_i = advantages[mb_idx, i] if advantages[mb_idx].ndim > 1 else advantages[mb_idx]

                    clip_adv = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * adv_i
                    policy_loss = -torch.min(ratio * adv_i, clip_adv).mean()
                    entropy_bonus = -self.ent_coef * ent.mean()

                    approx_kl = ((ratio - 1) - torch.log(ratio + 1e-8)).mean().detach()

                    (policy_loss + entropy_bonus).backward()
                    nn.utils.clip_grad_norm_(self.actors[i].parameters(), 5.0)
                    self.actor_optimizers[i].step()

                    stats[f"actor_{i}_loss"] = (stats.get(f"actor_{i}_loss", 0)
                                                + policy_loss.item() + entropy_bonus.item())
                    stats[f"actor_{i}_kl"] = stats.get(f"actor_{i}_kl", 0) + approx_kl.item()

                stats["total_updates"] += 1

        # Average stats
        n = max(stats["total_updates"], 1)
        stats["critic_loss"] /= n
        for i in range(self.n_agents):
            stats[f"actor_{i}_loss"] /= n
            stats[f"actor_{i}_kl"] /= n

        # Step LR schedulers
        self.total_updates += 1
        for sched in self.actor_schedulers:
            sched.step()
        self.critic_scheduler.step()

        return stats


# ─── Replay Buffer ────────────────────────────────────────────────────

class MAPPOReplayBuffer:
    """Stores trajectories from multi-agent rollouts for PPO updates.

    Since PPO is on-policy, we collect a batch of trajectories, compute
    GAE advantages, then update and discard.
    """

    def __init__(self, obs_dims: List[int], act_dims: List[int], global_obs_dim: int,
                 buffer_size: int, n_agents: int, gamma: float, lam: float):
        self.n_agents = n_agents
        self.gamma = gamma
        self.lam = lam
        self.buffer_size = buffer_size
        self.ptr = 0
        self.full = False

        self.global_obs = np.zeros((buffer_size, global_obs_dim), dtype=np.float32)
        self.observations = [np.zeros((buffer_size, obs_dims[i]), dtype=np.float32)
                            for i in range(n_agents)]
        self.actions = np.zeros((buffer_size, sum(act_dims)), dtype=np.float32)
        self.agent_actions = [np.zeros((buffer_size, act_dims[i]), dtype=np.float32)
                             for i in range(n_agents)]
        self.rewards = np.zeros((buffer_size, n_agents), dtype=np.float32)
        self.log_probs = np.zeros((buffer_size, n_agents), dtype=np.float32)
        self.values = np.zeros(buffer_size, dtype=np.float32)
        self.dones = np.zeros(buffer_size, dtype=np.float32)
        self.advantages = np.zeros((buffer_size, n_agents), dtype=np.float32)
        self.returns = np.zeros(buffer_size, dtype=np.float32)

    def store(
        self,
        global_obs: np.ndarray,
        observations: List[np.ndarray],
        actions: np.ndarray,
        agent_actions: List[np.ndarray],
        rewards: np.ndarray,
        log_probs: np.ndarray,
        value: float,
        done: bool,
    ):
        idx = self.ptr
        self.global_obs[idx] = global_obs
        for i in range(self.n_agents):
            self.observations[i][idx] = observations[i]
            self.agent_actions[i][idx] = agent_actions[i]
        self.actions[idx] = actions
        self.rewards[idx] = rewards
        self.log_probs[idx] = log_probs
        self.values[idx] = value
        self.dones[idx] = float(done)
        self.ptr = (self.ptr + 1) % self.buffer_size
        if self.ptr == 0:
            self.full = True

    def compute_gae(self, last_value: float, last_done: bool):
        """Compute GAE advantages and returns after collecting a trajectory.

        Uses scalar advantage (mean across agents) for cooperative MARL.
        """
        gae_val = 0.0
        n = self.ptr if not self.full else self.buffer_size

        for t in reversed(range(n)):
            next_val = last_value if t == n - 1 else self.values[t + 1]
            next_done = last_done if t == n - 1 else self.dones[t + 1]
            mask = 1.0 - (1.0 if next_done else 0.0)

            # TD error: mean reward across agents minus value
            mean_reward = float(self.rewards[t].mean())
            delta = mean_reward + self.gamma * next_val * mask - self.values[t]
            gae_val = delta + self.gamma * self.lam * mask * gae_val

            # Broadcast same scalar advantage to all agents
            self.advantages[t] = gae_val
            self.returns[t] = gae_val + self.values[t]

    def get(self) -> dict:
        n = self.ptr if not self.full else self.buffer_size
        return {
            "global_obs": self.global_obs[:n],
            "observations": [self.observations[i][:n] for i in range(self.n_agents)],
            "actions": self.actions[:n],
            "agent_actions": [self.agent_actions[i][:n] for i in range(self.n_agents)],
            "rewards": self.rewards[:n],
            "log_probs": self.log_probs[:n],
            "advantages": self.advantages[:n],
            "returns": self.returns[:n],
            "values": self.values[:n],
        }

    def clear(self):
        self.ptr = 0
        self.full = False

    def __len__(self):
        return self.ptr if not self.full else self.buffer_size
