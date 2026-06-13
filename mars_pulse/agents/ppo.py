"""
MarsPulse — 单智能体PPO算法
============================
模块说明: 标准Proximal Policy Optimization实现。
         含GaussianActor(Critic)、PPOBuffer(GAE优势估计)。
         用于消融实验中与MAPPO对比的单智能体基线。
         动作空间通过tanh压缩+硬截断确保物理合法性。
参考文献: Schulman et al., "Proximal Policy Optimization Algorithms", arXiv:1707.06347, 2017.
作者: 曾皓轩
开发时间: 2026年4月
所属项目: MarsPulse — 基于MARL的火星地下熔岩管基地资源智能调配与生存极限仿真
"""

from typing import List, Tuple, Dict
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal


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
    """Stochastic policy with bounded actions via tanh squashing + clamp."""

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

    def forward(self, obs):
        mean_raw = self.net(obs)
        mean = torch.tanh(mean_raw) * self.action_scale + self.action_bias
        std = torch.exp(self.log_std.clamp(-4, 2))
        return mean, std

    def get_action(self, obs, deterministic=False):
        mean, std = self.forward(obs)
        dist = Normal(mean, std)
        action = mean if deterministic else dist.rsample()
        action = torch.clamp(action, self.action_low, self.action_high)
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return action, log_prob, entropy, mean

    def evaluate(self, obs, action):
        mean, std = self.forward(obs)
        dist = Normal(mean, std)
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, entropy


class Critic(nn.Module):
    def __init__(self, obs_dim: int, hidden: List[int] = None):
        super().__init__()
        if hidden is None:
            hidden = [256, 256]
        self.net = mlp([obs_dim] + hidden + [1])

    def forward(self, obs):
        return self.net(obs).squeeze(-1)


class PPO:
    """Single-agent PPO for MarsPulse."""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        action_low: float = 0.0,
        action_high: float = 1.0,
        lr_actor: float = 3e-4,
        lr_critic: float = 1e-3,
        gamma: float = 0.99,
        lam: float = 0.95,
        clip_ratio: float = 0.2,
        target_kl: float = 0.015,
        ent_coef: float = 0.02,
        train_iters: int = 10,
        device: str = "cpu",
    ):
        self.gamma = gamma
        self.lam = lam
        self.clip_ratio = clip_ratio
        self.target_kl = target_kl
        self.ent_coef = ent_coef
        self.train_iters = train_iters
        self.device = torch.device(device)

        self.actor = GaussianActor(obs_dim, act_dim,
                                   action_low=action_low, action_high=action_high).to(self.device)
        self.critic = Critic(obs_dim).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

    def select_action(self, obs: np.ndarray, deterministic: bool = False):
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action, log_prob, entropy, mean = self.actor.get_action(obs_t, deterministic)
        return (
            action.cpu().numpy().reshape(-1),
            log_prob.cpu().item(),
            entropy.cpu().item(),
            mean.cpu().numpy().reshape(-1),
        )

    def compute_value(self, obs: np.ndarray) -> float:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            val = self.critic(obs_t)
        return val.item()


class PPOBuffer:
    """On-policy replay buffer for single-agent PPO."""

    def __init__(self, obs_dim: int, act_dim: int, buffer_size: int,
                 gamma: float, lam: float):
        self.gamma = gamma
        self.lam = lam
        self.buffer_size = buffer_size
        self.ptr = 0

        self.obs = np.zeros((buffer_size, obs_dim), dtype=np.float32)
        self.actions = np.zeros((buffer_size, act_dim), dtype=np.float32)
        self.rewards = np.zeros(buffer_size, dtype=np.float32)
        self.log_probs = np.zeros(buffer_size, dtype=np.float32)
        self.values = np.zeros(buffer_size, dtype=np.float32)
        self.dones = np.zeros(buffer_size, dtype=np.float32)
        self.advantages = np.zeros(buffer_size, dtype=np.float32)
        self.returns = np.zeros(buffer_size, dtype=np.float32)

    def store(self, obs, action, reward, log_prob, value, done):
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.log_probs[self.ptr] = log_prob
        self.values[self.ptr] = value
        self.dones[self.ptr] = float(done)
        self.ptr += 1

    def compute_gae(self, last_value: float, last_done: bool):
        gae = 0.0
        for t in reversed(range(self.ptr)):
            next_val = last_value if t == self.ptr - 1 else self.values[t + 1]
            next_done = last_done if t == self.ptr - 1 else self.dones[t + 1]
            mask = 1.0 - (1.0 if next_done else 0.0)
            delta = self.rewards[t] + self.gamma * next_val * mask - self.values[t]
            gae = delta + self.gamma * self.lam * mask * gae
            self.advantages[t] = gae
            self.returns[t] = gae + self.values[t]

    def get(self):
        return {
            "obs": self.obs[:self.ptr],
            "actions": self.actions[:self.ptr],
            "rewards": self.rewards[:self.ptr],
            "log_probs": self.log_probs[:self.ptr],
            "advantages": self.advantages[:self.ptr],
            "returns": self.returns[:self.ptr],
        }

    def clear(self):
        self.ptr = 0

    def __len__(self):
        return self.ptr
