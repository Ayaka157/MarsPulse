# MarsPulse — 项目工作记录

## 项目目标

基于 MARL（多智能体强化学习）的火星地下熔岩管基地资源智能调配与生存极限仿真研究。使用 MAPPO 算法训练三个协作智能体（电力管理、热控管理、科学管理），在 300-sol 双风暴火星年场景下实现自主资源分配，最大化基地生存时间。

---

## 1. 模型参数（最终版本）

### 场景配置

| 参数 | 值 | 物理依据 |
|---|---|---|
| 回合时长 | 300 sols (~308 地球日) | 一个火星年 |
| 时间步长 Δt | 0.05 sol (~1.2 地球小时) | 每 sol 20 步 |
| 风暴1 | sol 60–120, τ=2.5, 60 sols | 提前大型风暴 |
| 风暴2 | sol 200–235, τ=2.2, 35 sols | 沙尘季节中型风暴 |
| 电池容量 | 3,000 kWh | ~60 组 Tesla 电池包，约 10-12 sol 自主 |
| 太阳电池阵 | 150 kW 峰值 | ~750 m² 光伏板 |
| RTG | 10.0 kW | ~4 台 MMRTG（毅力号同型） |
| O₂ 储存 | 500 kg | 6 人 99 sol 呼吸 |
| 乘员 | 6 人 | 中型基地 |
| O₂ 消耗 | 0.84 kg/sol/人 | NASA BVAD 标准 |
| 目标舱温 | 20°C | |
| 岩温 | -20°C | 10–20 m 深度 |
| 加热器最大 | 15 kW | ~3 台热泵 |
| 故障概率 | 1.0%/sol/模式 | 4 种模式 |

### 物理常数

| 参数 | 值 |
|---|---|
| 太阳电池效率 η | 0.85 |
| 晴空光学厚度 τ_clear | 0.15 |
| 风暴光学厚度 τ_storm | 2.5 |
| 风暴2光学厚度 τ_storm2 | 2.2 |
| 太阳入射角 θ | 0 rad（赤道/追日阵列） |
| 栖息舱热质量 C_h | 50,000 kJ/°C |
| 导热系数 k_h | 0.18 kW/°C |
| 电解效率 η_elec | 0.15 kg O₂/kWh |
| 废热比例 | 70% |
| 稳定度权重 w_E/w_O/w_T | 0.40/0.35/0.25 |
| S_critical / S_fatal | 0.15 / 0.05 |

### MAPPO 训练超参数

| 参数 | 值 |
|---|---|
| Actor 学习率 | 3e-4（余弦退火） |
| Critic 学习率 | 8e-4（余弦退火） |
| γ | 0.995 |
| λ (GAE) | 0.95 |
| ε (PPO clip) | 0.2 |
| 目标 KL | 0.01 |
| 熵系数 | 0.05 |
| PPO epochs | 8 |
| Mini-batch | 512 |
| Actor 网络 | [128, 128] MLP, Tanh |
| Critic 网络 | [256, 256] MLP, Tanh |
| 总步数 | 300,000 |
| 每轮步数 | 3,000 |

### 智能体动作空间

| 智能体 | 动作 | 范围 |
|---|---|---|
| 电力管理 (Agent 0) | 生命保障功率 | [0, 30] kW |
| 热控管理 (Agent 1) | 加热功率 | [0, 20] kW |
| 科学管理 (Agent 2) | 实验功率 | [0, 25] kW |

---

## 2. 最终实验结果

### 步骤2：MARL vs 静态基线（50 回合）

| 智能体 | 生存时间 (sols) | 死亡率 | 完整存活率 |
|---|---|---|---|
| 静态规则 | 96.4 ± 46.9 | 100% | 0% |
| 静态保守 | 95.8 ± 47.1 | 100% | 0% |
| **MAPPO MARL** | **169.7 ± 118.5** | **62%** | **38%** |
| **提升** | **+76.0%** | | |

训练最佳评估生存: 135.1 sols（epoch 90）
训练时间: 33.6 min，300k 步
训练奖励趋势: r = +0.775（p < 0.001）

### 步骤3：敏感性分析

静态智能体对所有电池容量（1,000-5,000 kWh）和 τ（1.5-3.0）表现出 ~118 sols 的平坦响应——确认为「决策天花板」而非物理天花板。

最危险复合故障：加热器故障 + 栖息舱泄漏 = 74.2 ± 52.3 sols。

### 步骤4：消融实验（50 回合）

| 方法 | 生存 (sols) | 死亡率 |
|---|---|---|
| 静态规则 | 114.1 ± 73.8 | 92% |
| 单智能体 PPO | 2.5 ± 0.2 | 100% |
| 独立 PPO ×3 | 2.4 ± 0.2 | 100% |
| **MAPPO（中心化）** | **129.7 ± 122.2** | 70% |

中心化评论家贡献: +127.3 sols（vs. 独立 PPO），~54× 提升。

---

## 3. 文件结构

```
D:\2026_5_14_MarsPulse\
├── mars_pulse/
│   ├── config.py              # 场景配置（300 sol, 3000 kWh, 双风暴）
│   ├── physics.py             # 物理引擎（预计算常量优化，42 μs/step）
│   ├── environment.py         # 单智能体环境
│   ├── multi_agent_env.py     # 三智能体协作环境
│   ├── agents/
│   │   ├── static_agent.py    # 静态规则基线（含过热保护）
│   │   ├── ppo.py             # 单智能体 PPO（含动作边界）
│   │   └── mappo.py           # MAPPO（中心化评论家+去中心化执行者）
│   └── __init__.py
├── tests/
│   ├── test_step1.py          # 步骤1：基线测试
│   ├── train_step2.py         # 步骤2：MAPPO 训练
│   ├── eval_new_model.py      # 新模型评估
│   ├── step3_analysis.py      # 步骤3：敏感性与故障分析
│   ├── step4_ablation.py      # 步骤4：消融实验
│   ├── step5_final_report.py  # 步骤5：仪表盘生成
│   ├── plot_results.py        # 步骤1 图表生成
│   └── plot_step2.py          # 步骤2 图表生成
├── report/
│   ├── marspulse_EN.tex       # 英文论文（16 页）
│   ├── marspulse_EN.pdf       # 英文论文 PDF
│   ├── marspulse_CN.tex       # 中文论文（17 页）
│   └── marspulse_CN.pdf       # 中文论文 PDF
├── output/
│   ├── mappo_model.pt         # 训练好的 MAPPO 模型 (505 KB)
│   ├── step1_results.json     # 步骤1 聚合结果
│   ├── step1_detail.json      # 步骤1 详细轨迹
│   ├── step2_results.json     # 步骤2 评估结果
│   ├── step2_training_log.json# 步骤2 训练日志
│   ├── new_model_results.json # 新模型评估结果
│   ├── step3_sensitivity.json # 敏感性扫描数据
│   ├── step3_failures.json    # 故障模式数据
│   ├── step3_surface.json     # 生存曲面数据
│   ├── step3_phase.json       # 相变数据
│   ├── step4_ablation.json    # 消融实验数据
│   ├── step5_executive_summary.json # 执行摘要
│   ├── step5_final_table.txt  # 最终对比表
│   ├── step1_trajectories.png # 图1：资源轨迹（双风暴标注）
│   ├── step1_power_balance.png# 电力平衡图
│   ├── step1_comparison.png   # 基线对比图
│   ├── step2_training.png     # 图2：训练曲线
│   ├── step2_comparison.png   # 步骤2 对比图
│   ├── step2_agent_actions.png# 图3：智能体动作
│   ├── step3_sensitivity.png  # 图4：敏感性分析
│   ├── step3_failure_impact.png
│   ├── step3_survival_surface.png # 图5：生存曲面
│   ├── step3_phase_transition.png
│   ├── step4_ablation.png     # 图6：消融实验
│   └── step5_dashboard.png    # 6 面板集成仪表盘
└── reviews/
    ├── review1.pdf
    └── review2.pdf
```

---

## 4. 变动明细

### 4.1 配置演变

| 版本 | 回合 | 电池 | 太阳 | RTG | 风暴 | O₂ |
|---|---|---|---|---|---|---|
| v1（初始） | 60 sol | 600 kWh | 50 kW | 5 kW | 1×30 sol | 200 kg |
| v2（扩张） | 300 sol | 5,000 kWh | 150 kW | 10 kW | 2 storm | 500 kg |
| v3（最终） | 300 sol | 3,000 kWh | 150 kW | 10 kW | 2 storm | 500 kg |

v2→v3 变更原因：5,000 kWh 场景静态智能体100%存活，过于容易。降至3,000 kWh使静态基线降至~96 sols，形成真正的挑战。

### 4.2 关键 Bug 修复

1. **动作边界 Bug（最严重）**：GaussianActor 输出无界，智能体学会输出负功率值（违反能量守恒），产生人为高奖励。修复：tanh 压缩 + clamp + 非法动作惩罚。

2. **训练稳定性 Bug**：回合结束时的 `break` 导致 buffer 过小。修复：移除 break，回合结束后 reset 并继续收集。

3. **JSON 序列化**：numpy float32 不可序列化。修复：所有值包裹 `float()`。

4. **CUDA 设备不匹配**：compute_value() 中 tensor 创建在 CPU。修复：添加 `device=mappo.device`。

5. **PyTorch 2.6 weights_only**：torch.load() 默认变化。修复：`weights_only=False`。

6. **浮点死亡率判断**：`s < cfg.episode_length` 因浮点误差失败。修复：使用 `terminated` 标志。

7. **静态智能体过热**：基地规模扩大后设备功率增加导致过热。修复：静态智能体增加 T>26°C/T>30°C 减载逻辑。

8. **热质量参数**：初始 C_h=5,000 kJ/°C 过小，所有智能体在 0.1 sol 内因温度超界死亡。修复：增至 50,000 kJ/°C。

### 4.3 性能优化

- 物理引擎：预计算常量（`_dt_hours`, `_dt_thermal`, `_P_solar_max`, `_cos_theta`），42 μs/step
- MAPPO：小批量 PPO（mini_batch=512）、余弦退火学习率
- 训练：GPU 加速（CUDA），300k 步/33.6 min

### 4.4 论文修订

**审稿1 修正：**
- 方法论说明：解释跨表基线差异（不同种子集）
- 提升率从 32.3% 更正为 32.6%
- 故障基线从 46.3±17.8 更正为 44.8±18.9
- 中心化评论家说明扩展为 3 个因素
- 添加统计显著性（Cohen's d）
- 添加 8 处内文引用
- Pin 计算分解为 solar + RTG 分量

**审稿2 修正：**
- 图6（消融）数据：从旧值 46.3/58.8 更新为 114.1/129.7 sols
- 图1（轨迹）X 轴：从 40 sols 扩展至 70+ sols，正确显示风暴时间
- 表6 占位符：填入实际消融数据
- 表5 符号：+ → ±

**去 AI 味修改：**
- 移除公式化过渡词（"Notably", "Key finding:", "This validates..."）
- 减少过度强调（加粗仅用于表格重点）
- 变化句式结构，避免模板重复
- 结论用词更审慎（"suggests" 替代 "demonstrates"）
- 摘要从营销风格改写为事实性总结
- 删除冗余内部小结

### 4.5 图表生成

所有 11 张 PNG 图表均使用 300-sol 双风暴配置的最新数据重新生成：
- 步骤1 图表：双风暴标注（橙色/红色阴影）、X 轴延伸至风暴区域
- 步骤2 图表：300k 步训练曲线（含三阶段标注）、双风暴智能体动作轨迹
- 步骤3 图表：更新扫描范围（电池 1000-5000 kWh、RTG 4-15 kW）、基线标记移至 (3000, 2.5)
- 步骤4 图表：消融数据更新为当前 300-sol 结果
- 步骤5 仪表盘：6 面板集成，全部引用新数据

---

## 5. 核心公式

### 生存稳定度指数
$$S(t) = w_E E(t) + w_O O(t) + w_T T(t)$$

### 太阳电力输入（Beer-Lambert 消光）
$$P_{\text{in}}(t) = P_{\max} \cdot \eta \cdot e^{-\tau(t)} \cdot \cos(\theta) + P_{\text{rtg}}$$

晴空 (τ=0.15): 119.8 kW | 风暴峰值 (τ=2.5): 20.5 kW (-83%)

### 热动力学
$$C_h \frac{dT_h}{dt} = Q_{\text{heater}} + Q_{\text{waste}} - k_h(T_h - T_{\text{rock}})$$

### 氧气质量平衡
$$\frac{dM_{O_2}}{dt} = \eta_{\text{elec}} \cdot P_{\text{elec}} - n_{\text{crew}} \cdot \dot{m}_{\text{cons}}$$

### 动作边界强制
$$\mu = \tanh(\mu_{\text{raw}}) \cdot s + b \quad (s = \tfrac{a_{\text{high}} - a_{\text{low}}}{2},\ b = \tfrac{a_{\text{high}} + a_{\text{low}}}{2})$$

### PPO 裁剪目标
$$\mathcal{L}_i(\theta_i) = \mathbb{E}_t\left[\min\left(r_t(\theta_i)\hat{A}_t,\ \text{clip}(r_t(\theta_i), 1-\epsilon, 1+\epsilon)\hat{A}_t\right)\right]$$

---

## 6. 结论要点

1. MAPPO 将平均生存时间从 96.4 延长至 169.7 sols（+76%），完整存活率 0%→38%
2. 中心化评论家是承重组件：移除后策略崩溃至 ~2.5 sols（-111.6 vs 静态）
3. 静态智能体受决策架构限制（决策天花板），5 倍电池增加无任何收益
4. 最危险复合故障：加热器故障 + 栖息舱泄漏（74.2 sols）
5. 双峰结果分布：策略在临界边界运行，早期失败 (20%) vs 完整存活 (38%)
6. 论文已通过两轮审稿修正，中英文 PDF 零错误编译
