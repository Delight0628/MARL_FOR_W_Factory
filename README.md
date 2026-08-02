<div align="center">

```
╔═══════════════════════════════════════════════════════════════════╗
║                                                                   ║
║   🏭  W工厂生产调度 · 多智能体强化学习系统  🤖                   ║
║                                                                   ║
║          ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐      ║
║          │ 带锯 │→│ 五轴 │→│ 砂光 │→│ 组装 │→│ 包装 │      ║
║          └──────┘  └──────┘  └──────┘  └──────┘  └──────┘      ║
║                                                                   ║
╚═══════════════════════════════════════════════════════════════════╝
```

# 🏭 MARL FOR W Factory

**基于多智能体强化学习（MAPPO）的智能工厂生产调度系统**

[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/downloads/)
[![TensorFlow 2.15](https://img.shields.io/badge/TensorFlow-2.15-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![PettingZoo](https://img.shields.io/badge/PettingZoo-1.24+-4CAF50?style=for-the-badge&logo=python&logoColor=white)](https://pettingzoo.farama.org/)
[![SimPy](https://img.shields.io/badge/SimPy-4.0+-E91E63?style=for-the-badge)](https://simpy.readthedocs.io/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](https://opensource.org/licenses/MIT)
[![Code Style](https://img.shields.io/badge/Code%20Style-Black-000000?style=for-the-badge)](https://github.com/psf/black)

---

🎯 **最小化完工时间** · 📈 **最大化设备利用率** · ⏰ **最小化订单延期** · 🛡️ **提升动态鲁棒性**

[快速开始](#-快速开始) · [系统架构](#-系统架构) · [训练流程](#-训练流程) · [可视化应用](#-可视化应用) · [技术文档](#-技术亮点)

</div>

---

## 📋 目录

> 💡 **点击跳转到对应章节**

| 🏠 基础 | 🧠 核心技术 | 🚀 使用指南 | 📚 进阶 |
|:---|:---|:---|:---|
| [项目概述](#-项目概述) | [核心架构](#-系统架构) | [快速开始](#-快速开始) | [高级配置](#-高级配置) |
| [技术栈](#-核心技术栈) | [环境设计](#-环境设计) | [训练模型](#-训练模型) | [模块化示例](#-模块化使用示例) |
| [项目结构](#-项目结构) | [MAPPO算法](#-mappo算法实现) | [监控训练](#-监控训练) | [最佳实践](#-训练最佳实践) |
| [功能亮点](#-功能亮点) | [训练流程](#-训练流程) | [评估模型](#-评估模型) | [常见问题](#-常见问题) |

---

## 🎯 项目概述

<div align="center">

```
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│   🏭 W工厂：多品种 · 小批量 · 动态扰动                                  │
│                                                                         │
│   ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐               │
│   │ 黑胡桃木 │   │ 橡木书柜 │   │ 松木床架 │   │ 樱桃木椅 │               │
│   │  餐 桌   │   │         │   │         │   │         │               │
│   │  5道工序 │   │  3道工序 │   │  4道工序 │   │  3道工序 │               │
│   └─────────┘   └─────────┘   └─────────┘   └─────────┘               │
│        │              │              │              │                   │
│        ▼              ▼              ▼              ▼                   │
│   ┌─────────────────────────────────────────────────────┐              │
│   │  共享资源：带锯机 | 五轴加工中心 | 砂光机 | 组装台 | 包装台  │      │
│   └─────────────────────────────────────────────────────┘              │
│                                                                         │
│   🎯 优化目标：完工时间↓  设备利用率↑  订单延期↓  动态鲁棒性↑          │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

</div>

本项目旨在解决**多品种、小批量**生产模式下的智能工厂调度问题。W工厂作为典型的现代家具制造企业，面临以下核心挑战：

| 挑战 | 描述 | 解决方案 |
|:---:|:---|:---|
| 🏭 | **5种工作站**：带锯机、五轴加工中心、砂光机、组装台、包装台 | 多智能体协作决策 |
| 📦 | **4种产品**：黑胡桃木餐桌、橡木书柜、松木床架、樱桃木椅子 | 差异化观测空间设计 |
| 🔄 | **复杂工艺路线**：每种产品3-5道工序，共享设备资源 | 全局信息感知 |
| ⏰ | **严格交期要求**：多订单并行，需权衡完工时间与延期惩罚 | 稠密奖励系统 |
| ⚡ | **动态扰动**：设备故障、紧急插单等突发事件 | 两阶段鲁棒性训练 |

---

## 🌟 功能亮点

<div align="center">

```
┌──────────────────┬──────────────────┬──────────────────┬──────────────────┐
│   🧠 MAPPO算法   │  🏗️ 模块化架构   │  📊 课程学习     │  🔧 自适应熵调整  │
│                  │                  │                  │                  │
│  CTDE范式        │  5个独立模块     │  渐进式难度      │  动态探索-利用    │
│  多智能体协同    │  清晰职责分离    │  防止灾难性遗忘  │  平衡策略         │
│  无放回采样      │  易于扩展测试    │  稳定收敛        │  避免过早收敛     │
└──────────────────┴──────────────────┴──────────────────┴──────────────────┘
┌──────────────────┬──────────────────┬──────────────────┬──────────────────┐
│  ⚡ 并行采集     │  🛡️ 动态鲁棒性  │  🎨 Streamlit    │  📈 甘特图可视化  │
│                  │                  │                  │                  │
│  多进程Worker    │  设备故障恢复    │  交互式调度界面  │  设备维度/订单维度 │
│  加速数据采集    │  紧急插单应对    │  一键仿真执行    │  KPI雷达图对比   │
│  GPU+CPU协同     │  泛化能力增强    │  实时进度监控    │  数据导出支持     │
└──────────────────┴──────────────────┴──────────────────┴──────────────────┘
```

</div>

---

## 🔬 核心技术栈

| 组件 | 技术 | 版本 | 用途 |
|:---:|:---|:---:|:---|
| 🧠 | **TensorFlow** | 2.15.0 | 神经网络构建与训练 |
| 🤖 | **MAPPO** | Custom | 多智能体策略优化（CTDE范式） |
| 🌍 | **PettingZoo** | 1.24+ | 多智能体环境标准接口 |
| ⚙️ | **SimPy** | 4.0+ | 离散事件仿真（工厂物理过程模拟） |
| ⚡ | **ProcessPoolExecutor** | Python 3.8+ | 多进程并行数据采集 |
| 🎨 | **Streamlit** | 1.32+ | 交互式调度演示应用 |
| 📊 | **TensorBoard** | 2.10+ | 训练过程实时监控 |

---

## 🧩 系统架构

<div align="center">

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          🎨 训练管理层                                  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  │
│  │ auto_train  │  │ TensorBoard │  │ Checkpoint  │  │   Logging   │  │
│  │  自动训练   │  │  实时监控   │  │  模型管理   │  │   日志记录   │  │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘  │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                     🧠 MAPPO算法层 (mappo/)                            │
│                                                                         │
│  ┌───────────────────────────────────────────────────────────────┐     │
│  │  ppo_marl_train.py ──→ ppo_trainer.py (SimplePPOTrainer)     │     │
│  │   (训练入口)           (两阶段流程 · 课程学习 · 自适应熵)      │     │
│  └───────────────────────────────────────────────────────────────┘     │
│        │              │              │              │                   │
│        ▼              ▼              ▼              ▼                   │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐              │
│  │ ppo_net  │  │ ppo_buf  │  │ ppo_work │  │ sampling │              │
│  │  work.py │  │  fer.py  │  │  er.py   │  │  _utils  │              │
│  │          │  │          │  │          │  │          │              │
│  │Actor-    │  │GAE优势   │  │并行经验  │  │无放回    │              │
│  │Critic网络│  │函数计算  │  │采集      │  │采样工具  │              │
│  │ (457行)  │  │ (128行)  │  │ (304行)  │  │          │              │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘              │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │
                    state, reward ↕ action
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│              🌍 PettingZoo接口层 (w_factory_env.py)                    │
│                                                                         │
│  WFactoryEnv (ParallelEnv)                                             │
│  ├─ 📐 观测空间构建 (132维向量)                                        │
│  ├─ 🎯 动作空间映射 (MultiDiscrete [11] × 10)                         │
│  ├─ 💰 奖励计算 (三层稠密奖励系统)                                     │
│  └─ 📊 KPI统计收集                                                     │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │
                    control signals ↕ sim state
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                 ⚙️ SimPy仿真层 (WFactorySim)                          │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  离散事件仿真核心                                                │   │
│  │  ├─ 🔧 资源管理 (simpy.Resource) — 5种工作站                    │   │
│  │  ├─ 📋 队列系统 (simpy.Store) — 工件等待队列                    │   │
│  │  ├─ 🔄 零件流转 (_part_process) — 工序自动流转                  │   │
│  │  ├─ ⚠️ 设备故障 (_equipment_failure_process) — 随机故障         │   │
│  │  └─ 🚨 紧急插单 (_emergency_order_process) — 动态扰动          │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                 ⚙️ 配置层 (w_factory_config.py)                        │
│  工作站配置 │ 产品路线 │ 订单数据 │ 奖励系统 │ 训练参数                │
└─────────────────────────────────────────────────────────────────────────┘
```

</div>

### MAPPO网络架构

```
┌─────────────────────────────────────────────────────────────────────┐
│              🎯 Actor Network (分布式执行)                          │
│                                                                     │
│  输入: 局部观测 (132维)                                             │
│    ↓                                                               │
│  Dense(1024) → ReLU → Dropout(0.1)                                │
│    ↓                                                               │
│  Dense(512)  → ReLU → Dropout(0.1)                                │
│    ↓                                                               │
│  Dense(256)  → ReLU → Dropout(0.1)                                │
│    ↓                                                               │
│  MultiHead Output (10 heads × 11 actions/head)                     │
│  ├─ Head 1:  Softmax(11) → 候选工件1的概率分布                     │
│  ├─ Head 2:  Softmax(11) → 候选工件2的概率分布                     │
│  └─ ...                                                            │
│  输出: 动作概率分布 (支持无放回采样，避免多头选择相同工件)           │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│              🎯 Critic Network (集中式训练)                         │
│                                                                     │
│  输入: 全局状态 (global_state + agent_one_hot)                     │
│    ↓                                                               │
│  Dense(1024) → ReLU → Dropout(0.1)                                │
│    ↓                                                               │
│  Dense(512)  → ReLU → Dropout(0.1)                                │
│    ↓                                                               │
│  Dense(256)  → ReLU → Dropout(0.1)                                │
│    ↓                                                               │
│  Dense(1) → 状态价值估计 V(s)                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 🌍 环境设计

### 观测空间 (132维)

环境为每个智能体提供丰富的观测信息，结构如下：

```
┌─────────────────────────────────────────────────────────────────┐
│  [1] 🤖 Agent自身特征 (8维)                                     │
│      ├─ 身份one-hot (5维): [1,0,0,0,0] = 带锯机               │
│      ├─ 设备容量 (1维): 归一化容量值                            │
│      ├─ 设备繁忙率 (1维): 当前繁忙程度                          │
│      └─ 故障状态 (1维): 0=正常, 1=故障                         │
├─────────────────────────────────────────────────────────────────┤
│  [2] 🌍 全局宏观特征 (4维)                                      │
│      ├─ 时间进度 (1维): current_time / SIMULATION_TIME         │
│      ├─ WIP率 (1维): 在制品数量 / 总订单数                      │
│      ├─ 瓶颈拥堵度 (1维): 最长队列长度归一化                    │
│      └─ 平均队列长度 (1维): 所有队列平均长度                    │
├─────────────────────────────────────────────────────────────────┤
│  [3] 📋 当前队列摘要 (30维)                                     │
│      6种特征 × 5种统计量 (min, max, mean, std, median)          │
│      特征类型:                                                  │
│      ├─ 剩余工序数 │ 剩余加工时间 │ 当前工序时间               │
│      └─ 下游拥堵度 │ 优先级      │ 是否最终工序                 │
├─────────────────────────────────────────────────────────────────┤
│  [4] 🔧 候选工件详细特征 (90维)                                 │
│      10个候选工件 × 9维特征/工件                                │
│      工件特征:                                                  │
│      ├─ exists (1维): 候选工件是否存在                          │
│      ├─ 剩余工序数 (1维): 归一化                               │
│      ├─ 剩余加工时间 (1维): 归一化                             │
│      ├─ 当前工序时间 (1维): 归一化                             │
│      ├─ 下游拥堵度 (1维): 下一工位队列长度                      │
│      ├─ 优先级 (1维): 1/2/3 → 归一化                           │
│      ├─ 是否最终工序 (1维): 0/1标志位                           │
│      ├─ 产品类型one-hot (1维): 4种产品编码                      │
│      └─ ⏰ 时间压力感知 (1维): (due_date - current_time) / ... │
└─────────────────────────────────────────────────────────────────┘
总维度: 8 + 4 + 30 + 90 = 132维
```

**💡 关键设计理念**：

| 特性 | 说明 |
|:---|:---|
| 🎲 **多样性候选采样** | 混合EDD(紧急优先)、SPT(短作业优先)、随机采样，提供探索空间 |
| ⏰ **时间压力感知** | 基于物理时间关系计算，非启发式作弊 |
| 📐 **压缩归一化** | `y = x/norm / (1 + x/norm)` 避免特征饱和 |

### 动作空间 (MultiDiscrete)

每个智能体的动作空间为 `MultiDiscrete([11, 11, ..., 11])` (10个头)：

```
动作空间设计:
├─ 0: IDLE (空闲等待)
├─ 1-10: 选择候选工件1-10进行加工
└─ 采用无放回采样机制，避免多头选择相同工件
```

**🔄 无放回采样机制**：

```python
# 伪代码示例
for head_i in range(num_heads):
    masked_probs = probs_i * (1 - selected_mask)
    action_i = sample_from(masked_probs)
    selected_mask[action_i] = 1  # 标记已选
```

### 奖励系统 (稠密、目标导向)

奖励函数采用三层设计：

```
┌─────────────────────────────────────────────────────────────────┐
│  🥇 第一层：任务完成奖励 (主导信号)                              │
│  ├─ 零件完成奖励: +80                                          │
│  └─ 全部完成奖励: +500 (所有零件完成时)                         │
├─────────────────────────────────────────────────────────────────┤
│  🥈 第二层：时间质量奖励 (次要信号)                              │
│  ├─ 按时完成奖励: +80 (零件在交期内完成)                        │
│  └─ 延期惩罚: -10 × tardiness_minutes                          │
├─────────────────────────────────────────────────────────────────┤
│  🥉 第三层：过程塑形奖励 (引导信号)                              │
│  ├─ 进度塑形: +0.1 × progress_made                             │
│  ├─ 紧急度降低: +0.1 × urgency_reduction                       │
│  ├─ 不必要空闲惩罚: -1.0                                       │
│  ├─ 无效动作惩罚: -0.5                                         │
│  └─ 负松弛时间惩罚: -0.1 × max(0, -slack_time)                 │
└─────────────────────────────────────────────────────────────────┘
```

**💡 设计理念**：密集即时反馈 + 长期目标导向

---

## 🧠 MAPPO算法实现

### 算法框架

MAPPO (Multi-Agent Proximal Policy Optimization) 采用 **CTDE (Centralized Training with Decentralized Execution)** 范式：

- **训练阶段**：Critic使用全局状态 + 智能体ID条件化
- **执行阶段**：Actor仅使用局部观测，支持分布式部署

### 核心组件

#### 1. PPO损失函数

```python
# Actor损失 (Clipped Surrogate Objective)
ratio = exp(log_π_new(a|s) - log_π_old(a|s))
clipped_ratio = clip(ratio, 1-ε, 1+ε)
L_actor = -min(ratio × A, clipped_ratio × A) - β × H(π)

# Critic损失 (Value Function MSE)
L_critic = (V(s) - V_target)²

# 其中:
# A: GAE优势函数
# β: 熵系数 (自适应调整)
# ε: 裁剪比率 (0.2)
```

#### 2. GAE优势函数估计

```python
# Generalized Advantage Estimation
δ_t = r_t + γ × V(s_{t+1}) - V(s_t)
A_t = δ_t + γλ × A_{t+1}

# 超参数:
# γ (gamma) = 0.99 : 折扣因子
# λ (lambda_gae) = 0.95 : GAE平滑参数
```

#### 3. 自适应熵调整

```python
# 防止策略过早收敛的自适应机制
if performance_stagnant for N episodes:
    entropy_coeff *= (1 + boost_factor)  # 提升探索
elif completion_rate > 95%:
    entropy_coeff *= 0.995  # 降低探索，精细化策略
```

### 训练配置参数

| 参数 | 值 | 说明 |
|:---:|:---:|:---|
| **网络结构** | [1024, 512, 256] | 3层全连接 |
| **学习率调度** | 8e-5 → 1e-6 | 多项式衰减 |
| **PPO Epochs** | 12 | 每批数据更新12次 |
| **Mini-batches** | 4 | 批次内分4个小批 |
| **Clip Ratio** | 0.2 | PPO裁剪参数 |
| **初始熵系数** | 0.5 | 探索强度 |
| **梯度裁剪** | 1.0 | 防止梯度爆炸 |
| **优势裁剪** | 5.0 | 稳定训练 |
| **并行Workers** | 4 | 数据采集进程数 |

---

## 📈 训练流程

### 两阶段渐进式训练策略

```
┌────────────────────────────────────────────────────────────────────────┐
│  📗 阶段一：基础泛化训练 (Foundation Phase)                            │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │  🎯 目标：掌握随机订单环境下的泛化调度能力                       │  │
│  │                                                                  │  │
│  │  ┌────────────────────────────────────────────────────────────┐  │  │
│  │  │  📚 可选：课程学习 (Curriculum Learning)                   │  │  │
│  │  │  ├─ 阶段1: 40%订单 → 目标分数0.80, 零延期                  │  │  │
│  │  │  ├─ 阶段2: 80%订单 → 目标分数0.80, 延期<225min            │  │  │
│  │  │  └─ 阶段3: 100%订单 → 目标分数0.72, 延期<450min           │  │  │
│  │  └────────────────────────────────────────────────────────────┘  │  │
│  │                                                                  │  │
│  │  📋 训练策略：                                                   │  │
│  │  ├─ 75% workers: 随机订单 (5-8订单, 每单3-12件)                │  │
│  │  └─ 25% workers: BASE_ORDERS (稳定锚点，防遗忘)                │  │
│  │                                                                  │  │
│  │  ✅ 毕业标准：                                                   │  │
│  │  ├─ 综合评分 > 0.70                                            │  │
│  │  ├─ 完成率 > 95%                                               │  │
│  │  ├─ 延期 < 450min                                              │  │
│  │  └─ 连续8次达标                                                │  │
│  └──────────────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌────────────────────────────────────────────────────────────────────────┐
│  📘 阶段二：动态事件鲁棒性训练 (Generalization Phase)                  │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │  🎯 目标：在动态扰动下保持调度性能                               │  │
│  │                                                                  │  │
│  │  📋 训练策略：                                                   │  │
│  │  ├─ 75% workers: 随机订单 + 动态事件                            │  │
│  │  │   ├─ ⚠️ 设备故障 (MTBF=24h, MTTR=30min)                    │  │
│  │  │   └─ 🚨 紧急插单 (到达率0.1/h)                              │  │
│  │  └─ 25% workers: BASE_ORDERS (保持基准性能)                    │  │
│  │                                                                  │  │
│  │  ✅ 完成标准：                                                   │  │
│  │  ├─ 综合评分 > 0.60                                            │  │
│  │  ├─ 完成率 > 80%                                               │  │
│  │  └─ 连续10次达标                                               │  │
│  └──────────────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────────────┘
```

### 训练监控与日志

训练过程中实时监控以下指标：

```
🔂 训练回合 145/1000 | 平均奖励: 1245.3
   (每个worker奖励: [1234, 1256, 1240, 1251], 完成全部: 4/4)
   | Actor损失: 0.0234 | ⏱️本轮用時: 45.2s (CPU采集: 38.1s, GPU更新: 7.1s)
   | 本回合任务: [随机订单+动态事件]×4workers(均奖:1245.3)

📊 此回合KPI评估 - 总完工时间: 478.5min  
   | 设备利用率: 87.3% | 订单延期时间: 125.4min 
   | 完成零件数: 41/42 | 阶段: '动态事件鲁棒性训练'
   | 评估环境:[随机订单+故障✓+插单✓]

🚥 回合评分: 0.683 (全局最佳: 0.721)(泛化阶段最佳: 0.695)
   ✅ 泛化强化阶段最佳! 模型保存至: models/1028_1342/1028_1342_general_train_best_actor.h5

🔮 当前训练进度: 14.5% | 当前时间：13:45:23 | 预计完成时间: 18:32:15
```

---

## 🚀 快速开始

### 环境要求

```bash
# Python版本
Python 3.8+
# 核心依赖
tensorflow>=2.15.0
numpy>=1.24.0
gymnasium>=0.29.0
pettingzoo>=1.24.0
simpy>=4.0.0
streamlit>=1.30.0  # 可视化应用
matplotlib>=3.7.0
```

### 安装步骤

```bash
# 1. 克隆仓库
git clone https://github.com/Delight0628/MARL_FOR_W_Factory.git
cd MARL_FOR_W_Factory

# 2. 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. 安装依赖
pip install -r requirements.txt

# 4. 验证安装
python -c "import tensorflow as tf; print(f'TensorFlow {tf.__version__} installed')"
```

### 训练模型

#### 方式1: 使用自动化训练脚本（推荐）✅

```bash
# 启动自动化训练管理器
python auto_train.py

# 功能：
# - 自动创建带时间戳的模型目录
# - 实时监控训练进程
# - 自动保存最佳检查点
# - 后台运行TensorBoard
```

#### 方式2: 手动训练

```bash
# 基础训练（使用默认配置）
python mappo/ppo_marl_train.py

# 指定保存路径
python mappo/ppo_marl_train.py \
    --models-dir ./my_models \
    --logs-dir ./my_logs
```

#### 方式3: 自定义配置训练

编辑 `environments/w_factory_config.py` 修改训练参数：

```python
# 示例：调整训练流程
TRAINING_FLOW_CONFIG = {
    "foundation_phase": {
        "graduation_criteria": {
            "target_score": 0.75,  # 提高毕业标准
            "target_consistency": 10,
        },
        # ... 更多配置
    }
}

# 示例：调整网络结构
PPO_NETWORK_CONFIG = {
    "hidden_sizes": [2048, 1024, 512],  # 更大的网络
    "entropy_coeff": 0.6,  # 更强的探索
}
```

### 监控训练

```bash
# 启动TensorBoard
tensorboard --logdir=mappo/tensorboard_logs --port=6006

# 浏览器访问
http://localhost:6006
```

TensorBoard显示指标：
- 📉 训练损失曲线 (Actor Loss, Critic Loss)
- 🎲 策略熵变化
- 📊 KPI指标 (Makespan, Utilization, Tardiness)
- 📈 综合评分趋势

---

## 📊 性能评估

### 评估模型

```bash
# 评估单个模型
python evaluation.py --model-path models/best_model/actor.h5

# 完整评估（包含启发式基线对比）
python evaluation.py \
    --model-path models/best_model/actor.h5 \
    --comprehensive \
    --generate-gantt \
    --output-dir results/

# 生成甘特图可视化
python evaluation.py \
    --model-path models/best_model/actor.h5 \
    --generate-gantt
```

### 评估指标

评估脚本会输出以下KPI：

| 指标 | 说明 | 目标 |
|:---:|:---:|:---:|
| 🏁 **Makespan** | 总完工时间（分钟） | ↓ 越小越好 |
| 📈 **Mean Utilization** | 平均设备利用率 | ↑ 越高越好 |
| ⏰ **Total Tardiness** | 总延期时间（分钟） | ↓ 越小越好 |
| ✅ **Completed Parts** | 完成零件数 | ↑ 达到100% |
| 🏆 **Comprehensive Score** | 综合评分 (0-1) | ↑ 越高越好 |

综合评分计算公式：

```python
score = (
    completion_rate * 0.40 +      # 完成率权重40%
    tardiness_score * 0.35 +       # 延期质量35%
    makespan_score * 0.15 +        # 效率15%
    utilization_score * 0.10       # 利用率10%
)
```

### 对比基线算法

系统内置多种启发式算法作为基线：

| 算法 | 说明 | 策略 |
|:---:|:---|:---|
| 📥 **FIFO** | First In First Out | 先到先服务 |
| 📅 **EDD** | Earliest Due Date | 最早交期优先 |
| ⚡ **SPT** | Shortest Processing Time | 最短加工时间优先 |
| 🔴 **CR** | Critical Ratio | 紧急度比率优先 |

```bash
# 评估EDD启发式
python evaluation.py --heuristic EDD

# 对比所有方法
python evaluation.py --comprehensive --model-path models/best_model/actor.h5
```

### 调试工具

```bash
# 详细行为分析
python debug_marl_behavior.py \
    --model-path models/best_model/actor.h5 \
    --max-steps 600 \
    --snapshot-interval 100

# 输出：
# - 每个智能体的决策过程
# - 候选工件概率分布
# - 观测向量解码
# - 关键决策点分析
```

---

## 🎨 可视化应用

### 启动交互式调度系统

```bash
streamlit run app/app_scheduler.py
```

### 应用功能

| 功能 | 描述 |
|:---|:---|
| 🔍 **模型选择** | 自动扫描已训练模型，支持多检查点切换 |
| 📋 **订单配置** | 内置BASE_ORDERS，支持自定义订单，实时验证 |
| 🚀 **调度执行** | 一键启动仿真，实时进度显示，KPI实时更新 |
| 📊 **结果可视化** | 甘特图展示（设备/订单维度），利用率柱状图，KPI雷达图 |
| 💾 **数据导出** | 下载调度历史（CSV），保存甘特图（PNG） |

---

## 🔧 模块化使用示例

### 独立使用各模块

模块化设计允许您独立使用各个组件，或集成到自己的项目中：

#### 示例1: 独立使用PPONetwork

```python
from mappo.ppo_network import PPONetwork
import gymnasium as gym
import tensorflow as tf

# 创建网络实例
state_dim = 132
action_space = gym.spaces.MultiDiscrete([11] * 10)
global_state_dim = 50
lr = 1e-4

network = PPONetwork(
    state_dim=state_dim,
    action_space=action_space,
    lr=lr,
    global_state_dim=global_state_dim
)

# 使用网络进行推理
state = tf.random.normal([1, state_dim])
global_state = tf.random.normal([1, global_state_dim])
action, value, log_prob = network.get_action_and_value(state, global_state)
```

#### 示例2: 独立使用ExperienceBuffer

```python
from mappo.ppo_buffer import ExperienceBuffer

# 创建缓冲区
buffer = ExperienceBuffer()

# 存储经验
for step in range(100):
    buffer.store(
        state=observation,
        global_state=global_obs,
        action=action,
        reward=reward,
        value=value,
        action_prob=log_prob,
        done=done,
        truncated=truncated
    )

# 计算GAE并获取训练批次
states, global_states, actions, old_probs, advantages, returns = buffer.get_batch(
    gamma=0.99,
    lam=0.95,
    next_value_if_truncated=last_value
)
```

#### 示例3: 自定义并行Worker

```python
from mappo.ppo_worker import run_simulation_worker
from concurrent.futures import ProcessPoolExecutor

# 准备网络权重
network_weights = {
    'actor': network.actor.get_weights(),
    'critic': network.critic.get_weights()
}

# 并行运行多个worker
with ProcessPoolExecutor(max_workers=4) as pool:
    futures = []
    for i in range(4):
        future = pool.submit(
            run_simulation_worker,
            network_weights=network_weights,
            state_dim=state_dim,
            action_space=action_space,
            num_steps=1000,
            seed=42 + i,
            global_state_dim=global_state_dim,
            network_config=network_config,
            curriculum_config={'worker_id': i}
        )
        futures.append(future)
    
    # 收集结果
    results = [f.result() for f in futures]
```

---

## ✨ 技术亮点

### 1. 集中式训练分布式执行 (CTDE)

**设计理念**：

- **训练时**：Critic利用全局信息（包含所有智能体状态）评估价值
- **执行时**：Actor仅依赖局部观测，支持分布式部署

```python
# Critic输入：全局状态 + 智能体ID
global_state = concatenate([
    raw_global_state,  # 全局宏观信息
    agent_one_hot      # 当前智能体标识
])

# Actor输入：仅局部观测
local_obs = get_state_for_agent(agent_id)
```

### 2. MultiDiscrete动作空间 + 无放回采样

**问题**：多个智能体可能同时选择处理同一个工件

**解决方案**：

```python
# 逐头采样，已选动作被屏蔽
for head_i in range(num_heads):
    masked_probs = original_probs * (1 - mask)
    action_i = sample(masked_probs)
    mask[action_i] = 1  # 标记已选
```

### 3. 多任务混合训练

**防止灾难性遗忘**：

```python
# 每个训练回合
if episode % 4 == 0:
    # 25%的回合使用BASE_ORDERS（稳定锚点）
    orders = BASE_ORDERS
else:
    # 75%的回合使用随机订单（泛化训练）
    orders = generate_random_orders()
```

### 4. 自适应熵调整

**动态平衡探索与利用**：

```python
# 停滞检测
if no_improvement_for(patience_episodes):
    entropy_coeff *= (1 + boost_factor)  # 增强探索
    
# 过度探索检测
elif completion_rate > 95%:
    entropy_coeff *= decay_rate  # 收敛策略
```

### 5. 压缩归一化技术

**避免特征饱和**：

```python
# 传统归一化：x/norm → 可能>>1导致饱和
# 压缩归一化：y = (x/norm) / (1 + x/norm) → 始终在(0,1)
def compressed_normalize(x, norm):
    normalized = x / norm
    return normalized / (1 + normalized)
```

### 6. 并行环境采集

**提升数据效率**：

```python
# 使用进程池并行运行4个环境
with ProcessPoolExecutor(max_workers=4) as pool:
    futures = [
        pool.submit(run_worker, network_weights, config)
        for _ in range(4)
    ]
    experiences = [f.result() for f in futures]
```

---

## 📁 项目结构

```
MARL_FOR_W_Factory/
├── 🌍 environments/                    # 环境模块
│   ├── w_factory_env.py            # PettingZoo环境 + SimPy仿真
│   └── w_factory_config.py         # 统一配置文件（单一真理源）
│
├── 🧠 mappo/                           # MAPPO算法（模块化架构）
│   ├── ppo_marl_train.py           # 训练入口主脚本 (154行)
│   ├── ppo_trainer.py              # 训练器主类 (1818行)
│   ├── ppo_network.py              # Actor-Critic网络 (457行)
│   ├── ppo_buffer.py               # 经验缓冲与GAE计算 (128行)
│   ├── ppo_worker.py               # 并行Worker进程 (304行)
│   ├── sampling_utils.py           # 无放回采样工具
│   ├── ppo_models/                 # 模型保存目录
│   └── tensorboard_logs/           # TensorBoard日志
│
├── 🎨 app/                             # 可视化应用
│   ├── app_scheduler.py            # Streamlit交互界面
│   ├── i18n.py                     # 国际化支持
│   ├── app_state.json              # 应用状态配置
│   └── app_requirements.txt        # 应用依赖
│
├── 🚀 auto_train.py                    # 自动化训练管理器
├── 📊 evaluation.py                    # 模型评估脚本
├── 🔍 debug_marl_behavior.py          # 调试工具
├── 📈 plotting.py                      # 绘图工具
├── 📋 log_parser.py                    # 日志解析器
│
├── 📦 requirements.txt                 # Python依赖
├── 📖 README.md                        # 项目说明（本文件）
└── 📝 outline.md                       # 项目大纲
```

### 核心文件说明

#### 环境与仿真
| 文件 | 代码量 | 核心功能 |
|:---:|:---:|:---|
| `w_factory_env.py` | ~1720行 | SimPy仿真 + PettingZoo接口 |
| `w_factory_config.py` | ~550行 | 全局配置（工作站/订单/奖励/训练参数） |

#### MAPPO算法模块（模块化架构）
| 文件 | 代码量 | 核心功能 |
|:---:|:---:|:---|
| `ppo_marl_train.py` | 154行 | **训练入口** - 参数解析、流程启动 |
| `ppo_trainer.py` | 1818行 | **训练器主类** - 两阶段训练流程、课程学习、自适应熵、模型保存 |
| `ppo_network.py` | 457行 | **神经网络** - Actor-Critic架构、CTDE范式、MultiDiscrete支持 |
| `ppo_buffer.py` | 128行 | **经验缓冲** - 数据存储、GAE优势函数计算 |
| `ppo_worker.py` | 304行 | **并行Worker** - 多进程环境采集、设备管理 |

#### 评估与应用
| 文件 | 代码量 | 核心功能 |
|:---:|:---:|:---|
| `evaluation.py` | ~790行 | 模型评估 + 启发式对比 + 甘特图 |
| `debug_marl_behavior.py` | ~446行 | 详细行为分析 + 决策过程可视化 |
| `app_scheduler.py` | ~1330行 | Streamlit可视化应用 |
| `auto_train.py` | ~303行 | 自动化训练管理器 |

---

## 🔧 高级配置

### 课程学习配置

在 `w_factory_config.py` 中启用课程学习：

```python
TRAINING_FLOW_CONFIG = {
    "foundation_phase": {
        "curriculum_learning": {
            "enabled": True,  # 开启课程学习
            "stages": [
                {
                    "name": "入门阶段",
                    "orders_scale": 0.4,  # 40%订单量
                    "time_scale": 1.0,
                    "graduation_criteria": {
                        "target_score": 0.80,
                        "min_completion_rate": 100.0,
                        "target_consistency": 10
                    }
                },
                # 更多阶段...
            ]
        }
    }
}
```

### 动态事件配置

```python
# 设备故障
EQUIPMENT_FAILURE = {
    "enabled": True,           # 启用故障
    "mtbf_hours": 24,          # 平均24小时故障一次
    "mttr_minutes": 30,        # 平均30分钟修复
}

# 紧急插单
EMERGENCY_ORDERS = {
    "enabled": True,           # 启用紧急订单
    "arrival_rate": 0.1,       # 每小时0.1个
    "priority_boost": 0,       # 优先级提升
}
```

### 奖励权重调整

```python
REWARD_CONFIG = {
    "part_completion_reward": 100.0,  # 提高完成奖励
    "tardiness_penalty_scaler": -15.0,  # 加重延期惩罚
    "unnecessary_idle_penalty": -2.0,  # 加重空闲惩罚
    # ...
}
```

---

## 📝 训练最佳实践

### 1. 阶段化训练策略

```
推荐流程：
1. 课程学习（可选） → 40% → 80% → 100% 订单量
2. 基础泛化训练 → 随机订单 + 25% BASE_ORDERS锚点
3. 动态鲁棒性训练 → 设备故障 + 紧急插单
```

### 2. 超参数调优建议

| 场景 | 建议调整 |
|:---|:---|
| 📉 **训练不稳定** | ↓ 学习率 (5e-5), ↑ 梯度裁剪 (0.5) |
| 🐌 **收敛过慢** | ↑ 初始熵系数 (0.6), ↑ PPO epochs (15) |
| 🎯 **过拟合BASE_ORDERS** | ↑ 随机订单比例 (90%), ↓ BASE锚点 (10%) |
| ⚡ **动态事件性能差** | 延长阶段二训练, ↑ 故障频率 |

### 3. 模型检查点策略

```python
# 自动保存以下检查点：
1. 各课程阶段最佳模型
2. 基础训练阶段最佳模型
3. 泛化阶段最佳模型
4. 双达标模型（完成率100% + 最高分）
```

---

## 🐛 常见问题

<details>
<summary><b>Q1: 训练过程中出现NaN损失？</b></summary>

**原因**：梯度爆炸或奖励尺度过大

**解决**：
```python
# 1. 降低学习率
LEARNING_RATE_CONFIG["initial_lr"] = 5e-5

# 2. 增强梯度裁剪
PPO_NETWORK_CONFIG["grad_clip_norm"] = 0.5

# 3. 检查奖励尺度
print(f"Max reward: {max(episode_rewards)}")
```
</details>

<details>
<summary><b>Q2: 多进程采集报错 "CUDA initialization error"？</b></summary>

**原因**：子进程GPU资源冲突

**解决**：
```bash
# 方案1: 强制子进程使用CPU
export FORCE_WORKER_CPU=1
python mappo/ppo_marl_train.py

# 方案2: 减少worker数量
# 在w_factory_config.py中
SYSTEM_CONFIG["num_parallel_workers"] = 2
```
</details>

<details>
<summary><b>Q3: 模型加载失败 "Incompatible model format"？</b></summary>

**原因**：TensorFlow版本不兼容

**解决**：
```bash
# 检查TensorFlow版本
python -c "import tensorflow as tf; print(tf.__version__)"

# 如果版本不匹配，重新安装
pip install tensorflow==2.15.0

# 如果仍然失败，尝试加载权重而非完整模型
# 在evaluation.py中使用weights加载模式
```
</details>

<details>
<summary><b>Q4: 训练速度很慢（< 1 iter/min）？</b></summary>

**优化建议**：

```python
# 1. 减少PPO epochs
PPO_NETWORK_CONFIG["ppo_epochs"] = 8

# 2. 减少每回合步数
TRAINING_FLOW_CONFIG["general_params"]["steps_per_episode"] = 1000

# 3. 增加并行workers（如果CPU/内存充足）
SYSTEM_CONFIG["num_parallel_workers"] = 6

# 4. 使用GPU加速
# 确保CUDA可用: nvidia-smi
```
</details>

<details>
<summary><b>Q5: 如何调试特定模块的问题？</b></summary>

**模块化调试策略**：

```python
# 调试PPONetwork
from mappo.ppo_network import PPONetwork
network = PPONetwork(...)
# 添加断点或打印中间层输出

# 调试ExperienceBuffer
from mappo.ppo_buffer import ExperienceBuffer
buffer = ExperienceBuffer()
# 检查GAE计算逻辑

# 调试Worker
# 设置环境变量启用详细日志
export TF_CPP_MIN_LOG_LEVEL=0
# 单独运行worker测试
```
</details>

---

## 📚 参考文献

1. **PPO算法**  
   Schulman, J., et al. (2017). "Proximal Policy Optimization Algorithms." arXiv:1707.06347

2. **MAPPO**  
   Yu, C., et al. (2021). "The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games." arXiv:2103.01955

3. **GAE**  
   Schulman, J., et al. (2015). "High-Dimensional Continuous Control Using Generalized Advantage Estimation." arXiv:1506.02438

4. **CTDE范式**  
   Lowe, R., et al. (2017). "Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments." NIPS 2017

---

## 🤝 贡献指南

欢迎提交Issue和Pull Request！

**开发流程**：

1. 🍴 Fork本仓库
2. 🌿 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 💾 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 🚀 推送到分支 (`git push origin feature/AmazingFeature`)
5. 🔀 开启Pull Request

**代码规范**：

- ✅ 遵循PEP 8
- ✅ 添加类型注解
- ✅ 编写docstring
- ✅ 通过单元测试

**模块化开发建议**：

- 🎯 新功能优先考虑添加到现有模块
- 📦 如需新模块，确保职责单一
- 📖 更新相应的README文档
- 💡 添加使用示例

---

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

---

## 👥 作者与致谢

**项目作者**: Delight0628

**特别致谢**:
- 🧠 TensorFlow团队提供深度学习框架
- 🌍 PettingZoo团队提供多智能体环境接口
- ⚙️ SimPy团队提供离散事件仿真引擎

---

## 📞 联系方式

- **Issues**: [GitHub Issues](https://github.com/Delight0628/MARL_FOR_W_Factory/issues)
- **Discussions**: [GitHub Discussions](https://github.com/Delight0628/MARL_FOR_W_Factory/discussions)

---

<div align="center">

**⭐ 如果这个项目对您有帮助，请给我们一个Star！ ⭐**

```
┌─────────────────────────────────────────────────────────┐
│                                                         │
│   🏭 Made with ❤️ for Smart Manufacturing               │
│                                                         │
│   🤖 Multi-Agent Reinforcement Learning                 │
│   📊 Proximal Policy Optimization                       │
│   🎯 Factory Scheduling                                │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

</div>
