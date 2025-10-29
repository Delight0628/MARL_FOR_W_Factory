# W工厂生产调度多智能体强化学习系统

<div align="center">

**基于MAPPO算法的智能工厂生产调度系统**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.15](https://img.shields.io/badge/TensorFlow-2.15-orange.svg)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[English](#) | [中文](README_CN.md)

</div>

---

## 📋 目录

- [项目概述](#-项目概述)
- [核心技术](#-核心技术)
- [系统架构](#-系统架构)
- [环境设计](#-环境设计)
- [MAPPO算法实现](#-mappo算法实现)
- [训练流程](#-训练流程)
- [快速开始](#-快速开始)
- [性能评估](#-性能评估)
- [技术亮点](#-技术亮点)
- [项目结构](#-项目结构)

---

## 🎯 项目概述

### 项目背景

本项目旨在解决**多品种、小批量**生产模式下的智能工厂调度问题。W工厂作为典型的现代家具制造企业，面临以下挑战：

- 🏭 **5种工作站**：带锯机、五轴加工中心、砂光机、组装台、包装台
- 📦 **4种产品**：黑胡桃木餐桌、橡木书柜、松木床架、樱桃木椅子
- 🔄 **复杂工艺路线**：每种产品3-5道工序，共享设备资源
- ⏰ **严格交期要求**：多订单并行，需权衡完工时间与延期惩罚
- ⚡ **动态扰动**：设备故障、紧急插单等突发事件

### 核心目标

通过**多智能体强化学习（MARL）**训练一组协同工作的智能体，实现：

1. **最小化总完工时间 (Makespan)**
2. **最大化设备利用率 (Utilization)**  
3. **最小化订单延期 (Tardiness)**
4. **提升动态事件鲁棒性**

---

## 🔬 核心技术

### 技术栈

| 组件 | 技术 | 版本 | 用途 |
|------|------|------|------|
| 深度学习框架 | TensorFlow | 2.15.0 | 神经网络构建与训练 |
| 强化学习算法 | MAPPO | Custom | 多智能体策略优化 |
| 环境接口 | PettingZoo | 1.24+ | 多智能体环境标准 |
| 离散事件仿真 | SimPy | 4.0+ | 工厂物理过程模拟 |
| 并行计算 | ProcessPoolExecutor | Python 3.8+ | 多进程数据采集 |
| 可视化应用 | Streamlit | 1.30+ | 交互式调度演示 |

### 算法特性

- ✅ **集中式Critic + 分布式Actor (CTDE)**：利用全局信息训练，分布式执行
- ✅ **两阶段渐进式训练**：基础泛化 → 动态鲁棒性
- ✅ **课程学习支持**：从简单任务逐步提升难度
- ✅ **自适应熵调整**：动态平衡探索与利用
- ✅ **多任务混合训练**：防止灾难性遗忘
- ✅ **GAE优势函数估计**：提升样本效率

---

## 🏗️ 系统架构

### 整体架构图

```
┌─────────────────────────────────────────────────────────────┐
│                     训练管理层 (Training Layer)               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ auto_train.py│  │TensorBoard   │  │Checkpoint    │      │
│  │ 自动化训练    │  │实时监控       │  │模型管理       │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│              MAPPO算法层 (ppo_marl_train.py)                 │
│  ┌────────────────────────────────────────────────────┐     │
│  │  SimplePPOTrainer (训练器)                          │     │
│  │  ├─ PPONetwork (Actor-Critic网络)                   │     │
│  │  ├─ ExperienceBuffer (经验缓冲)                     │     │
│  │  └─ 并行环境采集 (ProcessPoolExecutor)              │     │
│  └────────────────────────────────────────────────────┘     │
│           ↓ (state, reward)          ↑ (action)            │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│           PettingZoo接口层 (w_factory_env.py)                │
│  ┌────────────────────────────────────────────────────┐     │
│  │  WFactoryEnv (ParallelEnv)                          │     │
│  │  ├─ 观测空间构建 (132维)                            │     │
│  │  ├─ 动作空间映射 (MultiDiscrete)                    │     │
│  │  └─ 奖励计算与返回                                  │     │
│  └────────────────────────────────────────────────────┘     │
│           ↓ (control signals)       ↑ (sim state)          │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│          SimPy仿真层 (WFactorySim Class)                     │
│  ┌────────────────────────────────────────────────────┐     │
│  │  离散事件仿真核心                                    │     │
│  │  ├─ 资源管理 (simpy.Resource)                       │     │
│  │  ├─ 队列系统 (simpy.Store)                          │     │
│  │  ├─ 零件流转 (_part_process)                        │     │
│  │  ├─ 设备故障 (_equipment_failure_process)           │     │
│  │  └─ 紧急插单 (_emergency_order_process)             │     │
│  └────────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│          配置层 (w_factory_config.py)                        │
│  工作站配置 | 产品路线 | 订单数据 | 奖励系统 | 训练参数       │
└─────────────────────────────────────────────────────────────┘
```

### MAPPO网络架构

```
┌─────────────────────────────────────────────────────────────┐
│                    Actor Network (分布式执行)                 │
│  输入: 局部观测 (132维)                                       │
│    ↓                                                         │
│  Dense(1024) + ReLU + Dropout(0.1)                          │
│    ↓                                                         │
│  Dense(512) + ReLU + Dropout(0.1)                           │
│    ↓                                                         │
│  Dense(256) + ReLU + Dropout(0.1)                           │
│    ↓                                                         │
│  MultiHead Output (10 heads × 11 actions/head)              │
│  ├─ Head 1: Softmax(11) → 候选工件1的概率分布                │
│  ├─ Head 2: Softmax(11) → 候选工件2的概率分布                │
│  └─ ...                                                      │
│  输出: 动作概率分布 (支持无放回采样)                          │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                   Critic Network (集中式训练)                 │
│  输入: 全局状态 (global_state + agent_one_hot)               │
│    ↓                                                         │
│  Dense(1024) + ReLU + Dropout(0.1)                          │
│    ↓                                                         │
│  Dense(512) + ReLU + Dropout(0.1)                           │
│    ↓                                                         │
│  Dense(256) + ReLU + Dropout(0.1)                           │
│    ↓                                                         │
│  Dense(1) → 状态价值估计 V(s)                                │
└─────────────────────────────────────────────────────────────┘
```

---

## 🌍 环境设计

### 观测空间 (132维)

环境为每个智能体提供丰富的观测信息，结构如下：

```
┌─────────────────────────────────────────────────────────────┐
│  [1] Agent自身特征 (8维)                                     │
│      ├─ 身份one-hot (5维): [1,0,0,0,0] 表示带锯机            │
│      ├─ 设备容量 (1维): 归一化容量值                          │
│      ├─ 设备繁忙率 (1维): 当前繁忙程度                        │
│      └─ 故障状态 (1维): 0=正常, 1=故障                        │
├─────────────────────────────────────────────────────────────┤
│  [2] 全局宏观特征 (4维)                                      │
│      ├─ 时间进度 (1维): current_time / SIMULATION_TIME      │
│      ├─ WIP率 (1维): 在制品数量 / 总订单数                   │
│      ├─ 瓶颈拥堵度 (1维): 最长队列长度归一化                  │
│      └─ 平均队列长度 (1维): 所有队列平均长度                  │
├─────────────────────────────────────────────────────────────┤
│  [3] 当前队列摘要 (30维)                                     │
│      6种特征 × 5种统计量 (min, max, mean, std, median)       │
│      特征类型:                                               │
│      ├─ 剩余工序数                                           │
│      ├─ 剩余加工时间                                         │
│      ├─ 当前工序时间                                         │
│      ├─ 下游拥堵度                                           │
│      ├─ 优先级                                               │
│      └─ 是否最终工序                                         │
├─────────────────────────────────────────────────────────────┤
│  [4] 候选工件详细特征 (90维)                                 │
│      10个候选工件 × 9维特征/工件                             │
│      工件特征:                                               │
│      ├─ exists (1维): 候选工件是否存在                        │
│      ├─ 剩余工序数 (1维): 归一化                             │
│      ├─ 剩余加工时间 (1维): 归一化                           │
│      ├─ 当前工序时间 (1维): 归一化                           │
│      ├─ 下游拥堵度 (1维): 下一工位队列长度                    │
│      ├─ 优先级 (1维): 1/2/3 → 归一化                         │
│      ├─ 是否最终工序 (1维): 0/1标志位                        │
│      ├─ 产品类型one-hot (1维): 4种产品编码                   │
│      └─ ⭐时间压力感知 (1维): (due_date - current_time) / ...│
└─────────────────────────────────────────────────────────────┘
总维度: 8 + 4 + 30 + 90 = 132维
```

**关键设计理念**：

1. **多样性候选采样**：混合EDD(紧急优先)、SPT(短作业优先)、随机采样，提供探索空间
2. **时间压力感知**：基于物理时间关系计算，非启发式作弊
3. **压缩归一化**：使用 `y = x/norm / (1 + x/norm)` 避免特征饱和

### 动作空间 (MultiDiscrete)

每个智能体的动作空间为 `MultiDiscrete([11, 11, ..., 11])` (10个头)：

```
动作空间设计:
├─ 0: IDLE (空闲等待)
├─ 1-10: 选择候选工件1-10进行加工
└─ 采用无放回采样机制，避免多头选择相同工件
```

**无放回采样机制**：

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
┌─────────────────────────────────────────────────────────────┐
│  第一层：任务完成奖励 (主导信号)                              │
│  ├─ 零件完成奖励: +80                                        │
│  └─ 全部完成奖励: +500 (所有零件完成时)                       │
├─────────────────────────────────────────────────────────────┤
│  第二层：时间质量奖励 (次要信号)                              │
│  ├─ 按时完成奖励: +80 (零件在交期内完成)                      │
│  └─ 延期惩罚: -10 × tardiness_minutes                        │
├─────────────────────────────────────────────────────────────┤
│  第三层：过程塑形奖励 (引导信号)                              │
│  ├─ 进度塑形: +0.1 × progress_made                           │
│  ├─ 紧急度降低: +0.1 × urgency_reduction                     │
│  ├─ 不必要空闲惩罚: -1.0                                     │
│  ├─ 无效动作惩罚: -0.5                                       │
│  └─ 负松弛时间惩罚: -0.1 × max(0, -slack_time)               │
└─────────────────────────────────────────────────────────────┘
```

**设计理念**：密集即时反馈 + 长期目标导向

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
|------|----|----|
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
┌──────────────────────────────────────────────────────────────┐
│  阶段一：基础泛化训练 (Foundation Phase)                       │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  目标：掌握随机订单环境下的泛化调度能力                  │  │
│  │  ┌──────────────────────────────────────────────────┐  │  │
│  │  │  可选：课程学习 (Curriculum Learning)             │  │  │
│  │  │  ├─ 阶段1: 40%订单 → 目标分数0.80, 零延期         │  │  │
│  │  │  ├─ 阶段2: 80%订单 → 目标分数0.80, 延期<225min   │  │  │
│  │  │  └─ 阶段3: 100%订单 → 目标分数0.72, 延期<450min  │  │  │
│  │  └──────────────────────────────────────────────────┘  │  │
│  │                                                          │  │
│  │  训练策略：                                               │  │
│  │  ├─ 75% workers: 随机订单 (5-8订单, 每单3-12件)         │  │
│  │  └─ 25% workers: BASE_ORDERS (稳定锚点，防遗忘)         │  │
│  │                                                          │  │
│  │  毕业标准：                                               │  │
│  │  ├─ 综合评分 > 0.70                                     │  │
│  │  ├─ 完成率 > 95%                                        │  │
│  │  ├─ 延期 < 450min                                       │  │
│  │  └─ 连续8次达标                                         │  │
│  └────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────┘
                            ↓
┌──────────────────────────────────────────────────────────────┐
│  阶段二：动态事件鲁棒性训练 (Generalization Phase)             │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  目标：在动态扰动下保持调度性能                          │  │
│  │                                                          │  │
│  │  训练策略：                                               │  │
│  │  ├─ 75% workers: 随机订单 + 动态事件                     │  │
│  │  │   ├─ 设备故障 (MTBF=24h, MTTR=30min)                │  │
│  │  │   └─ 紧急插单 (到达率0.1/h)                          │  │
│  │  └─ 25% workers: BASE_ORDERS (保持基准性能)             │  │
│  │                                                          │  │
│  │  完成标准：                                               │  │
│  │  ├─ 综合评分 > 0.60                                     │  │
│  │  ├─ 完成率 > 80%                                        │  │
│  │  └─ 连续10次达标                                        │  │
│  └────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────┘
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
git clone https://github.com/your-repo/MARL_FOR_W_Factory.git
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

#### 方式1: 使用自动化训练脚本（推荐）

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
- 训练损失曲线 (Actor Loss, Critic Loss)
- 策略熵变化
- KPI指标 (Makespan, Utilization, Tardiness)
- 综合评分趋势

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
|------|------|------|
| **Makespan** | 总完工时间（分钟） | ↓ 越小越好 |
| **Mean Utilization** | 平均设备利用率 | ↑ 越高越好 |
| **Total Tardiness** | 总延期时间（分钟） | ↓ 越小越好 |
| **Completed Parts** | 完成零件数 | ↑ 达到100% |
| **Comprehensive Score** | 综合评分 (0-1) | ↑ 越高越好 |

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

- **FIFO** (First In First Out): 先到先服务
- **EDD** (Earliest Due Date): 最早交期优先
- **SPT** (Shortest Processing Time): 最短加工时间优先
- **CR** (Critical Ratio): 紧急度比率优先

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

## ✨ 技术亮点

### 1. 集中式训练分布式执行 (CTDE)

**设计理念**：

- **训练时**：Critic利用全局信息（包含所有智能体状态）评估价值
- **执行时**：Actor仅依赖局部观测，支持分布式部署

**实现细节**：

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
├── environments/                    # 环境模块
│   ├── w_factory_env.py            # PettingZoo环境 + SimPy仿真
│   └── w_factory_config.py         # 统一配置文件（单一真理源）
│
├── mappo/                           # MAPPO算法
│   ├── ppo_marl_train.py           # 训练主脚本
│   ├── ppo_models/                 # 模型保存目录
│   └── tensorboard_logs/           # TensorBoard日志
│
├── app/                             # 可视化应用
│   ├── app_scheduler.py            # Streamlit交互界面
│   └── custom_products.json        # 自定义产品配置
│
├── auto_train.py                    # 自动化训练管理器
├── evaluation.py                    # 模型评估脚本
├── debug_marl_behavior.py          # 调试工具
│
├── requirements.txt                 # Python依赖
├── README.md                        # 项目说明（本文件）
└── .gitignore                       # Git忽略规则
```

### 核心文件说明

| 文件 | 代码量 | 核心功能 |
|------|--------|---------|
| `w_factory_env.py` | ~1720行 | SimPy仿真 + PettingZoo接口 |
| `ppo_marl_train.py` | ~2630行 | MAPPO算法 + 训练流程 |
| `w_factory_config.py` | ~550行 | 全局配置（工作站/订单/奖励/训练参数） |
| `evaluation.py` | ~790行 | 模型评估 + 启发式对比 + 甘特图 |
| `app_scheduler.py` | ~1330行 | Streamlit可视化应用 |

---

## 🎨 可视化应用

### 启动交互式调度系统

```bash
streamlit run app/app_scheduler.py
```

### 应用功能

1. **模型选择**
   - 自动扫描已训练模型
   - 支持多检查点切换

2. **订单配置**
   - 内置BASE_ORDERS
   - 支持自定义订单
   - 实时验证订单合理性

3. **调度执行**
   - 一键启动仿真
   - 实时进度显示
   - KPI实时更新

4. **结果可视化**
   - 甘特图展示（设备维度/订单维度）
   - 设备利用率柱状图
   - KPI对比雷达图

5. **数据导出**
   - 下载调度历史（CSV）
   - 保存甘特图（PNG）

---