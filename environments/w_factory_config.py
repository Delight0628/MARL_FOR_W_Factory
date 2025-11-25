"""
W工厂生产调度系统 - 全局配置文件
====================================
项目的唯一真理来源 (Single Source of Truth)

包含内容：
- 基础仿真参数（时间单位、超时设置）
- 工作站/设备配置（数量、容量、故障参数）
- 产品工艺路线（加工步骤、时间）
- 订单配置（基础订单、紧急订单、随机生成）
- 奖励系统（完成奖励、延期惩罚、空闲惩罚）
- 训练流程配置（两阶段渐进式训练策略）
- 强化学习超参数（PPO网络、学习率、熵系数）
- 系统资源配置（并行worker数、TensorFlow线程）
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional

# =============================================================================
# 1. 基础仿真参数 (Basic Simulation Parameters)
# =============================================================================
SIMULATION_TIME = 500  
TIME_UNIT = "minutes"  # 时间单位：分钟
SIMULATION_TIMEOUT_MULTIPLIER = 2.0
# =============================================================================
# 8. 核心训练流程配置 (Core Training Flow Configuration)
# =============================================================================
# 两阶段渐进式训练策略：
# - 阶段一：随机订单泛化训练（基础能力习得）+ 25% BASE_ORDERS锚点（防止遗忘）
# - 阶段二：动态事件鲁棒性训练（应对故障和插单）+ 25% BASE_ORDERS锚点（策略稳定）
TRAINING_FLOW_CONFIG = {
    # ========== 阶段一：基础能力训练（随机订单泛化）==========
    # 训练目标：学习通用的调度能力，能处理不同的订单组合
    "foundation_phase": {
        # 毕业标准：连续N次达到以下所有条件才能进入阶段二
        "graduation_criteria": {
            "target_score": 0.70,               # 综合评分阈值
            "target_consistency": 8,            # 连续达标次数
            "tardiness_threshold": 450.0,       # 最大可接受延期（分钟）
            "min_completion_rate": 95.0,        # 最低完成率（%）
        },
        
        # 随机订单生成器配置
        "random_orders_config": {
            "min_orders": 5,
            "max_orders": 8,
            "min_quantity_per_order": 3,
            "max_quantity_per_order": 12,
            "due_date_range": (200.0, 700.0),
            "priority_weights": [0.3, 0.5, 0.2],
        },
        
        # 多任务混合训练配置（防止灾难性遗忘）
        # 原理：每轮数据采集时，部分worker使用BASE_ORDERS作为学习锚点，
        # 其余worker使用随机订单探索新场景，两者梯度融合训练
        "multi_task_mixing": {
            "enabled": True,
            "base_worker_fraction": 0.25,   # 使用BASE_ORDERS的worker比例（0.0~1.0）
            "randomize_base_env": False     # 是否对BASE_ORDERS添加扰动
        },
        
        # 课程学习配置（可选）
        # 逐步增加任务难度，降低训练难度
        "curriculum_learning": {
            "enabled": False,  # 是否启用课程学习
            "stages": [
                {
                    "name": "基础入门", "orders_scale": 0.4, "time_scale": 1.0, "is_final_stage": False,
                    "graduation_criteria": {"target_score": 0.80, "min_completion_rate": 100.0, "target_consistency": 10,"tardiness_threshold": 0.0}
                },
                {
                    "name": "能力提升", "orders_scale": 0.8, "time_scale": 1.0, "is_final_stage": False,
                    "graduation_criteria": {"target_score": 0.80, "min_completion_rate": 100.0, "target_consistency": 10,"tardiness_threshold": 225.0}
                },
                {
                    "name": "完整挑战", "orders_scale": 1.0, "time_scale": 1.0, "is_final_stage": True,
                    "graduation_criteria": {"target_score": 0.72, "min_completion_rate": 100.0, "target_consistency": 6, "tardiness_threshold": 450.0}
                },
            ],
        }
    },

    # ========== 阶段二：鲁棒性训练（动态事件强化）==========
    # 训练目标：提升应对突发事件的能力（设备故障、紧急插单）
    "generalization_phase": {
        # 完成标准：连续N次达到以下所有条件即完成训练
        "completion_criteria": {
            "target_score": 0.60,               # 综合评分阈值（放宽）
            "target_consistency": 10,           # 连续达标次数（更严格）
            "min_completion_rate": 80.0,        # 最低完成率（%，放宽）
        },
        
        # 随机订单生成器配置
        "random_orders_config": {
            "min_orders": 5,
            "max_orders": 8,
            "min_quantity_per_order": 3,
            "max_quantity_per_order": 12,
            "due_date_range": (200.0, 700.0),
            "priority_weights": [0.3, 0.5, 0.2],
        },
        
        # 多任务混合训练配置（防止灾难性遗忘）
        # 原理：部分worker使用BASE_ORDERS作为稳定锚点，
        # 其余worker在随机订单+动态事件环境中强化鲁棒性
        "multi_task_mixing": {
            "enabled": True,
            "base_worker_fraction": 0.25,       # 使用BASE_ORDERS的worker比例
            "randomize_base_env": False         # 保持BASE_ORDERS稳定不变
        },
        
        # 动态事件配置（仅在阶段二启用）
        "dynamic_events": {
            "equipment_failure_enabled": True,  # 启用设备故障模拟
            "emergency_orders_enabled": True,   # 启用紧急插单模拟
        }
    },
    
    # --- 通用训练参数 ---
    "general_params": {
        "max_episodes": 1000,
        "steps_per_episode": 1500,          # 每回合最大步数
        "eval_frequency": 1,               # 默认每回合评估
        "early_stop_patience": 100,
        "performance_window": 15
    }
}


# 随机种子（用于可重复实验）
RANDOM_SEED = 42

# =============================================================================
# 2. 工作站/设备配置 (Workstation/Equipment Configuration)
# =============================================================================

# 工作站配置：设备数量和处理能力 
WORKSTATIONS = {
    "带锯机": {"count": 1, "capacity": 1},        
    "五轴加工中心": {"count": 2, "capacity": 1},   
    "砂光机": {"count": 1, "capacity": 1},        
    "组装台": {"count": 2, "capacity": 1},       
    "包装台": {"count": 2, "capacity": 1},        
}

# 设备故障参数
EQUIPMENT_FAILURE = {
    "enabled": False,                   # 是否启用设备故障 - 静态训练阶段禁用
    "mtbf_hours": 24,                  # 平均故障间隔时间（小时）
    "mttr_minutes": 30,                # 平均修复时间（分钟）
    "failure_probability": 0.02,       # 每小时故障概率
}

# =============================================================================
# 3. 产品工艺路线配置 (Product Process Routes)
# =============================================================================

# 产品工艺路线：每个产品的加工步骤和时间
PRODUCT_ROUTES = {
    "黑胡桃木餐桌": [
        {"station": "带锯机", "time": 8},      
        {"station": "五轴加工中心", "time": 20},  
        {"station": "砂光机", "time": 10},      
        {"station": "组装台", "time": 15},      
        {"station": "包装台", "time": 5},      
    ],
    "橡木书柜": [
        {"station": "带锯机", "time": 12},      
        {"station": "五轴加工中心", "time": 25},  
        {"station": "砂光机", "time": 15},      
        {"station": "组装台", "time": 20},      
        {"station": "包装台", "time": 8},      
    ],
    "松木床架": [
        {"station": "带锯机", "time": 10},      
        {"station": "砂光机", "time": 12},      
        {"station": "组装台", "time": 15},      
        {"station": "包装台", "time": 6},      
    ],
    "樱桃木椅子": [
        {"station": "带锯机", "time": 6},      
        {"station": "五轴加工中心", "time": 12},  
        {"station": "砂光机", "time": 8},      
        {"station": "组装台", "time": 10},      
        {"station": "包装台", "time": 4},      
    ],
}

# =============================================================================
# 4. 订单配置 (Order Configuration)
# =============================================================================

# 基础订单模板
BASE_ORDERS = [
    {"product": "黑胡桃木餐桌", "quantity": 6, "priority": 1, "due_date": 300},  # 数量6个，优先级1，交期时间300分钟
    {"product": "橡木书柜", "quantity": 6, "priority": 2, "due_date": 400},      
    {"product": "松木床架", "quantity": 6, "priority": 1, "due_date": 350},      
    {"product": "樱桃木椅子", "quantity": 4, "priority": 3, "due_date": 280},    
    {"product": "黑胡桃木餐桌", "quantity": 4, "priority": 2, "due_date": 450},  
    {"product": "橡木书柜", "quantity": 6, "priority": 1, "due_date": 320},      
    {"product": "松木床架", "quantity": 4, "priority": 2, "due_date": 380},      
    {"product": "樱桃木椅子", "quantity": 6, "priority": 1, "due_date": 250},    
]

# 队列容量配置
# 动态计算队列容量以防止死锁：取基础订单和随机订单的最大零件数，乘以2作为安全系数
_base_parts_count = sum(order["quantity"] for order in BASE_ORDERS)
_max_random_parts_count = TRAINING_FLOW_CONFIG["generalization_phase"]["random_orders_config"]["max_orders"] * \
                          TRAINING_FLOW_CONFIG["generalization_phase"]["random_orders_config"]["max_quantity_per_order"]
QUEUE_CAPACITY = max(_base_parts_count, _max_random_parts_count) * 2

# 紧急插单配置
EMERGENCY_ORDERS = {
    "enabled": False,                  # 该开关需要保留非训练场景需要默认值，app/交互演示、evaluation 的静态测试
    "arrival_rate": 0.1,               # 每小时紧急订单到达率
    "priority_boost": 0,               # 紧急订单优先级提升
    "due_date_reduction": 0.7,         # 交期缩短比例（0.7表示缩短30%）
}

# =============================================================================
# 5. 强化学习环境参数 (RL Environment Parameters)
# =============================================================================

# 观测空间配置
# 
# 观测空间结构（总维度132）：
#   [1] Agent自身特征 (8维): 身份one-hot(5) + 容量(1) + 繁忙率(1) + 故障状态(1)
#   [2] 全局宏观特征 (4维): 时间进度、WIP率、瓶颈拥堵度、队列长度
#   [3] 当前队列摘要 (30维): 6种特征 × 5种统计量
#   [4] 候选工件详细 (90维): 9维特征 × 10个候选工件（含时间压力感知）

ENHANCED_OBS_CONFIG = {
    # 候选工件配置
    "num_candidate_workpieces": 10,         # 候选工件数量
    "num_urgent_candidates": 5,             # EDD策略候选数（最早交期优先）
    "num_short_candidates": 3,              # SPT策略候选数（最短加工时间优先）
    "num_random_candidates": 2,             # 随机采样候选数（增加多样性）
    
    # 归一化参数
    "max_op_duration_norm": 60.0,           # 操作时长归一化基准
    "max_bom_ops_norm": 20,                 # 剩余工步数归一化基准
    "total_remaining_time_norm": 1000.0,    # 总剩余加工时间归一化基准
    "w_station_capacity_norm": 10.0,        # 队列长度归一化基准
    
    # 队列摘要统计配置
    "queue_summary_features": 6,            # 特征种类数（剩余工序、时间等）
    "queue_summary_stats": 5,               # 统计类型数（min, max, mean, std, median）
    
    # 候选工件特征配置
    # 9维特征：exists、剩余工序、剩余时间、当前工序时间、下游拥堵、
    #         优先级、是否最终工序、产品类型、时间压力感知
    "candidate_feature_dim": 9,
    
    # 压缩归一化开关：先归一化x/norm，再压缩y=y/(1+y)到(0,1)，防止特征饱和
    "use_compressed_norm": True,
}

# 🔧 方案A：纯候选动作空间配置（移除启发式作弊）
# 新设计理念：
#   [1] IDLE动作 (0): 允许agent选择等待
#   [2] 候选动作 (1-10): 让agent从多样性采样的候选工件中学习选择
#   [3] 移除所有启发式策略动作，强制agent学习真正的调度逻辑
# 这种设计确保智能体必须从零开始学习，而不是依赖内置算法
ACTION_CONFIG_ENHANCED = {
    # 移除固定的动作空间大小，因为它现在由环境根据设备数动态生成
    # "action_space_size": 11,  # 0=IDLE, 1-10=候选工件
    "action_names": [
        "IDLE",                          # 0: 不处理（等待）
        "CANDIDATE_1", "CANDIDATE_2",    # 1-2: 候选工件1-2（多样性采样）
        "CANDIDATE_3", "CANDIDATE_4",    # 3-4: 候选工件3-4
        "CANDIDATE_5", "CANDIDATE_6",    # 5-6: 候选工件5-6
        "CANDIDATE_7", "CANDIDATE_8",    # 7-8: 候选工件7-8
        "CANDIDATE_9", "CANDIDATE_10",   # 9-10: 候选工件9-10
    ],
}


# =============================================================================
# 6. 奖励系统配置 (Reward System) - V2：稠密、目标导向的设计
# =============================================================================

# 奖励退火配置（用于逐步关闭启发式护栏）
REWARD_ANNEALING_CONFIG = {
    "ANNEALING_END_EPISODE": 300,
}


REWARD_CONFIG = {
    # ============================================================
    # 第一层：任务完成奖励（主导信号）
    # ============================================================
    "part_completion_reward": 80.0,        
    "final_all_parts_completion_bonus": 500.0, 
    
    # ============================================================
    # 第二层：时间质量奖励（次要信号）
    # ============================================================
    "on_time_completion_reward": 80.0,      
    # 调低延期惩罚基准，避免负项主导
    "tardiness_penalty_scaler": -4.0,     
    
    # ============================================================
    # 第三层：过程塑形奖励（引导信号）
    # ============================================================
    # 3.1 进度塑形（鼓励持续推进）
    "progress_shaping_coeff": 0.1,          
    
    # 3.2 行为约束（最小化惩罚）
    "unnecessary_idle_penalty": -1.0,      
    "invalid_action_penalty": -0.5,      
    
    # 3.3 紧急度引导
    "urgency_reduction_reward": 0.1,         
    
    # 3.4 (核心改进) 基于负松弛时间的持续惩罚
    # 提供即时、密集的惩罚信号, 迫使智能体优先处理预计延期的工件
    # 调低松弛度持续惩罚系数，配合tanh与阈值裁剪
    "slack_time_penalty_coeff": -0.03, 
    # 迟期惩罚使用Huber稳健化（基于归一化迟期，单位: 无量纲）
    "use_huber_tardiness": True,
    "tardiness_huber_delta_norm": 0.3,
    # 松弛度惩罚的tanh缩放（分钟）与每步绝对上限（单agent）
    "slack_penalty_tanh_scale": 240.0,
    "slack_penalty_max_abs": 50.0,
    
    # ============================================================
    # 🔧 新增：基于Slack的非线性迟交惩罚（奖励函数重塑）
    # ============================================================
    # 对即将迟交的订单施加分段惩罚，避免在安全区间过度干扰
    "slack_based_tardiness_enabled": True,        # 是否启用基于Slack的惩罚
    "slack_tardiness_step_penalty": -0.8,        # 即将迟交的强惩罚系数（负松弛时间）
    "slack_tardiness_overdue_penalty": -3.0,     # 已迟交的增量惩罚系数（平方/Huber）
    "slack_tardiness_threshold": 0.0,             # Slack阈值，小于此值开始惩罚
    "slack_tardiness_normalize_scale": 480.0,    # 归一化基准（分钟）
    "wip_penalty_coeff": -0.01,                   # WIP拥塞惩罚系数
    "idle_penalty_coeff": -0.005,                 # 瓶颈闲置惩罚系数
}

# =============================================================================
# 7. 环境随机化配置 (Environment Randomization)
# =============================================================================
ENV_RANDOMIZATION_CONFIG = {
    "due_date_jitter": 50.0,      # 交货日期抖动范围 (+/- 分钟)
    "arrival_time_jitter": 30.0,  # 到达时间抖动范围 (0 to X 分钟)
}

# =============================================================================
# 8. 自定义MAPPO训练配置 (Custom PPO Training Configuration)
# =============================================================================

# PPO网络架构配置
PPO_NETWORK_CONFIG = {
    "hidden_sizes": [1024, 512, 256],   
    "dropout_rate": 0.1,
    "clip_ratio": 0.2,
    # 降低熵系数，避免熵项压制策略改进
    "entropy_coeff": 0.05,               
    "ppo_epochs": 12,                   
    "num_minibatches": 4,                
    "grad_clip_norm": 1.0,               # 🔧 新增：梯度裁剪的范数
    "advantage_clip_val": 5.0,           # 🔧 新增：优势函数的裁剪值
    "gamma": 0.99,                       # GAE折扣因子
    "lambda_gae": 0.95,                  # GAE平滑参数
}

# 🔧 新增：自适应熵调整配置
ADAPTIVE_ENTROPY_CONFIG = {
    # 暂停自适应熵，先确保策略能稳定朝优势方向改进
    "enabled": False,             # 是否启用
    "start_episode": 0,          # 🔧 从20改为0，立即启动自适应机制
    "patience": 30,              # 🔧 从200降到30，更快响应停滞
    "boost_factor": 0.15,        # 🔧 从0.1提升到0.15，更强的探索提升
    "high_completion_decay": 0.995, # 🔧 从0.999改为0.995，更快衰减避免过度探索
    "high_completion_threshold": 0.95, # 🔧 新增：定义"高完成率"的阈值
    "min_entropy": 0.01,         # 🔧 从0.005提升到0.01，保持最低探索水平
}

# 🔧 新增：评估流程配置
EVALUATION_CONFIG = {
    "exploration_rate": 0.0,  # 评估时使用的随机探索率，设置为0则为纯粹的确定性评估
    "deterministic_candidates": True, # 在评估时使用确定性候选，确保启发式基线可复现
}

# 说明：
# - evaluation.py 会将 EVALUATION_CONFIG 合并进评估环境，因此默认评估为确定性候选。
# - 训练阶段内置的 quick_kpi_evaluation 也会显式注入 deterministic_candidates=True，
#   以保证训练期评估的可复现性，与离线评估保持一致。

# 学习率调度配置
LEARNING_RATE_CONFIG = {
    # 适度提升初始学习率，配合较小熵项
    "initial_lr": 2e-4,
    "end_lr": 1e-6,
    "decay_power": 0.8,
    "critic_lr_multiplier": 0.5,         # 专家修复：为Critic设置一个较低的学习率乘数，以稳定价值学习
}

# 系统资源配置
SYSTEM_CONFIG = {
    "num_parallel_workers": 4,           # 并行worker数量（建议4-6，根据CPU核心数调整）
    "tf_inter_op_threads": 4,            # TensorFlow inter-op线程数（并行操作间）
    "tf_intra_op_threads": 8,            # TensorFlow intra-op线程数（单操作内）
}


# =============================================================================
# 10. 随机领域生成配置 (Random Domain Generation)
# =============================================================================

def generate_random_orders() -> List[Dict[str, Any]]:
    """
    生成随机订单配置（用于泛化训练）
    
    每次调用返回一套全新的随机订单组合，包含：
    - 随机订单数量（min_orders ~ max_orders）
    - 随机产品类型组合
    - 随机数量（min_quantity ~ max_quantity）
    - 随机优先级（基于权重）
    - 随机交期（due_date_range范围内）
    
    Returns:
        List[Dict]: 订单列表，每个订单包含 product, quantity, priority, due_date
    """
    import random
    
    config = TRAINING_FLOW_CONFIG["generalization_phase"]["random_orders_config"]
    product_types = list(PRODUCT_ROUTES.keys())
    
    # 随机订单数量
    num_orders = random.randint(config["min_orders"], config["max_orders"])
    
    generated_orders = []
    for i in range(num_orders):
        # 随机产品类型
        product = random.choice(product_types)
        
        # 随机订单数量
        quantity = random.randint(
            config["min_quantity_per_order"], 
            config["max_quantity_per_order"]
        )
        
        # 随机优先级（按权重采样）
        priority = random.choices([1, 2, 3], weights=config["priority_weights"])[0]
        
        # 随机交期
        due_date = random.uniform(*config["due_date_range"])
        
        generated_orders.append({
            "product": product,
            "quantity": quantity,
            "priority": priority,
            "due_date": due_date
        })
    
    return generated_orders


# =============================================================================
# 7. 评分与辅助函数 (Scoring and Helper Functions)
# =============================================================================

def calculate_episode_score(kpi_results: Dict[str, float], config: Dict = None) -> float:
    """
    根据单次仿真的KPI结果计算综合评分。
    config: WFactorySim的环境配置，用于获取课程学习信息
    """
    config = config or {}
    
    # 适配 `get_final_stats` 和 `quick_kpi_evaluation` 的不同key
    makespan = kpi_results.get('makespan', kpi_results.get('mean_makespan', 0))
    completed_parts = kpi_results.get('total_parts', kpi_results.get('mean_completed_parts', 0))
    utilization = kpi_results.get('mean_utilization', 0)
    tardiness = kpi_results.get('total_tardiness', kpi_results.get('mean_tardiness', 0))
    
    if completed_parts == 0:
        return 0.0
    
    makespan_score = max(0, 1 - makespan / (SIMULATION_TIME * 1.5))
    utilization_score = utilization
    tardiness_score = max(0, 1 - tardiness / (SIMULATION_TIME * 2.0))
    
    # 获取目标零件数
    if 'custom_orders' in config:
        target_parts = get_total_parts_count(config['custom_orders'])
    elif 'orders_scale' in config:
        target_parts = int(get_total_parts_count() * config.get('orders_scale', 1.0))
    else:
        target_parts = get_total_parts_count()

    # 🔧 裁剪完成率至[0,1]，避免动态插单导致>100%而扭曲评分
    completion_score = completed_parts / target_parts if target_parts > 0 else 0
    if completion_score > 1.0:
        completion_score = 1.0
    
    current_score = (
        completion_score * 0.40 +
        tardiness_score * 0.35 +
        makespan_score * 0.15 +
        utilization_score * 0.1
    )
    return current_score


def get_total_parts_count(orders_list: Optional[List[Dict[str, Any]]] = None) -> int:
    """
    获取指定订单列表的总零件数。
    如果未提供订单列表，则默认计算基础订单 (BASE_ORDERS) 的总数。
    """
    if orders_list is None:
        orders_to_process = BASE_ORDERS
    else:
        orders_to_process = orders_list
    return sum(order["quantity"] for order in orders_to_process)


def get_route_for_product(product: str) -> List[Dict[str, Any]]:
    """获取指定产品的工艺路线"""
    return PRODUCT_ROUTES.get(product, [])

def calculate_product_total_time(product: str) -> float:
    """计算产品总加工时间"""
    route = get_route_for_product(product)
    return sum(step["time"] for step in route)

def validate_config() -> bool:
    """验证配置文件的完整性和一致性"""
    # 检查工作站是否在产品路线中都有定义
    all_stations_in_routes = set()
    for route in PRODUCT_ROUTES.values():
        for step in route:
            all_stations_in_routes.add(step["station"])
    
    defined_stations = set(WORKSTATIONS.keys())
    
    if not all_stations_in_routes.issubset(defined_stations):
        missing = all_stations_in_routes - defined_stations
        print(f"警告：以下工作站在产品路线中使用但未定义：{missing}")
        return False
    
    # 检查订单中的产品是否都有对应的工艺路线
    order_products = set(order["product"] for order in BASE_ORDERS)
    defined_products = set(PRODUCT_ROUTES.keys())
    
    if not order_products.issubset(defined_products):
        missing = order_products - defined_products
        print(f"警告：以下产品在订单中使用但未定义工艺路线：{missing}")
        return False
    
    total_parts = sum(order["quantity"] for order in BASE_ORDERS)
    total_processing_time = 0
    
    for order in BASE_ORDERS:
        product_time = calculate_product_total_time(order["product"])
        total_processing_time += product_time * order["quantity"]
    
    # 计算瓶颈工作站的理论最小完工时间
    bottleneck_time = {}
    for station_name, station_config in WORKSTATIONS.items():
        station_load = 0
        for order in BASE_ORDERS:
            route = get_route_for_product(order["product"])
            for step in route:
                if step["station"] == station_name:
                    station_load += step["time"] * order["quantity"]
        
        # 考虑设备数量的并行处理能力
        bottleneck_time[station_name] = station_load / station_config["count"]
    
    theoretical_makespan = max(bottleneck_time.values())
    
    print("配置挑战性验证:")
    print(f"总零件数: {total_parts}")
    print(f"总加工时间: {total_processing_time:.1f}分钟")
    print(f"理论最短完工时间: {theoretical_makespan:.1f}分钟")
    print(f"仿真时间限制: {SIMULATION_TIME}分钟")
    
    if theoretical_makespan > SIMULATION_TIME * 0.8:
        print(f"🎯 环境具有高挑战性 (理论完工时间占仿真时间{theoretical_makespan/SIMULATION_TIME*100:.1f}%)")
    elif theoretical_makespan > SIMULATION_TIME * 0.5:
        print(f"⚠️ 环境具有中等挑战性 (理论完工时间占仿真时间{theoretical_makespan/SIMULATION_TIME*100:.1f}%)")
    else:
        print(f"❌ 环境挑战性不足 (理论完工时间仅占仿真时间{theoretical_makespan/SIMULATION_TIME*100:.1f}%)")
    
    # 检查瓶颈工作站
    bottleneck_station = max(bottleneck_time, key=bottleneck_time.get)
    print(f"🔍 瓶颈工作站: {bottleneck_station} (负荷: {bottleneck_time[bottleneck_station]:.1f}分钟)")
    
    print("配置文件验证通过！")
    return True

# 在模块加载时验证配置
if __name__ == "__main__":
    validate_config()