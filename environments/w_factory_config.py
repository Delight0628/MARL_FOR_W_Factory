"""
W工厂生产调度系统配置文件
这是项目的唯一真理来源 (Single Source of Truth)
包含所有工厂参数、设备信息、产品工艺路线和订单数据
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional

# =============================================================================
# 1. 基础仿真参数 (Basic Simulation Parameters)
# =============================================================================
SIMULATION_TIME = 600  # 10小时
TIME_UNIT = "minutes"  # 时间单位：分钟

# =============================================================================
# 8. 核心训练流程配置 (Core Training Flow Configuration)
# =============================================================================
TRAINING_FLOW_CONFIG = {
    # --- 阶段一：基础能力训练 ---
    # 目标：在标准静态环境下，让模型掌握完成100%任务的核心能力。
    "foundation_phase": {
        # 毕业标准：必须连续N次达到以下所有条件
        "graduation_criteria": {
            "target_score": 0.72,
            "target_consistency": 6,
            "tardiness_threshold": 450.0,  # 总延期不得超过450分钟
            "min_completion_rate": 100.0,   # 必须100%完成
        },
        
        # 可选：在基础训练内部启用课程学习，以循序渐进的方式达到最终目标
        "curriculum_learning": {
            "enabled": True,  # 关键开关：是否启用课程学习
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

    # --- 阶段二：泛化能力强化 ---
    # 目标：在动态随机环境下，训练模型的鲁棒性和对未知任务的适应能力。
    "generalization_phase": {
        # 训练完成标准：连续N次达到以下所有条件
        "completion_criteria": {
            "target_score": 0.65,  # 泛化阶段分数要求可略微放宽
            "target_consistency": 10, # 需要更长时间的稳定表现
            "min_completion_rate": 85.0, # 允许在随机高难度任务下有少量未完成
        },
        
        # 随机订单生成器配置
        "random_orders_config": {
            "min_orders": 5,
            "max_orders": 8,
            "min_quantity_per_order": 3,
            "max_quantity_per_order": 12,
            "due_date_range": (200.0, 700.0),
            "priority_weights": [0.3, 0.5, 0.2],
        }
    },
    
    # --- 通用训练参数 ---
    "general_params": {
        "max_episodes": 1000,
        "steps_per_episode": 1500,          # 🔧 新增：每回合最大步数
        "eval_frequency": 20,               # 🔧 新增：评估频率
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

# 队列设置
# 🔧 缺陷修复：动态计算队列容量以防止死锁
# 容量基于基础订单和随机订单可能产生的最大零件数，并乘以2作为安全系数
_base_parts_count = sum(order["quantity"] for order in BASE_ORDERS)
_max_random_parts_count = TRAINING_FLOW_CONFIG["generalization_phase"]["random_orders_config"]["max_orders"] * \
                          TRAINING_FLOW_CONFIG["generalization_phase"]["random_orders_config"]["max_quantity_per_order"]
QUEUE_CAPACITY = max(_base_parts_count, _max_random_parts_count) * 2

# 紧急插单配置
EMERGENCY_ORDERS = {
    "enabled": False,                  # 是否启用紧急插单 - 静态训练阶段禁用
    "arrival_rate": 0.1,               # 每小时紧急订单到达率
    "priority_boost": 0,               # 紧急订单优先级提升
    "due_date_reduction": 0.7,         # 交期缩短比例
}

# =============================================================================
# 5. 强化学习环境参数 (RL Environment Parameters)
# =============================================================================

# 🔧 V2 修复：重构的、信息更丰富的观测空间配置
ENHANCED_OBS_CONFIG = {
    "enabled": True,
    "obs_slot_size": 5,                     # 观测队列中前5个工件
    "max_op_duration_norm": 60.0,           # 用于归一化操作时长的最大值
    "max_bom_ops_norm": 20,                 # 用于归一化剩余工步数的最大值
    "time_slack_norm": 480.0,               # 用于归一化松弛时间的基准值 (一个8小时班次)
    "total_remaining_time_norm": 960.0,     # 用于归一化总剩余加工时间的基准值 (两个8小时班次)
    "w_station_capacity_norm": 10.0,        # 用于归一化队列长度的基准值
}

# 队列视图配置：启用按紧急度排序以去除“索引偏置”
QUEUE_VIEW_CONFIG = {
    "enabled": True,        # 若为True，则状态与动作均基于“紧急度排序视图”
}

# 动作空间配置，与观测空间保持一致
ACTION_CONFIG_ENHANCED = {
    "enabled": True,
    # 动作空间自动适应观测配置
    "action_space_size": ENHANCED_OBS_CONFIG["obs_slot_size"] + 1,
    "action_names": ["IDLE"] + [f"PROCESS_MOST_URGENT_{i+1}" for i in range(ENHANCED_OBS_CONFIG["obs_slot_size"])],
}


# =============================================================================
# 6. 奖励系统配置 (Reward System) - V2：稠密、目标导向的设计
# =============================================================================

# 奖励退火配置（用于逐步关闭启发式护栏）
REWARD_ANNEALING_CONFIG = {
    "ANNEALING_END_EPISODE": 100,
}

# 启发式护栏配置（只在错误极端时介入，且随训练退火）
HEURISTIC_GUARDRAILS_CONFIG = {
    "enabled": True,
    "critical_choice_penalty": 0.5, # 专家修复：名称调整并增加惩罚力度
    "critical_slack_threshold": -60.0,  # 分钟；更紧急
    "safe_slack_threshold": 120.0,      # 分钟；更安全
}

REWARD_CONFIG = {
    # === 事件驱动奖励 (Event-driven Rewards) ===
    "on_time_completion_reward": 10.0,        # 按时或提前完成一个工件的基础奖励
    "tardiness_penalty_scaler": -10.0,        # 延期惩罚的缩放系数，最终惩罚 = 此系数 * (延期分钟数 / 480)

    # === 行为塑造惩罚 (Behavior Shaping Penalties) ===
    "unnecessary_idle_penalty": -10.0,        # 在有工件排队时选择“空闲”动作的惩罚

    # === 终局奖励 (Episode End Bonus) ===
    "final_all_parts_completion_bonus": 1000.0, # 全部完成时给予的巨大奖励，激励完成所有任务
    "invalid_action_penalty": -5.0,          # 选择一个无效的动作（比如队列为空的槽位）
}



# =============================================================================
# 8. 自定义MAPPO训练配置 (Custom PPO Training Configuration)
# =============================================================================

# PPO网络架构配置
PPO_NETWORK_CONFIG = {
    "hidden_sizes": [1024, 512, 256],    # 🔧 关键：增加网络深度和宽度
    "dropout_rate": 0.1,
    "clip_ratio": 0.25,
    "entropy_coeff": 0.05,
    "ppo_epochs": 10,                    # 专家修复：重命名，明确其为Epochs
    "num_minibatches": 4,                # 专家修复：新增Mini-batch数量
}

# 🔧 新增：自适应熵调整配置
ADAPTIVE_ENTROPY_CONFIG = {
    "enabled": True,             # 是否启用
    "start_episode": 100,        # 从第几回合开始启用
    "patience": 50,              # 连续多少回合无改进则提升熵
    "boost_factor": 0.1,         # 每次提升熵的比例
}

# 学习率调度配置
LEARNING_RATE_CONFIG = {
    "initial_lr": 8e-5,                  # 方案三：微调初始学习率
    "end_lr": 1e-6,
    "decay_power": 0.8,
    "critic_lr_multiplier": 0.5,         # 专家修复：为Critic设置一个较低的学习率乘数，以稳定价值学习
}

# 系统资源配置
SYSTEM_CONFIG = {
    "num_parallel_workers": 4,           # 并行worker数量
    "tf_inter_op_threads": 4,            # TensorFlow inter-op线程数
    "tf_intra_op_threads": 8,            # TensorFlow intra-op线程数
}


# =============================================================================
# 10. 随机领域生成配置 (Random Domain Generation)
# =============================================================================

def generate_random_orders() -> List[Dict[str, Any]]:
    """
    生成随机订单配置，用于泛化能力训练
    每次调用都会返回一套全新的、随机的订单组合
    """
    import random
    
    config = TRAINING_FLOW_CONFIG["generalization_phase"]["random_orders_config"]
    product_types = list(PRODUCT_ROUTES.keys())
    
    # 随机决定订单数量
    num_orders = random.randint(config["min_orders"], config["max_orders"])
    
    generated_orders = []
    for i in range(num_orders):
        # 随机选择产品类型
        product = random.choice(product_types)
        
        # 随机订单数量
        quantity = random.randint(
            config["min_quantity_per_order"], 
            config["max_quantity_per_order"]
        )
        
        # 随机优先级（基于权重）
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

    completion_score = completed_parts / target_parts if target_parts > 0 else 0
    
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