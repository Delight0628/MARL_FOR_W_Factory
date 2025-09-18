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

# 课程学习配置
CURRICULUM_CONFIG = {
    "enabled": False, # 关键：禁用旧的课程学习，新的两阶段逻辑在ppo_marl_train.py中实现
    "stages": [
        {"name": "基础入门", "orders_scale": 0.4, "time_scale": 1.6, "iterations": 30, "graduation_thresholds": 95},
        {"name": "能力提升", "orders_scale": 0.8, "time_scale": 1.2, "iterations": 50, "graduation_thresholds": 90},
        {"name": "完整挑战", "orders_scale": 1.0, "time_scale": 1.0, "iterations": 100, "graduation_thresholds": 85},
    ],
    # 毕业考试配置
    "graduation_config": {
        "exam_episodes": 5,           # 毕业考试回合5轮
        "stability_requirement": 2,   # 需要连续2次考试通过才能毕业
        "max_retries": 5,             # 最大重考次数
        "retry_extension": 10,        # 每次重考延长10轮训练
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
QUEUE_CAPACITY = sum(order["quantity"] for order in BASE_ORDERS)

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

# 🔧 修复：增强观测配置，提供更多信息
ENHANCED_OBS_CONFIG = {
    "enabled": True,
    "top_n_parts": 3,                     # 🔧 恢复到3个零件，降低观察复杂性
    "include_downstream_info": True,      # 保持启用下游信息
    "time_feature_normalization": 100.0,
}

# 动作空间配置，与观测空间保持一致
ACTION_CONFIG_ENHANCED = {
    "enabled": True,
    # 动作空间自动适应观测配置
    "action_space_size": ENHANCED_OBS_CONFIG["top_n_parts"] + 1,  # 现在是6个动作（0=IDLE, 1-5=处理零件1-5）
    "action_names": ["IDLE"] + [f"PROCESS_PART_{i+1}" for i in range(ENHANCED_OBS_CONFIG["top_n_parts"])],
}


# =============================================================================
# 6. 奖励系统配置 (Reward System) - 简洁目标导向设计
# =============================================================================

REWARD_CONFIG = {
    # === 核心奖励组件 (简化，让位于时间奖励) ===
    "part_completion_reward": 10.0,
    "order_completion_reward": 50.0,
    
    # 3. 延期惩罚 - 质量约束
    "continuous_lateness_penalty": -1,
    "final_tardiness_penalty": -1,
    
    # 4. 闲置惩罚与工作激励 - 效率约束
    "idle_penalty": -2.0,
    "idle_penalty_threshold": 3,
    "work_bonus": 1.0, # 略微降低，因为主要奖励来自时间
    
    # 6. 完工大奖 (保持)
    "final_all_parts_completion_bonus": 500.0,
    
    # --- 方案一：全新的、与时间大小直接挂钩的奖励机制 ---
    "urgency_bonus_scaler": 5.0,           # 对选择最紧急零件的奖励进行缩放
    "slack_time_reward_multiplier": 0.01,  # 对选择的零件，其正的松弛时间（提前量）的奖励系数
    "lateness_penalty_multiplier": 0.1,   # 对选择的零件，其负的松弛时间（延期量）的惩罚系数
    
    # 禁用旧的、模糊的奖励
    # "urgent_part_bonus": 1.5,

    # 通用惩罚 (保持)
    "wip_penalty": -0.05, # Penalty per part in queue, per agent, per reward step
    "time_step_penalty": -0.01, # A small penalty for every time step to encourage speed
}



# =============================================================================
# 8. 自定义MAPPO训练配置 (Custom PPO Training Configuration)
# =============================================================================

# 自适应训练配置
ADAPTIVE_TRAINING_CONFIG = {
    "target_score": 0.72,                # 合理的目标分数
    "target_consistency": 6,             # 合理的一致性要求
    "max_episodes": 1000,                # 充分的训练轮数
    "early_stop_patience": 100,          # 更长的耐心，防止过早停止
    "performance_window": 15,            # 适中的性能窗口
    # 新增：基础训练毕业的延期硬性门槛 (分钟)
    "foundation_training_tardiness_threshold": 450.0,
}

# PPO网络架构配置
PPO_NETWORK_CONFIG = {
    "hidden_sizes": [1024, 512, 256],    # 🔧 关键：增加网络深度和宽度
    "dropout_rate": 0.1,
    "clip_ratio": 0.25,
    "entropy_coeff": 0.03,               
    "num_policy_updates": 16,            # 方案三：增加更新次数
}

# 学习率调度配置
LEARNING_RATE_CONFIG = {
    "initial_lr": 8e-5,                  # 方案三：微调初始学习率
    "end_lr": 1e-6,
    "decay_power": 0.8,
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

# 随机订单生成参数
RANDOM_ORDERS_CONFIG = {
    "min_orders": 5,           # 最少订单数
    "max_orders": 8,           # 最多订单数
    "min_quantity_per_order": 3,  # 每个订单最少零件数
    "max_quantity_per_order": 12, # 每个订单最多零件数
    "due_date_range": (200.0, 700.0),  # 交期范围
    "priority_weights": [0.3, 0.5, 0.2],  # 优先级1,2,3的概率权重
}

def generate_random_orders() -> List[Dict[str, Any]]:
    """
    生成随机订单配置，用于泛化能力训练
    每次调用都会返回一套全新的、随机的订单组合
    """
    import random
    
    config = RANDOM_ORDERS_CONFIG
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