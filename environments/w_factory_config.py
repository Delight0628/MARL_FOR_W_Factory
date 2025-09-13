"""
W工厂生产调度系统配置文件
这是项目的唯一真理来源 (Single Source of Truth)
包含所有工厂参数、设备信息、产品工艺路线和订单数据
"""

import numpy as np
from typing import Dict, List, Tuple, Any

# =============================================================================
# 1. 基础仿真参数 (Basic Simulation Parameters)
# =============================================================================
SIMULATION_TIME = 600  # 10小时
TIME_UNIT = "minutes"  # 时间单位：分钟

# 课程学习配置
CURRICULUM_CONFIG = {
    "enabled": False, # 启用课程学习，从简单到复杂
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

# 简化观测配置，MAPPO的集中式Critic已提供全局视野
ENHANCED_OBS_CONFIG = {
    "enabled": True,                      # 保持启用，但简化配置
    "top_n_parts": 2,                     # 减少到2个零件，降低复杂度
    "include_downstream_info": False,     # 禁用下游信息，MAPPO全局状态已包含
    "time_feature_normalization": 100.0,  # 保持不变
}

# 动作空间配置，与简化的观测空间保持一致
ACTION_CONFIG_ENHANCED = {
    "enabled": True,                      # 保持启用扩展动作空间
    # 动作空间自动适应观测配置
    "action_space_size": ENHANCED_OBS_CONFIG["top_n_parts"] + 1,  # 现在是3个动作
    "action_names": ["IDLE"] + [f"PROCESS_PART_{i+1}" for i in range(ENHANCED_OBS_CONFIG["top_n_parts"])],
}


# =============================================================================
# 6. 奖励系统配置 (Reward System) - 简洁目标导向设计
# =============================================================================

REWARD_CONFIG = {
    # === 核心奖励组件 (6个) ===
    
    # 1. 零件完成奖励 - 主要驱动力
    "part_completion_reward": 20.0,        # 增加到20分，强化主要激励
    
    # 2. 订单完成奖励 - 协调激励  
    "order_completion_reward": 100.0,      # 增加到100分，强化订单完成
    
    # 3. 延期惩罚 - 质量约束 (重构版)
    "continuous_lateness_penalty": -0.2,   # 持续惩罚：加强延期压力
    "final_tardiness_penalty": -0.5,       # 终局惩罚：适度降低
    
    # 4. 闲置惩罚与工作激励 - 效率约束
    "idle_penalty": -1.0,                  # 适度的闲置惩罚
    "idle_penalty_threshold": 5,           # 触发阈值
    "work_bonus": 1.0,                     # 增加工作奖励
    
    # 5. 终局完成率奖励/惩罚 - 全局目标
    "final_completion_bonus_per_percent": 5.0,    # 每完成1%获得5分，强化激励
    "final_incompletion_penalty_per_percent": -5.0,  # 每未完成1%扣5分
    
    # 6. 为100%完成率设置巨额"完工大奖"
    "final_all_parts_completion_bonus": 1000.0, # 增加到1000分的超级大奖
}



# =============================================================================
# 8. 自定义MAPPO训练配置 (Custom PPO Training Configuration)
# =============================================================================

# 自适应训练配置
ADAPTIVE_TRAINING_CONFIG = {
    "target_score": 0.68,                # 合理的目标分数
    "target_consistency": 6,             # 合理的一致性要求
    "max_episodes": 1000,                # 充分的训练轮数
    "early_stop_patience": 100,          # 更长的耐心，防止过早停止
    "performance_window": 15,            # 适中的性能窗口
}

# PPO网络架构配置
PPO_NETWORK_CONFIG = {
    "hidden_sizes": [768, 384],          # 适中的网络规模，平衡能力和泛化
    "dropout_rate": 0.15,                # 适度的Dropout
    "clip_ratio": 0.3,                   # 适中的裁剪比例
    "entropy_coeff": 0.1,                # 恢复适中的熵系数，保持探索能力
    "num_policy_updates": 8,             # 适度的更新次数
}

# 学习率调度配置
LEARNING_RATE_CONFIG = {
    "initial_lr": 3e-4,                  # 恢复合理的初始学习率
    "end_lr": 1e-5,                      # 合理的最终学习率
    "decay_power": 0.8,                  # 更平滑的衰减曲线
}

# 系统资源配置
SYSTEM_CONFIG = {
    "num_parallel_workers": 4,           # 并行worker数量
    "tf_inter_op_threads": 4,            # TensorFlow inter-op线程数
    "tf_intra_op_threads": 8,            # TensorFlow intra-op线程数
}


def get_total_parts_count() -> int:
    """获取基础订单的总零件数"""
    return sum(order["quantity"] for order in BASE_ORDERS)

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