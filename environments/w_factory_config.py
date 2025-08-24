"""
W工厂生产调度系统配置文件
这是项目的唯一真理来源 (Single Source of Truth)
包含所有工厂参数、设备信息、产品工艺路线和订单数据

当前配置：静态训练模式
- 禁用设备故障 (EQUIPMENT_FAILURE["enabled"] = False)
- 禁用紧急插单 (EMERGENCY_ORDERS["enabled"] = False)
- 取消预热时间 (WARMUP_TIME = 0)
- 使用TensorFlow框架 (framework = "tf2")
"""

import numpy as np
from typing import Dict, List, Tuple, Any

# =============================================================================
# 1. 基础仿真参数 (Basic Simulation Parameters)
# =============================================================================

# 仿真时间设置
# =============================================================================
# 2. 仿真时间配置 (Simulation Time) - 🔧 V5紧迫感修复版
# =============================================================================

# 🔧 V8修复：恢复合理的仿真时间，制造时间压力
SIMULATION_TIME = 600  # 🔧 恢复到600分钟 (10小时，制造适度时间压力)
TIME_UNIT = "minutes"  # 时间单位：分钟

# 🔧 V8新增：环境终止条件配置说明
TERMINATION_CONFIG = {
    "max_time_multiplier": 2.0,      # 🔧 V8修复：最大时间 = SIMULATION_TIME * 2.0 = 1200分钟
    "priority": "task_completion",    # 优先级：任务完成 > 时间限制
    "early_termination": True,        # 所有订单完成时立即终止
}

# 🔧 V21优化：课程学习配置 - 更渐进的难度递增
CURRICULUM_CONFIG = {
    "enabled": True,                  # 启用课程学习，从简单到复杂
    "stages": [
        {"name": "效率入门", "orders_scale": 0.2, "time_scale": 1.8, "iterations": 50},  # 6个零件，1.8倍时间
        {"name": "效率基础", "orders_scale": 0.3, "time_scale": 1.6, "iterations": 40},  # 9个零件，1.6倍时间
        {"name": "效率强化", "orders_scale": 0.5, "time_scale": 1.4, "iterations": 30},
        {"name": "中级挑战", "orders_scale": 0.7, "time_scale": 1.2, "iterations": 25},
        {"name": "高级训练", "orders_scale": 0.85, "time_scale": 1.1, "iterations": 20}, # 🔧 V23：平滑过渡
        {"name": "完整挑战", "orders_scale": 1.0, "time_scale": 1.0, "iterations": 15},
    ]
}

# 🔧 修复：移除过度复杂的时间压力配置
# TIME_PRESSURE_CONFIG = {
#     "target_completion_time": 400,
#     "warning_time": 360,
#     "critical_time": 420,
#     "overtime_penalty": -5.0,
# }

WARMUP_TIME = 0       # 预热时间（分钟）- 暂时不考虑预热

# 队列和容量设置
QUEUE_CAPACITY = 20   # 每个设备前队列的最大容量
MAX_ORDERS = 20       # 🔧 从50减少到20，减少任务量

# 随机种子（用于可重复实验）
RANDOM_SEED = 42

# =============================================================================
# 2. 工作站/设备配置 (Workstation/Equipment Configuration)
# =============================================================================

# 工作站配置：设备数量和处理能力 - 🔧 V8修复：制造瓶颈
WORKSTATIONS = {
    "带锯机": {"count": 2, "capacity": 1},        # 2台设备，并行处理
    "五轴加工中心": {"count": 1, "capacity": 1},   # 🔧 V8修复：恢复到1台，制造关键瓶颈
    "砂光机": {"count": 1, "capacity": 1},        # 🔧 V8修复：减少到1台，增加挑战
    "组装台": {"count": 2, "capacity": 1},        # 2台设备，并行处理
    "包装台": {"count": 1, "capacity": 1},        # 🔧 V8修复：减少到1台，制造最终瓶颈
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

# 产品工艺路线：每个产品的加工步骤和时间 - 🔧 V8修复：恢复现实加工时间
PRODUCT_ROUTES = {
    "黑胡桃木餐桌": [
        {"station": "带锯机", "time": 8, "setup_time": 1},      # 🔧 V8修复：恢复到8分钟
        {"station": "五轴加工中心", "time": 20, "setup_time": 1},  # 🔧 V8修复：恢复到20分钟
        {"station": "砂光机", "time": 10, "setup_time": 1},      # 🔧 V8修复：恢复到10分钟
        {"station": "组装台", "time": 15, "setup_time": 1},      # 🔧 V8修复：恢复到15分钟
        {"station": "包装台", "time": 5, "setup_time": 1},      # 🔧 V8修复：恢复到5分钟
    ],
    "橡木书柜": [
        {"station": "带锯机", "time": 12, "setup_time": 1},      # 🔧 V8修复：恢复到12分钟
        {"station": "五轴加工中心", "time": 25, "setup_time": 1},  # 🔧 V8修复：恢复到25分钟
        {"station": "砂光机", "time": 15, "setup_time": 1},      # 🔧 V8修复：恢复到15分钟
        {"station": "组装台", "time": 20, "setup_time": 1},      # 🔧 V8修复：恢复到20分钟
        {"station": "包装台", "time": 8, "setup_time": 1},      # 🔧 V8修复：恢复到8分钟
    ],
    "松木床架": [
        {"station": "带锯机", "time": 10, "setup_time": 1},      # 🔧 V8修复：恢复到10分钟
        {"station": "砂光机", "time": 12, "setup_time": 1},      # 🔧 V8修复：恢复到12分钟
        {"station": "组装台", "time": 15, "setup_time": 1},      # 🔧 V8修复：恢复到15分钟
        {"station": "包装台", "time": 6, "setup_time": 1},      # 🔧 V8修复：恢复到6分钟
    ],
    "樱桃木椅子": [
        {"station": "带锯机", "time": 6, "setup_time": 1},      # 🔧 V8修复：恢复到6分钟
        {"station": "五轴加工中心", "time": 12, "setup_time": 1},  # 🔧 V8修复：恢复到12分钟
        {"station": "砂光机", "time": 8, "setup_time": 1},      # 🔧 V8修复：恢复到8分钟
        {"station": "组装台", "time": 10, "setup_time": 1},      # 🔧 V8修复：恢复到10分钟
        {"station": "包装台", "time": 4, "setup_time": 1},      # 🔧 V8修复：恢复到4分钟
    ],
}

# =============================================================================
# 4. 订单配置 (Order Configuration)
# =============================================================================

# 基础订单模板 - 🔧 V8修复：增加订单量，缩短交期，制造真正的挑战
BASE_ORDERS = [
    {"product": "黑胡桃木餐桌", "quantity": 4, "priority": 1, "due_date": 300},  # 🔧 V8修复：增加数量，缩短交期
    {"product": "橡木书柜", "quantity": 3, "priority": 2, "due_date": 400},      # 🔧 V8修复：增加数量，缩短交期
    {"product": "松木床架", "quantity": 5, "priority": 1, "due_date": 350},      # 🔧 V8修复：增加数量，缩短交期
    {"product": "樱桃木椅子", "quantity": 8, "priority": 3, "due_date": 280},    # 🔧 V8修复：增加数量，缩短交期
    {"product": "黑胡桃木餐桌", "quantity": 2, "priority": 2, "due_date": 450},  # 🔧 V8修复：增加数量，缩短交期
    {"product": "橡木书柜", "quantity": 2, "priority": 1, "due_date": 320},      # 🔧 V8修复：增加数量，缩短交期
    {"product": "松木床架", "quantity": 3, "priority": 2, "due_date": 380},      # 🔧 V8修复：增加数量，缩短交期
    {"product": "樱桃木椅子", "quantity": 6, "priority": 1, "due_date": 250},    # 🔧 V8修复：增加数量，缩短交期
]

# 紧急插单配置
EMERGENCY_ORDERS = {
    "enabled": False,                   # 是否启用紧急插单 - 静态训练阶段禁用
    "arrival_rate": 0.1,               # 每小时紧急订单到达率
    "priority_boost": 0,               # 紧急订单优先级提升
    "due_date_reduction": 0.7,         # 交期缩短比例
}

# =============================================================================
# 5. 强化学习环境参数 (RL Environment Parameters)
# =============================================================================

# 状态空间配置
STATE_CONFIG = {
    "queue_normalization": QUEUE_CAPACITY,  # 队列长度归一化基数
    "time_normalization": SIMULATION_TIME,  # 时间归一化基数
    "include_global_info": True,            # 是否包含全局信息
}

# 动作空间配置
ACTION_CONFIG = {
    "action_space_size": 2,             # 动作空间大小：0=IDLE, 1=PROCESS
    "action_names": ["IDLE", "PROCESS"], # 动作名称
}

# 🔧 V7 新增：增强观测空间和动作空间配置
ENHANCED_OBS_CONFIG = {
    "enabled": True,                      # 是否启用增强观测
    "top_n_parts": 3,                     # 观测队列中前N个零件的信息
    "include_downstream_info": True,      # 是否包含下游工作站信息
    "time_feature_normalization": 100.0,  # 时间相关特征的归一化基数
}

# 🔧 V7 扩展：动作空间配置
ACTION_CONFIG_ENHANCED = {
    "enabled": True,                      # 是否启用扩展动作空间
    # 动作空间将变为 N+1 (0=IDLE, 1=处理第1个零件, 2=处理第2个, ...)
    "action_space_size": ENHANCED_OBS_CONFIG["top_n_parts"] + 1,
    "action_names": ["IDLE"] + [f"PROCESS_PART_{i+1}" for i in range(ENHANCED_OBS_CONFIG["top_n_parts"])],
}


# =============================================================================
# 6. 奖励系统配置 (Reward System) - 🔧 V4 平衡修复版
# =============================================================================

# 🔧 V16 奖励系统深度重构：解决"伪收敛"和奖励鸿沟问题
REWARD_CONFIG = {
    # 🔧 V16核心修复：缩小奖励鸿沟，增强过程引导
    "base_reward": 0.0,                    # 保持无基础奖励
    
    # 🔧 V17进一步优化：基于专家建议进一步增强过程奖励
    "order_completion_reward": 500.0,      # 🔧 V23: 恢复高奖励，提供强信号
    "part_completion_reward": 5.0,         # 🔧 V23: 恢复并增强零件奖励
    "step_reward": -0.01,                  # 🔧 V23: 引入轻微的时间惩罚，鼓励效率
    
    # 🔧 V12数值优化：里程碑奖励机制
    "order_progress_bonus": 20.0,          # 🔧 V23: 大幅增强进度奖励，搭建桥梁
    "critical_path_bonus": 0.5,            # 🔧 V23: 适度恢复
    "bottleneck_priority_bonus": 0.2,      # 🔧 V23: 适度恢复
    
    # 🔧 V12数值优化：效率奖励
    "order_efficiency_bonus": 10.0,        # 🔧 V23: 增强效率奖励
    "balanced_utilization_bonus": 0.5,     # 🔧 V12：从50降到0.5
    
    # 🔧 V12数值优化：全局协调奖励
    "coordination_reward": 0.0,            # 🔧 V15 关键修复：禁用协调奖励（可能在无生产时发放）
    "flow_optimization_bonus": 0.0,        # 🔧 V15 关键修复：禁用流程优化奖励
    
    # 🔧 V16调整：惩罚机制与新奖励体系匹配
    "order_tardiness_penalty": -2.0,       # 🔧 V12：从-200降到-2，延期仍有惩罚
    "order_abandonment_penalty": -10.0,    # 🔧 V12：从-1000降到-10，遗弃订单仍有损失
    "order_abandonment_threshold": 300,    # 保持300分钟的检测阈值
    "incomplete_order_final_penalty": -20.0,  # 🔧 V12：从-2000降到-20，仍然严厉但数值合理
    "resource_waste_penalty": -0.05,       # 🔧 V12：从-5降到-0.05
    
    # 🔧 V12数值优化：精细化控制参数
    "idle_penalty": -0.01,                 # 🔧 V15：加大闲置惩罚，防止什么都不做
    "idle_penalty_threshold": 30,          # 保持30步的阈值
    "tardiness_penalty_per_agent": False,
    
    # 🔧 V12数值优化：系数调整
    "reward_scale_factor": 1.0,            # 🔧 V23：恢复到1.0，因为基础奖励已足够大
    "penalty_scale_factor": 1.0,           # 🔧 V12：从0.1恢复到1.0，不再特意缩小惩罚
    
    # 🔧 修复后：奖励数值说明
    # 现在的奖励范围大致在-20到+50之间，没有额外的10倍缩放，数值更加合理
    
    # 🔧 V16新增：塑形奖励（Reward Shaping）
    "shaping_enabled": True,                # 启用塑形奖励
    "same_order_bonus": 0.3,               # 连续完成同一订单零件的奖励
    "urgent_order_bonus": 0.5,             # 处理紧急订单的额外奖励
    "flow_smoothness_bonus": 0.2,          # 保持生产线流畅的奖励
    "queue_balance_bonus": 0.1,            # 队列均衡奖励
    "early_completion_bonus": 1.0,         # 提前完成订单的奖励
}

# 新增：设备利用率统计配置（文档化口径，不影响ray逻辑）
UTILIZATION_CONFIG = {
    # method: "busy_machine_time" 表示使用“忙碌机器-时间面积 / (总时间 * 设备数量)”的平均利用率口径
    # 该口径已在 environments/w_factory_env.py 中实现并默认使用
    "method": "busy_machine_time"
}

# 网络架构配置
MODEL_CONFIG = {
    "fcnet_hiddens": [256, 256],        # 全连接层隐藏单元
    "fcnet_activation": "relu",         # 激活函数
    "use_lstm": False,                  # 是否使用LSTM
    "lstm_cell_size": 256,              # LSTM单元大小
}

# 训练停止条件
STOP_CONFIG = {
    "training_iteration": 1000,         # 最大训练迭代次数
    "timesteps_total": 1000000,         # 最大时间步数
    "episode_reward_mean": 500,         # 目标平均奖励
}

# =============================================================================
# 7. 训练参数配置 (Training Parameters)
# =============================================================================

# PPO/MAPPO算法参数
TRAINING_CONFIG = {
    "algorithm": "PPO",
    "framework": "torch",
    "num_workers": 4,                   # 并行工作进程数
    "num_envs_per_worker": 1,           # 每个工作进程的环境数
    "rollout_fragment_length": 200,     # 回滚片段长度
    "train_batch_size": 4000,           # 训练批次大小
    "sgd_minibatch_size": 128,          # SGD小批次大小
    "num_sgd_iter": 10,                 # SGD迭代次数
    "lr": 3e-4,                         # 学习率
    "gamma": 0.99,                      # 折扣因子
    "lambda": 0.95,                     # GAE参数
    "clip_param": 0.2,                  # PPO裁剪参数
    "vf_clip_param": 10.0,              # 价值函数裁剪参数
    "entropy_coeff": 0.01,              # 熵系数
    "vf_loss_coeff": 0.5,               # 价值函数损失系数
}

# =============================================================================
# 7. 评估和基准测试配置 (Evaluation & Benchmark Configuration)
# =============================================================================

# 评估参数
EVALUATION_CONFIG = {
    "evaluation_interval": 50,          # 评估间隔
    "evaluation_duration": 10,          # 评估持续轮数
    "evaluation_num_workers": 1,        # 评估工作进程数
    "evaluation_config": {
        "explore": False,               # 评估时不探索
        "render_env": False,            # 不渲染环境
    }
}

# 基准算法配置
BENCHMARK_CONFIG = {
    "algorithms": ["FIFO", "SPT", "EDD", "RANDOM"],  # 基准算法列表
    "num_runs": 10,                     # 每个算法运行次数
    "confidence_level": 0.95,           # 置信水平
}

# =============================================================================
# 8. 辅助函数 (Utility Functions)
# =============================================================================

def get_workstation_list() -> List[str]:
    """获取所有工作站名称列表"""
    return list(WORKSTATIONS.keys())

def get_product_list() -> List[str]:
    """获取所有产品名称列表"""
    return list(PRODUCT_ROUTES.keys())

def get_total_equipment_count() -> int:
    """获取设备总数"""
    return sum(ws["count"] for ws in WORKSTATIONS.values())

def get_route_for_product(product: str) -> List[Dict[str, Any]]:
    """获取指定产品的工艺路线"""
    return PRODUCT_ROUTES.get(product, [])

def calculate_product_total_time(product: str) -> float:
    """计算产品总加工时间（不包括setup时间）"""
    route = get_route_for_product(product)
    return sum(step["time"] for step in route)

def validate_config() -> bool:
    """验证配置文件的完整性和一致性 - 🔧 V8增强版"""
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
    
    # 🔧 V8新增：挑战性验证
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
    
    print("🔧 V8配置挑战性验证:")
    print(f"   总零件数: {total_parts}")
    print(f"   总加工时间: {total_processing_time:.1f}分钟")
    print(f"   理论最短完工时间: {theoretical_makespan:.1f}分钟")
    print(f"   仿真时间限制: {SIMULATION_TIME}分钟")
    
    if theoretical_makespan > SIMULATION_TIME * 0.8:
        print(f"   🎯 环境具有高挑战性 (理论完工时间占仿真时间{theoretical_makespan/SIMULATION_TIME*100:.1f}%)")
    elif theoretical_makespan > SIMULATION_TIME * 0.5:
        print(f"   ⚠️  环境具有中等挑战性 (理论完工时间占仿真时间{theoretical_makespan/SIMULATION_TIME*100:.1f}%)")
    else:
        print(f"   ❌ 环境挑战性不足 (理论完工时间仅占仿真时间{theoretical_makespan/SIMULATION_TIME*100:.1f}%)")
    
    # 检查瓶颈工作站
    bottleneck_station = max(bottleneck_time, key=bottleneck_time.get)
    print(f"   🔍 瓶颈工作站: {bottleneck_station} (负荷: {bottleneck_time[bottleneck_station]:.1f}分钟)")
    
    print("配置文件验证通过！")
    return True

# 在模块加载时验证配置
if __name__ == "__main__":
    validate_config()
    print(f"工作站数量: {len(WORKSTATIONS)}")
    print(f"产品种类: {len(PRODUCT_ROUTES)}")
    print(f"基础订单数: {len(BASE_ORDERS)}")
    print(f"设备总数: {get_total_equipment_count()}") 