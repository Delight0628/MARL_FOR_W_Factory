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
SIMULATION_TIME = 480  # 仿真总时长（分钟），8小时工作日
WARMUP_TIME = 0       # 预热时间（分钟）- 暂时不考虑预热

# 队列和容量设置
QUEUE_CAPACITY = 20   # 每个设备前队列的最大容量
MAX_ORDERS = 50       # 最大订单数量

# 随机种子（用于可重复实验）
RANDOM_SEED = 42

# =============================================================================
# 2. 工作站/设备配置 (Workstation/Equipment Configuration)
# =============================================================================

# 工作站定义：每个工作站类型及其设备数量
WORKSTATIONS = {
    "带锯机": {"count": 2, "oee": 0.85},      # 2台带锯机，OEE=85%
    "五轴加工中心": {"count": 1, "oee": 0.90},  # 1台五轴加工中心，OEE=90%
    "砂光机": {"count": 3, "oee": 0.80},      # 3台砂光机，OEE=80%
    "封边机": {"count": 2, "oee": 0.88},      # 2台封边机，OEE=88%
    "钻孔机": {"count": 2, "oee": 0.82},      # 2台钻孔机，OEE=82%
    "组装台": {"count": 4, "oee": 0.95},      # 4个组装台，OEE=95%
    "包装台": {"count": 2, "oee": 0.98},      # 2个包装台，OEE=98%
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
        {"station": "带锯机", "time": 8, "setup_time": 2},
        {"station": "五轴加工中心", "time": 20, "setup_time": 3},
        {"station": "砂光机", "time": 10, "setup_time": 2},
        {"station": "组装台", "time": 15, "setup_time": 2},
        {"station": "包装台", "time": 5, "setup_time": 1},
    ],
    "橡木书柜": [
        {"station": "带锯机", "time": 12, "setup_time": 2},
        {"station": "五轴加工中心", "time": 25, "setup_time": 4},
        {"station": "砂光机", "time": 15, "setup_time": 2},
        {"station": "组装台", "time": 20, "setup_time": 3},
        {"station": "包装台", "time": 8, "setup_time": 1},
    ],
    "松木床架": [
        {"station": "带锯机", "time": 10, "setup_time": 2},
        {"station": "砂光机", "time": 12, "setup_time": 2},
        {"station": "组装台", "time": 15, "setup_time": 3},
        {"station": "包装台", "time": 6, "setup_time": 1},
    ],
    "樱桃木椅子": [
        {"station": "带锯机", "time": 6, "setup_time": 1},
        {"station": "五轴加工中心", "time": 12, "setup_time": 2},
        {"station": "砂光机", "time": 8, "setup_time": 1},
        {"station": "组装台", "time": 10, "setup_time": 2},
        {"station": "包装台", "time": 4, "setup_time": 1},
    ],
}

# =============================================================================
# 4. 订单配置 (Order Configuration)
# =============================================================================

# 基础订单模板
BASE_ORDERS = [
    {"product": "黑胡桃木餐桌", "quantity": 3, "priority": 1, "due_date": 300},
    {"product": "橡木书柜", "quantity": 2, "priority": 2, "due_date": 400},
    {"product": "松木床架", "quantity": 4, "priority": 1, "due_date": 350},
    {"product": "樱桃木椅子", "quantity": 8, "priority": 3, "due_date": 280},
    {"product": "黑胡桃木餐桌", "quantity": 2, "priority": 2, "due_date": 450},
    {"product": "橡木书柜", "quantity": 1, "priority": 1, "due_date": 320},
    {"product": "松木床架", "quantity": 3, "priority": 2, "due_date": 380},
    {"product": "樱桃木椅子", "quantity": 6, "priority": 1, "due_date": 250},
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

# 奖励函数配置
REWARD_CONFIG = {
    "completion_reward": 100.0,         # 完成一个零件的奖励（增加）
    "tardiness_penalty": -10.0,         # 延期惩罚（减少）
    "idle_penalty": -0.01,              # 空闲惩罚（减少）
    "utilization_bonus": 5.0,           # 设备利用率奖励
    "makespan_penalty_weight": 0.001,   # 最大完工时间惩罚权重（减少）
}

# =============================================================================
# 6. 训练参数配置 (Training Parameters)
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
    
    print("配置文件验证通过！")
    return True

# 在模块加载时验证配置
if __name__ == "__main__":
    validate_config()
    print(f"工作站数量: {len(WORKSTATIONS)}")
    print(f"产品种类: {len(PRODUCT_ROUTES)}")
    print(f"基础订单数: {len(BASE_ORDERS)}")
    print(f"设备总数: {get_total_equipment_count()}") 