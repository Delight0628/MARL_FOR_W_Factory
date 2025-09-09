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

# 基础订单模板 
BASE_ORDERS = [
    {"product": "黑胡桃木餐桌", "quantity": 6, "priority": 1, "due_date": 300},  # 🔧 V8修复：增加数量，缩短交期
    {"product": "橡木书柜", "quantity": 6, "priority": 2, "due_date": 400},      # 🔧 V8修复：增加数量，缩短交期
    {"product": "松木床架", "quantity": 6, "priority": 1, "due_date": 350},      # 🔧 V8修复：增加数量，缩短交期
    {"product": "樱桃木椅子", "quantity": 4, "priority": 3, "due_date": 280},    # 🔧 V8修复：增加数量，缩短交期
    {"product": "黑胡桃木餐桌", "quantity": 4, "priority": 2, "due_date": 450},  # 🔧 V8修复：增加数量，缩短交期
    {"product": "橡木书柜", "quantity": 6, "priority": 1, "due_date": 320},      # 🔧 V8修复：增加数量，缩短交期
    {"product": "松木床架", "quantity": 4, "priority": 2, "due_date": 380},      # 🔧 V8修复：增加数量，缩短交期
    {"product": "樱桃木椅子", "quantity": 6, "priority": 1, "due_date": 250},    # 🔧 V8修复：增加数量，缩短交期
]

# 辅助函数：计算总零件数
def _calculate_total_parts():
    return sum(order["quantity"] for order in BASE_ORDERS)

# =============================================================================
# 1. 基础仿真参数 (Basic Simulation Parameters)
# =============================================================================

# 仿真时间设置
# =============================================================================
# 2. 仿真时间配置 (Simulation Time)
# =============================================================================

# 🔧 V8修复：恢复合理的仿真时间，制造时间压力
SIMULATION_TIME = 600  # 🔧 恢复到600分钟 (10小时，制造适度时间压力)
TIME_UNIT = "minutes"  # 时间单位：分钟

# 🔧 V31 突破性优化：课程学习配置 - 专门解决60%完成率陷阱
CURRICULUM_CONFIG = {
    "enabled": True,                  # 启用课程学习，从简单到复杂
    "stages": [
        {"name": "基础入门", "orders_scale": 0.4, "time_scale": 1.6, "iterations": 30, "graduation_thresholds": 95},
        {"name": "能力提升", "orders_scale": 0.8, "time_scale": 1.2, "iterations": 50, "graduation_thresholds": 90},
        {"name": "完整挑战", "orders_scale": 1.0, "time_scale": 1.0, "iterations": 100, "graduation_thresholds": 85},
    ],
    
    # 🔧 V31 新增：毕业考试强化配置
    "graduation_config": {
        "exam_episodes": 5,           # 毕业考试回合数增加到5轮
        "stability_requirement": 2,   # 需要连续2次考试通过才能毕业
        "max_retries": 5,             
        "retry_extension": 10,        # 每次重考延长10轮训练
    }
}


# 队列设置

QUEUE_CAPACITY = _calculate_total_parts()


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
    "enabled": True,                   # 是否启用设备故障 - 静态训练阶段禁用
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
        {"station": "带锯机", "time": 8},      # 🔧 MAPPO后清理：移除setup_time，简化配置
        {"station": "五轴加工中心", "time": 20},  # 🔧 MAPPO后清理：移除setup_time，简化配置
        {"station": "砂光机", "time": 10},      # 🔧 MAPPO后清理：移除setup_time，简化配置
        {"station": "组装台", "time": 15},      # 🔧 MAPPO后清理：移除setup_time，简化配置
        {"station": "包装台", "time": 5},      # 🔧 MAPPO后清理：移除setup_time，简化配置
    ],
    "橡木书柜": [
        {"station": "带锯机", "time": 12},      # 🔧 MAPPO后清理：移除setup_time，简化配置
        {"station": "五轴加工中心", "time": 25},  # 🔧 MAPPO后清理：移除setup_time，简化配置
        {"station": "砂光机", "time": 15},      # 🔧 MAPPO后清理：移除setup_time，简化配置
        {"station": "组装台", "time": 20},      # 🔧 MAPPO后清理：移除setup_time，简化配置
        {"station": "包装台", "time": 8},      # 🔧 MAPPO后清理：移除setup_time，简化配置
    ],
    "松木床架": [
        {"station": "带锯机", "time": 10},      # 🔧 MAPPO后清理：移除setup_time，简化配置
        {"station": "砂光机", "time": 12},      # 🔧 MAPPO后清理：移除setup_time，简化配置
        {"station": "组装台", "time": 15},      # 🔧 MAPPO后清理：移除setup_time，简化配置
        {"station": "包装台", "time": 6},      # 🔧 MAPPO后清理：移除setup_time，简化配置
    ],
    "樱桃木椅子": [
        {"station": "带锯机", "time": 6},      # 🔧 MAPPO后清理：移除setup_time，简化配置
        {"station": "五轴加工中心", "time": 12},  # 🔧 MAPPO后清理：移除setup_time，简化配置
        {"station": "砂光机", "time": 8},      # 🔧 MAPPO后清理：移除setup_time，简化配置
        {"station": "组装台", "time": 10},      # 🔧 MAPPO后清理：移除setup_time，简化配置
        {"station": "包装台", "time": 4},      # 🔧 MAPPO后清理：移除setup_time，简化配置
    ],
}

# =============================================================================
# 4. 订单配置 (Order Configuration)
# =============================================================================

# 基础订单模板 - 🔧 V8修复：增加订单量，缩短交期，制造真正的挑战
BASE_ORDERS = [
    {"product": "黑胡桃木餐桌", "quantity": 6, "priority": 1, "due_date": 300},  # 🔧 V8修复：增加数量，缩短交期
    {"product": "橡木书柜", "quantity": 6, "priority": 2, "due_date": 400},      # 🔧 V8修复：增加数量，缩短交期
    {"product": "松木床架", "quantity": 6, "priority": 1, "due_date": 350},      # 🔧 V8修复：增加数量，缩短交期
    {"product": "樱桃木椅子", "quantity": 4, "priority": 3, "due_date": 280},    # 🔧 V8修复：增加数量，缩短交期
    {"product": "黑胡桃木餐桌", "quantity": 4, "priority": 2, "due_date": 450},  # 🔧 V8修复：增加数量，缩短交期
    {"product": "橡木书柜", "quantity": 6, "priority": 1, "due_date": 320},      # 🔧 V8修复：增加数量，缩短交期
    {"product": "松木床架", "quantity": 4, "priority": 2, "due_date": 380},      # 🔧 V8修复：增加数量，缩短交期
    {"product": "樱桃木椅子", "quantity": 6, "priority": 1, "due_date": 250},    # 🔧 V8修复：增加数量，缩短交期
]

# 辅助函数：计算总零件数
def _calculate_total_parts():
    return sum(order["quantity"] for order in BASE_ORDERS)

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

# 🔧 MAPPO后优化：简化观测配置，MAPPO的集中式Critic已提供全局视野
ENHANCED_OBS_CONFIG = {
    "enabled": True,                      # 保持启用，但简化配置
    "top_n_parts": 2,                     # 🔧 优化：减少到2个零件，降低复杂度
    "include_downstream_info": False,     # 🔧 优化：禁用下游信息，MAPPO全局状态已包含
    "time_feature_normalization": 100.0,  # 保持不变
}

# 🔧 MAPPO后优化：动作空间配置，与简化的观测空间保持一致
ACTION_CONFIG_ENHANCED = {
    "enabled": True,                      # 保持启用扩展动作空间
    # 🔧 优化：动作空间自动适应观测配置
    "action_space_size": ENHANCED_OBS_CONFIG["top_n_parts"] + 1,  # 现在是3个动作
    "action_names": ["IDLE"] + [f"PROCESS_PART_{i+1}" for i in range(ENHANCED_OBS_CONFIG["top_n_parts"])],
}


# =============================================================================
# 6. 奖励系统配置 (Reward System) - 🔧 重构版：简洁目标导向设计
# =============================================================================

# 🔧 奖励系统重构：从23个组件简化为5个核心组件
# 设计原则：简洁性、目标导向、可解释性
REWARD_CONFIG = {
    # === 核心奖励组件 (5个) ===
    
    # 1. 零件完成奖励 - 主要驱动力
    "part_completion_reward": 10.0,        # 每完成一个零件获得10分
    
    # 2. 订单完成奖励 - 协调激励  
    "order_completion_reward": 50.0,       # 每完成一个订单额外获得50分
    
    # 3. 延期惩罚 - 质量约束 (重构版)
    "continuous_lateness_penalty": -0.1,  # 持续惩罚：每个late的零件，每分钟扣0.1分
    "final_tardiness_penalty": -1.0,      # 终局惩罚：最终总延期时间，每分钟扣1分
    
    # 4. 闲置惩罚与工作激励 - 效率约束 (🔧 基于日志分析加强)
    "idle_penalty": -2.0,                  # 🔧 从-0.1加强到-2.0，严厉惩罚闲置
    "idle_penalty_threshold": 5,           # 🔧 从10步降到5步，更快触发惩罚
    "work_bonus": 0.5,                     # 🔧 新增：每步积极工作的基础奖励
    
    # 5. 终局完成率奖励/惩罚 - 全局目标
    "final_completion_bonus_per_percent": 2.0,  # 每完成1%额外获得2分 (100%完成可获200分)
    "final_incompletion_penalty_per_percent": -3.0,  # 每未完成1%扣3分
    
    # 🔧 核心改造：为100%完成率设置巨额“完工大奖”
    "final_all_parts_completion_bonus": 500.0, # 必须完成所有零件才能获得的大奖
}



# =============================================================================
# 8. 自定义PPO训练配置 (Custom PPO Training Configuration)
# =============================================================================

# 🔧 MAPPO后优化：适应MAPPO强大能力的自适应训练配置
ADAPTIVE_TRAINING_CONFIG = {
    "target_score": 0.70,                # 🎯 核心目标：综合评分达到0.70（难度增加后提升目标）
    "target_consistency": 8,             # 🎯 核心目标：连续8次达标
    "max_episodes": 800,                 # 🔧 MAPPO优化：降低最大轮数，避免过度训练
    "early_stop_patience": 60,           # 🔧 适当延长耐心
    "performance_window": 10,            # 🔧 MAPPO优化：缩短性能窗口
}

# 🔧 V32 新增：PPO网络架构配置
PPO_NETWORK_CONFIG = {
    "hidden_sizes": [1024, 512],         # 🔧 神经网络隐藏层大小
    "dropout_rate": 0.1,                 # 🔧 Dropout率防过拟合
    "clip_ratio": 0.4,                   # 🔧 PPO裁剪比例
    "entropy_coeff": 0.3,                # 🔧 熵系数，增强探索
    "num_policy_updates": 10,            # 🔧 每轮策略更新次数
}

# 🔧 V32 新增：学习率调度配置
LEARNING_RATE_CONFIG = {
    "initial_lr": 2e-4,                  # 🔧 初始学习率
    "end_lr": 1e-5,                      # 🔧 最终学习率
    "decay_power": 1.0,                  # 🔧 衰减指数（1.0=线性衰减）
}

# 🔧 V32 新增：系统资源配置
SYSTEM_CONFIG = {
    "num_parallel_workers": 4,           # 🔧 并行worker数量
    "tf_inter_op_threads": 4,            # 🔧 TensorFlow inter-op线程数
    "tf_intra_op_threads": 8,            # 🔧 TensorFlow intra-op线程数
}

# =============================================================================
# 9. 评估和基准测试配置 (Evaluation & Benchmark Configuration)
# =============================================================================

# =============================================================================
# 10. 辅助函数 (Utility Functions)
# =============================================================================

def get_total_parts_count() -> int:
    """🔧 新增：获取基础订单的总零件数"""
    return sum(order["quantity"] for order in BASE_ORDERS)

def get_route_for_product(product: str) -> List[Dict[str, Any]]:
    """获取指定产品的工艺路线"""
    return PRODUCT_ROUTES.get(product, [])

def calculate_product_total_time(product: str) -> float:
    """计算产品总加工时间"""
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

def analyze_task_feasibility() -> Dict[str, Any]:
    """🔧 V38 新增：分析任务的理论可行性"""
    print("\n" + "="*60)
    print("🔬 任务可行性分析")
    print("="*60)
    
    # 计算总零件数
    total_parts = sum(order['quantity'] for order in BASE_ORDERS)
    print(f"📊 总零件数: {total_parts}")
    
    # 计算各工作站负荷
    station_loads = {}
    for station_name in WORKSTATIONS.keys():
        total_load = 0
        for order in BASE_ORDERS:
            route = get_route_for_product(order["product"])
            for step in route:
                if step["station"] == station_name:
                    # 计算加工时间
                    total_time = step["time"]
                    total_load += total_time * order["quantity"]
        
        # 考虑设备数量
        equipment_count = WORKSTATIONS[station_name]["count"]
        effective_load = total_load / equipment_count
        station_loads[station_name] = {
            'total_time': total_load,
            'equipment_count': equipment_count,
            'effective_load': effective_load
        }
    
    # 找出瓶颈工作站
    bottleneck_station = max(station_loads.keys(), key=lambda x: station_loads[x]['effective_load'])
    bottleneck_time = station_loads[bottleneck_station]['effective_load']
    
    print(f"\n🔧 各工作站负荷分析:")
    sorted_stations = sorted(station_loads.items(), key=lambda item: item[1]['effective_load'], reverse=True)

    for station, load_info in sorted_stations:
        is_bottleneck = station == bottleneck_station
        mark = "🚨" if is_bottleneck else "  "
        print(f"{mark} {station:<25}: {load_info['effective_load']:.1f}分钟 (设备数: {load_info['equipment_count']})")
    
    print(f"\n🎯 理论瓶颈 (最短完工时间): {bottleneck_time:.1f}分钟")
    print(f"⏰ 仿真时间限制 (time_scale=1): {SIMULATION_TIME}分钟")
    
    # 实际的仿真终止时间通常是 SIMULATION_TIME * time_scale
    # 在我们的环境中，time_scale 通常是 2.0
    actual_timeout = SIMULATION_TIME * 2.0 
    print(f"⏱️  实际仿真终止时间 (time_scale=2): {actual_timeout}分钟")
    
    # 可行性判断
    is_feasible = bottleneck_time < actual_timeout
    challenge_ratio = bottleneck_time / SIMULATION_TIME
    
    print(f"\n[结论]")
    if is_feasible:
        print(f"✅ 理论上可行: 最短完工时间 ({bottleneck_time:.1f}分钟) < 实际终止时间 ({actual_timeout}分钟)")
    else:
        print(f"❌ 理论上不可行: 最短完工时间 ({bottleneck_time:.1f}分钟) > 实际终止时间 ({actual_timeout}分钟)")
        print("   🔥 这是一个不可能完成的任务！智能体的任何策略都无法在规定时间内完成。")

    print(f"📈 任务挑战度: {challenge_ratio * 100:.1f}% (最短完工时间 / 标准仿真时间)")
    
    if challenge_ratio < 0.6:
        print("   - 建议: 任务过于简单，智能体可能无法学到复杂调度，可以增加订单量或缩短交期。")
    elif challenge_ratio > 0.95: # 调整阈值
        print("   - 警告: 任务挑战度极高，非常接近理论极限，智能体需要极优策略才能完成。")
    else:
        print("   - 评估: 任务挑战度适中，适合强化学习训练。")
    
    print("="*60 + "\n")

    return {
        'total_parts': total_parts,
        'bottleneck_station': bottleneck_station,
        'bottleneck_time': bottleneck_time,
        'simulation_time': SIMULATION_TIME,
        'is_feasible': is_feasible,
        'challenge_ratio': challenge_ratio,
        'station_loads': station_loads
    }

# 在模块加载时验证配置
if __name__ == "__main__":
    validate_config()
    analyze_task_feasibility() # 增加调用
    print(f"工作站数量: {len(WORKSTATIONS)}")
    print(f"产品种类: {len(PRODUCT_ROUTES)}")
    print(f"基础订单数: {len(BASE_ORDERS)}") 