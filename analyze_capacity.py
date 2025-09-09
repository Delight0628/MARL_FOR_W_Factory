import pprint
from environments.w_factory_config import (
    BASE_ORDERS,
    PRODUCT_ROUTES,
    WORKSTATIONS,
    CURRICULUM_CONFIG # 修正：正确的配置变量名
)

def analyze_bottlenecks():
    """
    分析并打印出当前配置下的理论产能瓶颈。
    """
    print("="*50)
    print("🏭 生产环境理论产能分析 🏭")
    print("="*50)

    # 1. 计算每个工作站的总工作负载
    station_loads = {station: 0 for station in WORKSTATIONS.keys()}
    
    for order in BASE_ORDERS:
        product_type = order["product"]
        quantity = order["quantity"]
        route = PRODUCT_ROUTES.get(product_type, [])
        
        for step in route:
            station = step["station"]
            processing_time = step["time"] # 修正: "time" 是正确的键
            if station in station_loads:
                station_loads[station] += quantity * processing_time

    print("\n[1] 各工作站总工作负载 (分钟):")
    pprint.pprint(station_loads)

    # 2. 计算最大可用时间窗口
    # 我们假设最宽松的情况，即所有任务必须在最晚的交货期内完成
    # 订单生成器会在 [min, max] 范围内随机选择一个due_date
    # 我们用max_due_date作为理论上的最大可用时间
    # 在课程学习配置中，time_scale会影响due_date，但基础范围在BASE_ORDERS的due_date中定义
    # 为了简化理论分析，我们直接取"完整挑战"阶段的配置来估算
    # 注意：实际due_date是 arrival_time + (base_due_date * time_scale)
    # 这里我们做一个近似，直接使用基础订单里的最大due_date作为时间窗口
    
    # 查找基础订单中的最晚交货时间作为基准
    max_base_due_date = 0
    for order in BASE_ORDERS:
        if order['due_date'] > max_base_due_date:
            max_base_due_date = order['due_date']
            
    # "完整挑战" 阶段的时间缩放
    final_stage_time_scale = CURRICULUM_CONFIG['stages'][-1]['time_scale']
    max_due_date = max_base_due_date * final_stage_time_scale

    print(f"\n[2] 理论最大可用时间窗口: {max_due_date:.1f} 分钟 (基于'完整挑战'阶段的最晚交期估算)")

    # 3. 计算每个工作站的总可用产能
    station_capacity = {
        station: config["count"] * max_due_date # 修正: 使用 config["count"]
        for station, config in WORKSTATIONS.items()
    }

    print("\n[3] 各工作站理论最大总产能 (分钟):")
    pprint.pprint(station_capacity)
    
    # 4. 对比负载与产能，识别瓶颈
    print("\n[4] 产能瓶颈分析 (负载 vs 产能):")
    bottlenecks = {}
    for station in station_loads:
        load = station_loads[station]
        capacity = station_capacity[station]
        utilization_ratio = (load / capacity) * 100 if capacity > 0 else 0
        
        print(f"  - 工作站: {station}")
        print(f"    - 负载: {load:.1f} 分钟")
        print(f"    - 产能: {capacity:.1f} 分钟")
        print(f"    - 理论负载率: {utilization_ratio:.2f}%")
        
        if load > capacity:
            bottlenecks[station] = utilization_ratio
            print("    - 结论: ⚠️ 严重瓶颈！理论上无法在时限内完成所有工作。")
        else:
            print("    - 结论: ✅ 产能够用。")

    print("="*50)
    if bottlenecks:
        print("\n🔥 最终诊断: 存在以下产能瓶颈，导致零延期100%完工在理论上不可能：")
        for station, ratio in bottlenecks.items():
            print(f"  - {station} (负载率: {ratio:.2f}%)")
    else:
        print("\n✅ 最终诊断: 恭喜！当前配置在理论上不存在产能瓶颈。")
    print("="*50)


if __name__ == "__main__":
    analyze_bottlenecks()
