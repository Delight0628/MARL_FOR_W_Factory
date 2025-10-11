import os
import sys

# 关键修复：强制评估脚本在CPU上运行，避免与训练进程争夺GPU资源
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 屏蔽TensorFlow的INFO级别日志

import numpy as np
import tensorflow as tf
import pandas as pd
from tqdm import tqdm
import argparse
import contextlib
import time # 导入time模块
import copy

from plotting import generate_gantt_chart

# 添加环境路径
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from environments.w_factory_env import WFactoryEnv
from environments.w_factory_config import (
    get_total_parts_count, SIMULATION_TIME, BASE_ORDERS,
    ACTION_CONFIG_ENHANCED, WORKSTATIONS, calculate_episode_score,
    QUEUE_VIEW_CONFIG
)

# =============================================================================
# 1. 核心配置 (Core Configuration)
# =============================================================================
NUM_EVAL_EPISODES = 30 

# 静态评估环境配置 (确保公平对比)
# 使用100%订单，标准时间，且禁用所有随机事件
STATIC_EVAL_CONFIG = {
    'orders_scale': 1.0,
    'time_scale': 1.0,
    'disable_failures': True, # 明确禁用设备故障
    'stage_name': '静态评估'
}

# =============================================================================
# 🌟 新增：泛化能力测试订单配置 (Generalization Test Configurations) 配置是否合理
# =============================================================================

# 测试配置1：高压力短交期场景
GENERALIZATION_CONFIG_1 = {
    'custom_orders': [
        # 紧急小批量订单 - 测试模型对时间压力的应对
        {"product": "黑胡桃木餐桌", "quantity": 8, "priority": 1, "due_date": 200.0},
        {"product": "橡木书柜", "quantity": 6, "priority": 1, "due_date": 180.0},
        {"product": "松木床架", "quantity": 10, "priority": 2, "due_date": 250.0},
        {"product": "樱桃木椅子", "quantity": 12, "priority": 1, "due_date": 300.0},
        {"product": "黑胡桃木餐桌", "quantity": 6, "priority": 3, "due_date": 400.0},
    ],
    'disable_failures': True,
    'stage_name': '泛化测试1-高压力短交期'
}

# 测试配置2：混合优先级复杂场景
GENERALIZATION_CONFIG_2 = {
    'custom_orders': [
        # 不同优先级和规模的混合订单 - 测试优先级平衡能力
        {"product": "橡木书柜", "quantity": 15, "priority": 2, "due_date": 450.0},
        {"product": "樱桃木椅子", "quantity": 8, "priority": 1, "due_date": 350.0},
        {"product": "黑胡桃木餐桌", "quantity": 20, "priority": 3, "due_date": 600.0},
        {"product": "松木床架", "quantity": 5, "priority": 1, "due_date": 280.0},
        {"product": "橡木书柜", "quantity": 12, "priority": 2, "due_date": 520.0},
    ],
    'disable_failures': True,
    'stage_name': '泛化测试2-混合优先级'
}

# 测试配置3：大批量长周期场景
GENERALIZATION_CONFIG_3 = {
    'custom_orders': [
        # 大批量长周期订单 - 测试资源调度和长期规划能力
        {"product": "黑胡桃木餐桌", "quantity": 25, "priority": 2, "due_date": 800.0},
        {"product": "松木床架", "quantity": 18, "priority": 1, "due_date": 700.0},
        {"product": "樱桃木椅子", "quantity": 22, "priority": 3, "due_date": 900.0},
        {"product": "橡木书柜", "quantity": 15, "priority": 2, "due_date": 750.0},
    ],
    'disable_failures': True,
    'stage_name': '泛化测试3-大批量长周期'
}

# =============================================================================
# 3. 环境创建与配置 (Environment Creation & Configuration)
# =============================================================================



# =============================================================================
# 4. 评估执行器 (Evaluation Runners)
# =============================================================================

def run_single_episode(env: WFactoryEnv, policy_fn, seed: int, config: dict = None):
    """运行单次回合的通用函数"""
    obs, info = env.reset(seed=seed)
    step_count = 0
    
    while step_count < 1500: # 与训练时保持一致的最大步数
        actions = policy_fn(obs, env)
        obs, rewards, terminations, truncations, info = env.step(actions)
        step_count += 1
        
        if any(terminations.values()) or any(truncations.values()):
            break
            
    final_stats = env.sim.get_final_stats()
    score = calculate_episode_score(final_stats, config)
    
    # 仅在第一个回合（seed=0）返回详细的加工历史
    history = env.sim.gantt_chart_history if seed == 0 else None
    
    return final_stats, score, history

def evaluate_marl_model(model_path: str, config: dict = STATIC_EVAL_CONFIG, generate_gantt: bool = False, output_dir: str = None, run_name: str = None, env_config_overrides: dict = None):
    """评估MARL模型"""
    config_name = config.get('stage_name', '未知配置')
    print(f"🧠 开始评估MARL模型: {model_path}", flush=True)
    print(f"📋 测试配置: {config_name}", flush=True)
    
    # 🔧 新增：显示自定义订单信息
    if 'custom_orders' in config:
        total_parts = sum(order["quantity"] for order in config['custom_orders'])
        print(f"📦 自定义订单: {len(config['custom_orders'])}个订单, 总计{total_parts}个零件", flush=True)
    
    if not os.path.exists(model_path):
        print(f"❌ 错误: 模型文件不存在 at {model_path}", flush=True)
        return None, None

    try:
        actor_model = tf.keras.models.load_model(model_path)
    except Exception as e:
        print(f"❌ 加载模型失败: {e}", flush=True)
        return None, None

    def marl_policy(obs, env):
        actions = {}
        for agent in env.agents:
            if agent in obs:
                state = tf.expand_dims(obs[agent], 0)
                action_probs = actor_model(state, training=False)
                # # 🔧 重要修复：评估时使用微软随机策略，避免完全卡死
                # # 根据概率分布采样，但主要选择高概率动作
                # if np.random.random() < 0.2:  # 20%概率使用概率采样
                #     action = tf.random.categorical(tf.math.log(action_probs + 1e-8), 1)[0, 0].numpy()
                # else:  # 80%概率使用确定性
                #     action = int(tf.argmax(action_probs[0]))
                # 评估时使用纯确定性策略（argmax）
                action = int(tf.argmax(action_probs[0]))
                actions[agent] = action
        return actions

    # 🔧 V4 修复：直接通过config传递自定义订单，无需上下文管理器
    all_kpis = []
    all_scores = []
    first_episode_history = None

    # 🔧 关键修复 V2: 合并来自优化器的基础配置和评估场景的特定配置
    final_config_for_eval = copy.deepcopy(env_config_overrides) if env_config_overrides else {}
    final_config_for_eval.update(config)

    env = WFactoryEnv(config=final_config_for_eval)
    
    # 动态选择迭代器：交互式终端使用tqdm，否则使用普通range
    is_tty = sys.stdout.isatty()
    iterator = range(NUM_EVAL_EPISODES)
    if is_tty:
        iterator = tqdm(iterator, desc=f"MARL模型评估({config_name})")

    start_time = time.time()
    for i in iterator:
        final_stats, score, history = run_single_episode(env, marl_policy, seed=i, config=config)
        all_kpis.append(final_stats)
        all_scores.append(score)
        if history is not None:
            first_episode_history = history
    
    if not is_tty:
        end_time = time.time()
        duration = end_time - start_time
        it_per_s = NUM_EVAL_EPISODES / duration if duration > 0 else float('inf')
        desc = f"MARL模型评估({config_name})"
        # 手动格式化输出，模拟tqdm的最终行
        print(f"{desc}: 100%|{'█'*10}| {NUM_EVAL_EPISODES}/{NUM_EVAL_EPISODES} [{duration:.2f}s, {it_per_s:.2f}it/s]", file=sys.stdout, flush=True)

    # 生成甘特图
    if generate_gantt and first_episode_history:
        generate_gantt_chart(first_episode_history, "MARL_PPO", config_name, output_dir=output_dir, run_name=run_name)

    env.close()
    
    return all_kpis, all_scores

def evaluate_heuristic(heuristic_name: str, config: dict = STATIC_EVAL_CONFIG, generate_gantt: bool = False, output_dir: str = None, run_name: str = None):
    """评估启发式算法"""
    config_name = config.get('stage_name', '未知配置')
    print(f"⚙️  开始评估启发式算法: {heuristic_name}", flush=True)
    print(f"📋 测试配置: {config_name}", flush=True)
    
    # 🔧 新增：显示自定义订单信息
    if 'custom_orders' in config:
        total_parts = sum(order["quantity"] for order in config['custom_orders'])
        print(f"📦 自定义订单: {len(config['custom_orders'])}个订单, 总计{total_parts}个零件", flush=True)

    def heuristic_policy(obs, env):
        """
        🔧 动态适配方案B：使用策略型动作而非索引型动作
        方案B动作空间：
          0: IDLE
          1: URGENT_EDD (最紧急)
          2: SHORT_SPT (最短加工)
          3: BALANCE (负载均衡)
          4: FIFO (先进先出)
          5: RANDOM (随机)
          6-15: CANDIDATE_1 ~ CANDIDATE_10
        """
        sim = env.sim
        actions = {}
        
        # 🔧 从配置中读取策略型动作的映射
        action_names = ACTION_CONFIG_ENHANCED.get("action_names", [])
        
        # 🔧 动态查找策略动作的ID
        action_map = {}
        for idx, name in enumerate(action_names):
            action_map[name] = idx
        
        for agent_id in env.agents:
            station_name = agent_id.replace("agent_", "")
            queue = sim.queues[station_name].items
            
            if not queue:
                actions[agent_id] = 0  # IDLE
                continue

            # # 🔧 关键修复：根据是否启用排序视图来选择不同的策略
            # if QUEUE_VIEW_CONFIG.get("enabled", False):
            #     # 使用排序视图：获取按紧急度排序的队列视图
            #     sorted_view = sim._get_sorted_queue_view(station_name)
                
            #     if not sorted_view:
            #         actions[agent_id] = 0 # IDLE
            #         continue
                
            #     # 在排序视图中根据启发式规则选择零件
            #     if heuristic_name == 'FIFO':
            #         # 先进先出：应选择物理队列的第一个（orig_index==0）
            #         # 在排序视图中查找orig_index==0对应的元素位置
            #         best_view_index = None
            #         for idx, item in enumerate(sorted_view):
            #             if item.get("orig_index", -1) == 0:
            #                 best_view_index = idx
            #                 break
            #         # 若物理队列第一个未出现在可见top-k中，退化为选择可见集合中orig_index最小者
            #         if best_view_index is None:
            #             best_view_index = int(np.argmin([item.get("orig_index", 1e9) for item in sorted_view]))
            #     elif heuristic_name == 'EDD':
            #         # 最早交期: 在排序视图中选择交期最小的
            #         best_view_index = np.argmin([item["part"].due_date for item in sorted_view])
            #     elif heuristic_name == 'SPT':
            #         # 最短处理时间: 在排序视图中选择当前工序处理时间最短的
            #         best_view_index = np.argmin([item["part"].get_processing_time() for item in sorted_view])
            #     else:
            #         raise ValueError(f"未知的启发式规则: {heuristic_name}")
                
            #     # 动作ID = 排序视图索引 + 1
            #     actions[agent_id] = best_view_index + 1
                
            # else:
            #     # 使用物理队列：原有的逻辑
            #     if heuristic_name == 'FIFO':
            #         # 先进先出: 直接选择队列头的第一个 (index 0)
            #         best_part_index = 0
            #     elif heuristic_name == 'EDD':
            #         # 最早交期: 选择交期最小的
            #         best_part_index = np.argmin([part.due_date for part in queue])
            #     elif heuristic_name == 'SPT':
            #         # 最短处理时间: 选择当前工序处理时间最短的
            #         best_part_index = np.argmin([part.get_processing_time() for part in queue])
            #     else:
            #         raise ValueError(f"未知的启发式规则: {heuristic_name}")

            #     # 动作ID = 零件索引 + 1
            #     actions[agent_id] = best_part_index + 1
            
            # 🔧 方案B：直接使用策略型动作
            if heuristic_name == 'FIFO':
                actions[agent_id] = action_map.get("FIFO", 4)  # 默认为4
            elif heuristic_name == 'EDD':
                actions[agent_id] = action_map.get("URGENT_EDD", 1)  # 默认为1
            elif heuristic_name == 'SPT':
                actions[agent_id] = action_map.get("SHORT_SPT", 2)  # 默认为2
            else:
                raise ValueError(f"未知的启发式规则: {heuristic_name}")
            
        return actions

    # 🔧 V4 修复：直接通过config传递自定义订单，无需上下文管理器
    all_kpis = []
    all_scores = []
    first_episode_history = None

    env = WFactoryEnv(config=config)
    
    # 动态选择迭代器：交互式终端使用tqdm，否则使用普通range
    is_tty = sys.stdout.isatty()
    iterator = range(NUM_EVAL_EPISODES)
    if is_tty:
        iterator = tqdm(iterator, desc=f"{heuristic_name}评估({config_name})")

    start_time = time.time()
    for i in iterator:
        final_stats, score, history = run_single_episode(env, heuristic_policy, seed=i, config=config)
        all_kpis.append(final_stats)
        all_scores.append(score)
        if history is not None:
            first_episode_history = history

    if not is_tty:
        end_time = time.time()
        duration = end_time - start_time
        it_per_s = NUM_EVAL_EPISODES / duration if duration > 0 else float('inf')
        desc = f"{heuristic_name}评估({config_name})"
        # 手动格式化输出，模拟tqdm的最终行
        print(f"{desc}: 100%|{'█'*10}| {NUM_EVAL_EPISODES}/{NUM_EVAL_EPISODES} [{duration:.2f}s, {it_per_s:.2f}it/s]", file=sys.stdout, flush=True)
    
    # 生成甘特图
    if generate_gantt and first_episode_history:
        generate_gantt_chart(first_episode_history, heuristic_name, config_name, output_dir=output_dir, run_name=run_name)
        
    env.close()
    return all_kpis, all_scores

# =============================================================================
# 5. 结果汇总与展示 (Result Aggregation & Display)
# =============================================================================

def aggregate_results(method_name: str, all_kpis: list, all_scores: list, config: dict = None):
    """汇总多次运行的结果，计算均值和标准差"""
    if all_kpis is None:
        return {
            "Method": method_name,
            "Avg Score": "N/A",
            "Std Score": "N/A",
            "Avg Completion %": "N/A",
            "Avg Makespan": "N/A",
            "Avg Tardiness": "N/A",
            "Avg Utilization %": "N/A",
        }

    # 🌟 新增：根据配置确定目标零件数
    if config and 'custom_orders' in config:
        target_parts = sum(order["quantity"] for order in config['custom_orders'])
    else:
        target_parts = get_total_parts_count()
        
    completion_rates = [(k['total_parts'] / target_parts) * 100 for k in all_kpis]
    
    return {
        "Method": method_name,
        "Avg Score": f"{np.mean(all_scores):.3f}",
        "Std Score": f"{np.std(all_scores):.3f}",
        "Avg Completion %": f"{np.mean(completion_rates):.1f}",
        "Avg Makespan": f"{np.mean([k['makespan'] for k in all_kpis]):.1f}",
        "Avg Tardiness": f"{np.mean([k['total_tardiness'] for k in all_kpis]):.1f}",
        "Avg Utilization %": f"{np.mean([k['mean_utilization'] for k in all_kpis]) * 100:.1f}",
    }

def run_comprehensive_evaluation(model_path: str, generate_gantt: bool = False, output_dir: str = None, run_name: str = None):
    """运行综合评估：包括基准测试和泛化能力测试"""
    
    print("="*80, flush=True)
    print("🚀 开始进行静态环境下的调度策略综合评估", flush=True)
    print(f"🔁 每个策略将独立运行 {NUM_EVAL_EPISODES} 次以获取可靠的统计结果。", flush=True)
    print("="*80, flush=True)

    # 测试配置列表
    test_configs = [
        ("基准测试", STATIC_EVAL_CONFIG),
        ("泛化测试1-高压力短交期", GENERALIZATION_CONFIG_1),
        ("泛化测试2-混合优先级", GENERALIZATION_CONFIG_2),
        ("泛化测试3-大批量长周期", GENERALIZATION_CONFIG_3),
    ]
    
    all_results = []
    
    for test_name, config in test_configs:
        print(f"\n🔬 开始 {test_name}", flush=True)
        print("="*60, flush=True)
        
        # 🔧 V4 修复：直接传递config，无需上下文管理器
        # 1. 评估MARL模型
        marl_kpis, marl_scores = evaluate_marl_model(model_path, config, generate_gantt=generate_gantt, output_dir=output_dir, run_name=run_name)
        
        # 2. 评估启发式算法 (甘特图保存到父目录)
        heuristic_output_dir = os.path.dirname(output_dir) if output_dir else None
        fifo_kpis, fifo_scores = evaluate_heuristic('FIFO', config, generate_gantt=generate_gantt, output_dir=heuristic_output_dir, run_name=run_name)
        edd_kpis, edd_scores = evaluate_heuristic('EDD', config, generate_gantt=generate_gantt, output_dir=heuristic_output_dir, run_name=run_name)
        spt_kpis, spt_scores = evaluate_heuristic('SPT', config, generate_gantt=generate_gantt, output_dir=heuristic_output_dir, run_name=run_name)

        # 3. 汇总结果
        results = [
            aggregate_results("MARL (PPO)", marl_kpis, marl_scores, config),
            aggregate_results("SPT", spt_kpis, spt_scores, config),
            aggregate_results("EDD", edd_kpis, edd_scores, config),
            aggregate_results("FIFO", fifo_kpis, fifo_scores, config),
        ]
        
        # 4. 打印当前测试结果
        df = pd.DataFrame(results)
        print(f"\n🏆 {test_name} - 评估对比结果", flush=True)
        print("-"*60, flush=True)
        print(df.to_string(index=False), flush=True)
        
        # 保存结果用于最终汇总
        for result in results:
            result['Test_Config'] = test_name
        all_results.extend(results)
        
        print("\n" + "="*60, flush=True)
    
    # 5. 生成最终汇总报告
    print(f"\n🎯 最终汇总报告 - 泛化能力分析", flush=True)
    print("="*80, flush=True)
    
    # 按方法分组展示结果
    methods = ["MARL (PPO)", "SPT", "EDD", "FIFO"]
    
    for method in methods:
        method_results = [r for r in all_results if r['Method'] == method]
        if method_results:
            print(f"\n📊 {method} 在不同测试配置下的表现:", flush=True)
            method_df = pd.DataFrame(method_results)
            # 重新排列列顺序，把Test_Config放在前面
            cols = ['Test_Config'] + [col for col in method_df.columns if col != 'Test_Config']
            method_df = method_df[cols]
            print(method_df.to_string(index=False), flush=True)

def main():
    parser = argparse.ArgumentParser(description="评估MARL模型与启发式算法的性能")
    parser.add_argument(
        "--model_path", 
        type=str, 
        required=True,
        help="指向已训练好的MARL actor模型文件 (.keras) 的路径"
    )
    parser.add_argument(
        "--generalization", 
        action="store_true",
        help="是否进行泛化能力测试 (默认只进行基准测试)"
    )
    parser.add_argument(
        "--gantt",
        action="store_true",
        help="是否为每个评估场景生成详细的调度甘特图"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="指定一个目录来存放所有输出的甘特图文件"
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="为本次运行提供一个名称，将用作甘特图文件名的前缀"
    )
    args = parser.parse_args()

    if args.generalization:
        # 运行完整的泛化能力测试
        run_comprehensive_evaluation(args.model_path, generate_gantt=args.gantt, output_dir=args.output_dir, run_name=args.run_name)
    else:
        # 仅运行基准测试 (原有功能)
        print("="*80, flush=True)
        print("🚀 开始进行静态环境下的调度策略综合评估", flush=True)
        print(f"🔁 每个策略将独立运行 {NUM_EVAL_EPISODES} 次以获取可靠的统计结果。", flush=True)
        print("="*80, flush=True)

        # 1. 评估MARL模型
        marl_kpis, marl_scores = evaluate_marl_model(args.model_path, generate_gantt=args.gantt, output_dir=args.output_dir, run_name=args.run_name)
        
        # 2. 评估启发式算法 (甘特图保存到父目录)
        heuristic_output_dir = os.path.dirname(args.output_dir) if args.output_dir else None
        fifo_kpis, fifo_scores = evaluate_heuristic('FIFO', generate_gantt=args.gantt, output_dir=heuristic_output_dir, run_name=args.run_name)
        edd_kpis, edd_scores = evaluate_heuristic('EDD', generate_gantt=args.gantt, output_dir=heuristic_output_dir, run_name=args.run_name)
        spt_kpis, spt_scores = evaluate_heuristic('SPT', generate_gantt=args.gantt, output_dir=heuristic_output_dir, run_name=args.run_name)

        # 3. 汇总结果
        results = [
            aggregate_results("MARL (PPO)", marl_kpis, marl_scores),
            aggregate_results("SPT", spt_kpis, spt_scores),
            aggregate_results("EDD", edd_kpis, edd_scores),
            aggregate_results("FIFO", fifo_kpis, fifo_scores),
        ]
        
        # 4. 创建并打印结果表格
        df = pd.DataFrame(results)
        
        print("\n" + "="*80, flush=True)
        print("🏆 最终评估对比结果", flush=True)
        print("="*80, flush=True)
        print(df.to_string(index=False), flush=True)
        print("="*80, flush=True)


if __name__ == "__main__":
    main()
