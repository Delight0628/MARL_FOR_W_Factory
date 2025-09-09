import time
import numpy as np
import pprint
from datetime import datetime

# 核心：复用与RL训练完全相同的仿真环境
from environments.w_factory_env import WFactorySim
from environments.w_factory_config import get_total_parts_count

class HeuristicDispatcher:
    """启发式调度器：根据指定规则选择零件"""
    
    def __init__(self, rule: str):
        if rule not in ['FIFO', 'EDD', 'SPT']:
            raise ValueError(f"未知的启发式规则: {rule}")
        self.rule = rule

    def choose_part(self, parts: list, station_name: str):
        """
        从可用零件列表中根据规则选择一个。
        
        Args:
            parts (list): 可供选择的 Part 对象列表。
            station_name (str): 当前工作站的名称。

        Returns:
            Part: 被选中的 Part 对象。
        """
        if not parts:
            return None

        if self.rule == 'FIFO':
            # SimPy的Store本质上就是FIFO，所以直接选择第一个即可
            return parts[0]
        
        elif self.rule == 'EDD':
            # Earliest Due Date: 选择交期最早的零件
            return min(parts, key=lambda p: p.due_date)
            
        elif self.rule == 'SPT':
            # Shortest Processing Time: 选择在此工作站加工时间最短的零件
            return min(parts, key=lambda p: p.get_processing_time())
            
        return None

def run_single_episode(rule_name: str, episode_num: int, total_episodes: int):
    """运行单次回合的启发式仿真并打印日志"""
    
    iteration_start_time = time.time()
    
    sim = WFactorySim()
    sim.reset() # 确保每个回合都是从干净的状态开始
    dispatcher = HeuristicDispatcher(rule_name)
    
    while not sim.is_done():
        actions = {}
        # 修正: 工作站的名称应该从 sim.resources.keys() 获取
        for station_name in sim.resources.keys():
            # 检查工作站是否空闲且队列中有零件
            if not sim.equipment_status[station_name]['busy_count'] > 0 and len(sim.queues[station_name].items) > 0:
                available_parts = sim.queues[station_name].items
                
                # 使用调度器选择零件
                chosen_part = dispatcher.choose_part(list(available_parts), station_name)
                
                if chosen_part:
                    # 获取选中零件在队列中的索引（+1作为动作）
                    part_index = available_parts.index(chosen_part)
                    # 关键修复：actions的键需要匹配agent_id的格式，即 "agent_{station_name}"
                    agent_id = f"agent_{station_name}"
                    actions[agent_id] = part_index + 1
    
        sim.step_with_actions(actions)
        
    iteration_end_time = time.time()
    iteration_duration = iteration_end_time - iteration_start_time
    
    # 修正: 正确的KPI路径是 stats['kpi']
    kpi = sim.get_final_stats()

    # --- 仿照 ppo_marl_train.py 的日志格式 ---
    completed_parts = kpi['total_parts']
    total_parts = get_total_parts_count()
    makespan = kpi['makespan']
    utilization = kpi['mean_utilization']
    tardiness = kpi['total_tardiness']
    
    # 第一行
    line1 = f"🔂 回合 {episode_num:3d}/{total_episodes} | 规则: {rule_name:<4s} | ⏱️本轮用时: {iteration_duration:.1f}s"
    
    # 第二行
    line2 = f"📊 KPI - 总完工时间: {makespan:.1f}min  | 设备利用率: {utilization:.1%} | 延期时间: {tardiness:.1f}min |  完成零件数: {completed_parts:.0f}/{total_parts}"

    # 第三行
    line3 = f"📜 调度规则: {rule_name}"

    # 第四行
    current_time = datetime.now().strftime('%H:%M:%S')
    line4 = f"🔮 当前时间：{current_time}"

    print(line1)
    print(line2)
    print(line3)
    print(line4)
    print()

    return kpi

def main():
    """主函数：运行所有启发式算法并生成对比报告"""
    
    rules_to_test = ['FIFO', 'EDD', 'SPT']
    episodes_per_rule = 5  # 为了快速得到结果，每个规则运行5轮
    
    all_results = {}

    print("=" * 80)
    print("🚀 开始运行启发式调度算法基准测试 🚀")
    print(f"   将要测试的规则: {rules_to_test}")
    print(f"   每个规则的仿真回合数: {episodes_per_rule}")
    print("=" * 80)
    
    for rule in rules_to_test:
        rule_kpis = []
        print(f"\n--- 开始测试规则: {rule} ---\n")
        for i in range(episodes_per_rule):
            kpi = run_single_episode(rule, i + 1, episodes_per_rule)
            rule_kpis.append(kpi)
        all_results[rule] = rule_kpis

    print("\n" + "=" * 80)
    print("🏆 基准测试完成！最终性能摘要 🏆")
    print("=" * 80)
    
    # 计算并打印平均性能
    print(f"{'调度规则':<15} | {'平均完成率':<15} | {'平均完工时间':<15} | {'平均利用率':<15} | {'平均延期时间':<15}")
    print("-" * 85)
    
    total_parts = get_total_parts_count()
    for rule, kpis in all_results.items():
        avg_completion = np.mean([k['total_parts'] for k in kpis]) # 修正: 'total_parts'
        avg_completion_rate = (avg_completion / total_parts) * 100
        avg_makespan = np.mean([k['makespan'] for k in kpis])
        avg_utilization = np.mean([k['mean_utilization'] for k in kpis]) * 100 # 修正: 'mean_utilization'
        avg_tardiness = np.mean([k['total_tardiness'] for k in kpis]) # 修正: 'total_tardiness'
        
        print(f"{rule:<15} | {avg_completion_rate:13.2f}% | {avg_makespan:13.1f} min | {avg_utilization:13.2f}% | {avg_tardiness:13.1f} min")
        
    print("=" * 85)

if __name__ == "__main__":
    main()
