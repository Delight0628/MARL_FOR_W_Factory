"""
W工厂生产调度系统 - 模型评估脚本
用于评估训练好的MARL模型性能并与基准算法对比
"""

import os
import sys
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Tuple
import time

import ray
from ray.rllib.algorithms.ppo import PPO
from ray.rllib.env import PettingZooEnv
from ray.tune.registry import register_env

# 添加环境路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from environments.w_factory_env import make_parallel_env
from environments.w_factory_config import *

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# =============================================================================
# 1. 环境注册和配置 (Environment Registration & Configuration)
# =============================================================================

def env_creator(config):
    """环境创建函数"""
    return PettingZooEnv(make_parallel_env(config))

register_env("w_factory", env_creator)

# =============================================================================
# 2. 模型评估类 (Model Evaluation Class)
# =============================================================================

class ModelEvaluator:
    """模型评估器"""
    
    def __init__(self, checkpoint_path: str = None):
        self.checkpoint_path = checkpoint_path
        self.algorithm = None
        self.results = {}
        
        if checkpoint_path:
            self._load_model()
    
    def _load_model(self):
        """加载训练好的模型"""
        try:
            print(f"加载模型检查点: {self.checkpoint_path}")
            
            # 初始化Ray (Windows兼容模式)
            if not ray.is_initialized():
                ray.init(local_mode=True, ignore_reinit_error=True)
            
            # 加载算法
            self.algorithm = PPO.from_checkpoint(self.checkpoint_path)
            print("✓ 模型加载成功")
            
        except Exception as e:
            print(f"✗ 模型加载失败: {e}")
            self.algorithm = None
    
    def evaluate_model(self, num_episodes: int = 10) -> Dict[str, Any]:
        """评估MARL模型"""
        if not self.algorithm:
            print("模型未加载，无法评估")
            return {}
        
        print(f"开始评估MARL模型 ({num_episodes} 轮)")
        
        env = make_parallel_env()
        episode_rewards = []
        episode_stats = []
        
        for episode in range(num_episodes):
            print(f"评估轮次 {episode + 1}/{num_episodes}")
            
            observations, infos = env.reset(seed=episode)
            episode_reward = 0
            step_count = 0
            
            while True:
                # 获取动作
                actions = {}
                for agent_id in env.agents:
                    action = self.algorithm.compute_single_action(
                        observations[agent_id],
                        policy_id="shared_policy"
                    )
                    actions[agent_id] = action
                
                # 执行动作
                observations, rewards, terminations, truncations, infos = env.step(actions)
                
                episode_reward += sum(rewards.values())
                step_count += 1
                
                # 检查是否结束
                if any(terminations.values()) or any(truncations.values()):
                    break
            
            episode_rewards.append(episode_reward)
            
            # 获取最终统计
            if any(infos.values()) and "final_stats" in list(infos.values())[0]:
                final_stats = list(infos.values())[0]["final_stats"]
                final_stats["episode_reward"] = episode_reward
                final_stats["steps"] = step_count
                episode_stats.append(final_stats)
        
        # 计算统计结果
        results = {
            "algorithm": "MARL",
            "num_episodes": num_episodes,
            "mean_reward": np.mean(episode_rewards),
            "std_reward": np.std(episode_rewards),
            "min_reward": np.min(episode_rewards),
            "max_reward": np.max(episode_rewards),
            "episode_rewards": episode_rewards
        }
        
        if episode_stats:
            # 计算性能指标的平均值
            metrics = ["makespan", "total_tardiness", "max_tardiness"]
            for metric in metrics:
                values = [stats.get(metric, 0) for stats in episode_stats]
                results[f"mean_{metric}"] = np.mean(values)
                results[f"std_{metric}"] = np.std(values)
        
        self.results["MARL"] = results
        return results

def evaluate_baseline_algorithms(num_runs: int = 10) -> Dict[str, Any]:
    """评估基准算法"""
    print(f"评估基准算法 ({num_runs} 次运行)")
    
    from main import FIFOScheduler, SPTScheduler
    
    algorithms = {
        "FIFO": FIFOScheduler(),
        "SPT": SPTScheduler()
    }
    
    results = {}
    
    for name, scheduler in algorithms.items():
        print(f"评估 {name} 算法...")
        
        run_results = []
        computation_times = []
        
        for run in range(num_runs):
            start_time = time.time()
            stats = scheduler.schedule(BASE_ORDERS)
            end_time = time.time()
            
            stats["computation_time"] = end_time - start_time
            run_results.append(stats)
            computation_times.append(stats["computation_time"])
        
        # 计算统计结果
        metrics = ["makespan", "total_tardiness", "max_tardiness"]
        algorithm_results = {
            "algorithm": name,
            "num_runs": num_runs,
            "mean_computation_time": np.mean(computation_times),
            "std_computation_time": np.std(computation_times)
        }
        
        for metric in metrics:
            values = [result[metric] for result in run_results]
            algorithm_results[f"mean_{metric}"] = np.mean(values)
            algorithm_results[f"std_{metric}"] = np.std(values)
        
        results[name] = algorithm_results
    
    return results

# =============================================================================
# 3. 结果分析和可视化 (Result Analysis & Visualization)
# =============================================================================

def create_comparison_plots(results: Dict[str, Any], output_dir: str = "results"):
    """创建对比图表"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 准备数据
    algorithms = list(results.keys())
    metrics = ["makespan", "total_tardiness", "max_tardiness"]
    
    # 创建对比图
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle("W工厂生产调度算法性能对比", fontsize=16)
    
    # 1. Makespan对比
    ax = axes[0, 0]
    makespan_means = [results[alg].get(f"mean_makespan", 0) for alg in algorithms]
    makespan_stds = [results[alg].get(f"std_makespan", 0) for alg in algorithms]
    
    bars = ax.bar(algorithms, makespan_means, yerr=makespan_stds, capsize=5)
    ax.set_title("最大完工时间 (Makespan)")
    ax.set_ylabel("时间 (分钟)")
    
    # 添加数值标签
    for bar, mean in zip(bars, makespan_means):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{mean:.1f}', ha='center', va='bottom')
    
    # 2. 总延期时间对比
    ax = axes[0, 1]
    tardiness_means = [results[alg].get(f"mean_total_tardiness", 0) for alg in algorithms]
    tardiness_stds = [results[alg].get(f"std_total_tardiness", 0) for alg in algorithms]
    
    bars = ax.bar(algorithms, tardiness_means, yerr=tardiness_stds, capsize=5, color='orange')
    ax.set_title("总延期时间")
    ax.set_ylabel("时间 (分钟)")
    
    for bar, mean in zip(bars, tardiness_means):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{mean:.1f}', ha='center', va='bottom')
    
    # 3. 最大延期时间对比
    ax = axes[1, 0]
    max_tardiness_means = [results[alg].get(f"mean_max_tardiness", 0) for alg in algorithms]
    max_tardiness_stds = [results[alg].get(f"std_max_tardiness", 0) for alg in algorithms]
    
    bars = ax.bar(algorithms, max_tardiness_means, yerr=max_tardiness_stds, capsize=5, color='red')
    ax.set_title("最大延期时间")
    ax.set_ylabel("时间 (分钟)")
    
    for bar, mean in zip(bars, max_tardiness_means):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{mean:.1f}', ha='center', va='bottom')
    
    # 4. 综合性能雷达图
    ax = axes[1, 1]
    
    # 归一化指标（越小越好，所以用倒数）
    normalized_metrics = {}
    for alg in algorithms:
        normalized_metrics[alg] = []
        for metric in metrics:
            value = results[alg].get(f"mean_{metric}", 1)
            # 避免除零错误
            normalized_value = 1 / (value + 1) if value > 0 else 1
            normalized_metrics[alg].append(normalized_value)
    
    # 简化的性能对比条形图
    x_pos = np.arange(len(algorithms))
    overall_scores = [np.mean(normalized_metrics[alg]) for alg in algorithms]
    
    bars = ax.bar(algorithms, overall_scores, color='green', alpha=0.7)
    ax.set_title("综合性能得分")
    ax.set_ylabel("归一化得分 (越高越好)")
    
    for bar, score in zip(bars, overall_scores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{score:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "algorithm_comparison.png"), dpi=300, bbox_inches='tight')
    plt.show()

def generate_report(results: Dict[str, Any], output_dir: str = "results"):
    """生成评估报告"""
    os.makedirs(output_dir, exist_ok=True)
    
    report_path = os.path.join(output_dir, "evaluation_report.md")
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# W工厂生产调度系统 - 算法性能评估报告\n\n")
        f.write(f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## 评估概述\n\n")
        f.write("本报告对比了多智能体强化学习(MARL)算法与传统基准算法在W工厂生产调度问题上的性能。\n\n")
        
        f.write("## 算法性能对比\n\n")
        f.write("| 算法 | 平均Makespan | 平均总延期 | 平均最大延期 |\n")
        f.write("|------|-------------|-----------|-------------|\n")
        
        for alg_name, result in results.items():
            makespan = result.get("mean_makespan", 0)
            total_tardiness = result.get("mean_total_tardiness", 0)
            max_tardiness = result.get("mean_max_tardiness", 0)
            
            f.write(f"| {alg_name} | {makespan:.2f} | {total_tardiness:.2f} | {max_tardiness:.2f} |\n")
        
        f.write("\n## 详细结果\n\n")
        
        for alg_name, result in results.items():
            f.write(f"### {alg_name}\n\n")
            
            for key, value in result.items():
                if isinstance(value, (int, float)):
                    f.write(f"- {key}: {value:.4f}\n")
                elif isinstance(value, list) and len(value) <= 10:
                    f.write(f"- {key}: {value}\n")
            
            f.write("\n")
        
        f.write("## 结论\n\n")
        
        # 找出最佳算法
        best_makespan = min(results.keys(), key=lambda x: results[x].get("mean_makespan", float('inf')))
        best_tardiness = min(results.keys(), key=lambda x: results[x].get("mean_total_tardiness", float('inf')))
        
        f.write(f"- 最小Makespan: {best_makespan} ({results[best_makespan].get('mean_makespan', 0):.2f}分钟)\n")
        f.write(f"- 最小总延期: {best_tardiness} ({results[best_tardiness].get('mean_total_tardiness', 0):.2f}分钟)\n")
        
        if "MARL" in results:
            marl_makespan = results["MARL"].get("mean_makespan", 0)
            fifo_makespan = results.get("FIFO", {}).get("mean_makespan", 0)
            
            if fifo_makespan > 0:
                improvement = (fifo_makespan - marl_makespan) / fifo_makespan * 100
                f.write(f"- MARL相比FIFO的Makespan改进: {improvement:.2f}%\n")
    
    print(f"评估报告已保存到: {report_path}")

# =============================================================================
# 4. 主函数 (Main Function)
# =============================================================================

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="W工厂生产调度系统模型评估")
    parser.add_argument("--checkpoint", type=str, default=None,
                       help="MARL模型检查点路径")
    parser.add_argument("--episodes", type=int, default=10,
                       help="MARL评估轮数")
    parser.add_argument("--baseline-runs", type=int, default=10,
                       help="基准算法运行次数")
    parser.add_argument("--output-dir", type=str, default="results",
                       help="结果输出目录")
    parser.add_argument("--no-plots", action="store_true",
                       help="不生成图表")
    
    args = parser.parse_args()
    
    print("W工厂生产调度系统 - 模型评估")
    print("=" * 60)
    
    all_results = {}
    
    # 评估基准算法
    print("\n1. 评估基准算法...")
    baseline_results = evaluate_baseline_algorithms(args.baseline_runs)
    all_results.update(baseline_results)
    
    # 评估MARL模型
    if args.checkpoint:
        print("\n2. 评估MARL模型...")
        evaluator = ModelEvaluator(args.checkpoint)
        marl_results = evaluator.evaluate_model(args.episodes)
        if marl_results:
            all_results.update({"MARL": marl_results})
    else:
        print("\n2. 跳过MARL评估（未提供检查点路径）")
    
    # 保存结果
    results_file = os.path.join(args.output_dir, f"evaluation_results_{int(time.time())}.json")
    os.makedirs(args.output_dir, exist_ok=True)
    
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    print(f"\n结果已保存到: {results_file}")
    
    # 生成图表
    if not args.no_plots and len(all_results) > 1:
        print("\n3. 生成对比图表...")
        create_comparison_plots(all_results, args.output_dir)
    
    # 生成报告
    print("\n4. 生成评估报告...")
    generate_report(all_results, args.output_dir)
    
    # 显示结果摘要
    print("\n" + "=" * 60)
    print("评估结果摘要")
    print("=" * 60)
    
    for alg_name, result in all_results.items():
        print(f"\n{alg_name}:")
        makespan = result.get("mean_makespan", 0)
        tardiness = result.get("mean_total_tardiness", 0)
        max_tardiness = result.get("mean_max_tardiness", 0)
        
        print(f"  平均Makespan: {makespan:.2f}")
        print(f"  平均总延期: {tardiness:.2f}")
        print(f"  平均最大延期: {max_tardiness:.2f}")
    
    print("\n评估完成！")

if __name__ == "__main__":
    main() 