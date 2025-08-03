"""
W工厂生产调度系统 - 主训练脚本
支持MARL训练和基准算法对比
"""

import os
import sys
import argparse
import time
import json
from typing import Dict, Any, List
import numpy as np
import pandas as pd

import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env import PettingZooEnv
from ray.tune.registry import register_env

# 添加环境路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from environments.w_factory_env import make_parallel_env
from environments.w_factory_config import *

# =============================================================================
# 1. 环境注册 (Environment Registration)
# =============================================================================

def env_creator(config):
    """环境创建函数"""
    return PettingZooEnv(make_parallel_env(config))

# 注册环境
register_env("w_factory", env_creator)

# =============================================================================
# 2. 基准算法实现 (Benchmark Algorithms)
# =============================================================================

class BaselineScheduler:
    """基准调度算法基类"""
    
    def __init__(self, algorithm: str):
        self.algorithm = algorithm
        self.stats = {
            'makespan': 0,
            'total_tardiness': 0,
            'max_tardiness': 0,
            'equipment_utilization': {},
            'completed_parts': 0
        }
    
    def schedule(self, orders: List[Dict]) -> Dict[str, Any]:
        """执行调度算法"""
        raise NotImplementedError
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计结果"""
        return self.stats

class FIFOScheduler(BaselineScheduler):
    """先进先出调度算法"""
    
    def __init__(self):
        super().__init__("FIFO")
    
    def schedule(self, orders: List[Dict]) -> Dict[str, Any]:
        """FIFO调度实现"""
        # 简化的FIFO实现
        total_time = 0
        total_tardiness = 0
        max_tardiness = 0
        
        for order in orders:
            product = order["product"]
            quantity = order["quantity"]
            due_date = order["due_date"]
            
            # 计算产品总加工时间
            processing_time = calculate_product_total_time(product) * quantity
            total_time += processing_time
            
            # 计算延期
            tardiness = max(0, total_time - due_date)
            total_tardiness += tardiness
            max_tardiness = max(max_tardiness, tardiness)
        
        self.stats.update({
            'makespan': total_time,
            'total_tardiness': total_tardiness,
            'max_tardiness': max_tardiness,
            'completed_parts': sum(order["quantity"] for order in orders)
        })
        
        return self.stats

class SPTScheduler(BaselineScheduler):
    """最短处理时间优先调度算法"""
    
    def __init__(self):
        super().__init__("SPT")
    
    def schedule(self, orders: List[Dict]) -> Dict[str, Any]:
        """SPT调度实现"""
        # 按处理时间排序
        sorted_orders = sorted(orders, 
                             key=lambda x: calculate_product_total_time(x["product"]))
        
        total_time = 0
        total_tardiness = 0
        max_tardiness = 0
        
        for order in sorted_orders:
            product = order["product"]
            quantity = order["quantity"]
            due_date = order["due_date"]
            
            processing_time = calculate_product_total_time(product) * quantity
            total_time += processing_time
            
            tardiness = max(0, total_time - due_date)
            total_tardiness += tardiness
            max_tardiness = max(max_tardiness, tardiness)
        
        self.stats.update({
            'makespan': total_time,
            'total_tardiness': total_tardiness,
            'max_tardiness': max_tardiness,
            'completed_parts': sum(order["quantity"] for order in orders)
        })
        
        return self.stats

def run_baseline_comparison():
    """运行基准算法对比"""
    print("=" * 60)
    print("运行基准算法对比测试")
    print("=" * 60)
    
    algorithms = {
        "FIFO": FIFOScheduler(),
        "SPT": SPTScheduler()
    }
    
    results = {}
    
    for name, scheduler in algorithms.items():
        print(f"\n运行 {name} 算法...")
        start_time = time.time()
        
        stats = scheduler.schedule(BASE_ORDERS)
        
        end_time = time.time()
        stats['computation_time'] = end_time - start_time
        
        results[name] = stats
        
        print(f"  最大完工时间: {stats['makespan']:.2f}")
        print(f"  总延期时间: {stats['total_tardiness']:.2f}")
        print(f"  最大延期时间: {stats['max_tardiness']:.2f}")
        print(f"  计算时间: {stats['computation_time']:.4f}秒")
    
    return results

# =============================================================================
# 3. MARL训练配置 (MARL Training Configuration)
# =============================================================================

def get_training_config() -> Dict[str, Any]:
    """获取训练配置"""
    
    # 基础PPO配置
    config = (
        PPOConfig()
        .environment(
            env="w_factory",
            env_config={},
            disable_env_checking=True
        )
        .framework("tf2")
        .rollouts(
            num_rollout_workers=0,  # Windows兼容：使用单进程模式
            num_envs_per_worker=1,
            rollout_fragment_length=TRAINING_CONFIG["rollout_fragment_length"]
        )
        .training(
            train_batch_size=TRAINING_CONFIG["train_batch_size"],
            sgd_minibatch_size=TRAINING_CONFIG["sgd_minibatch_size"],
            num_sgd_iter=TRAINING_CONFIG["num_sgd_iter"],
            lr=TRAINING_CONFIG["lr"],
            gamma=TRAINING_CONFIG["gamma"],
            lambda_=TRAINING_CONFIG["lambda"],
            clip_param=TRAINING_CONFIG["clip_param"],
            vf_clip_param=TRAINING_CONFIG["vf_clip_param"],
            entropy_coeff=TRAINING_CONFIG["entropy_coeff"],
            vf_loss_coeff=TRAINING_CONFIG["vf_loss_coeff"]
        )
        .multi_agent(
            policies={
                "shared_policy": (
                    None,  # 使用默认策略类
                    gym.spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32),
                    gym.spaces.Discrete(2),
                    {}
                )
            },
            policy_mapping_fn=lambda agent_id, episode, worker, **kwargs: "shared_policy",
            policies_to_train=["shared_policy"]
        )
        .resources(
            num_gpus=0,  # 使用CPU训练
            num_cpus_per_worker=0  # local_mode下不需要额外CPU
        )
        .evaluation(
            evaluation_interval=EVALUATION_CONFIG["evaluation_interval"],
            evaluation_duration=EVALUATION_CONFIG["evaluation_duration"],
            evaluation_num_workers=0,  # Windows兼容：评估也使用单进程
            evaluation_config=EVALUATION_CONFIG["evaluation_config"]
        )
    )
    
    return config

def run_marl_training(config: Dict[str, Any], checkpoint_dir: str = None):
    """运行MARL训练"""
    print("=" * 60)
    print("开始多智能体强化学习训练")
    print("=" * 60)
    
    # 初始化Ray (Windows兼容模式)
    if not ray.is_initialized():
        ray.init(local_mode=True, ignore_reinit_error=True)  # 使用local_mode避免Windows进程问题
    
    # 创建结果目录
    results_dir = os.path.join(os.getcwd(), "results")
    os.makedirs(results_dir, exist_ok=True)
    
    # 训练配置
    training_config = get_training_config()
    
    # 运行训练
    tuner = tune.Tuner(
        "PPO",
        param_space=training_config.to_dict(),
        run_config=tune.RunConfig(
            name="w_factory_marl",
            local_dir=results_dir,
            stop=STOP_CONFIG,
            checkpoint_config=tune.CheckpointConfig(
                checkpoint_frequency=50,
                num_to_keep=5
            )
        )
    )
    
    print("开始训练...")
    results = tuner.fit()
    
    # 获取最佳结果
    best_result = results.get_best_result(metric="episode_reward_mean", mode="max")
    
    print(f"\n训练完成！")
    print(f"最佳平均奖励: {best_result.metrics['episode_reward_mean']:.2f}")
    print(f"最佳检查点: {best_result.checkpoint}")
    
    return best_result

# =============================================================================
# 4. 主函数 (Main Function)
# =============================================================================

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="W工厂生产调度系统训练")
    parser.add_argument("--mode", type=str, default="train", 
                       choices=["train", "fifo", "spt", "baseline", "all"],
                       help="运行模式")
    parser.add_argument("--checkpoint", type=str, default=None,
                       help="检查点路径（用于继续训练）")
    parser.add_argument("--episodes", type=int, default=1000,
                       help="训练轮数")
    parser.add_argument("--workers", type=int, default=4,
                       help="并行工作进程数")
    
    args = parser.parse_args()
    
    print("W工厂生产调度多智能体强化学习系统")
    print("=" * 60)
    print(f"运行模式: {args.mode}")
    print(f"配置验证: ", end="")
    if validate_config():
        print("✓ 通过")
    else:
        print("✗ 失败")
        return
    
    # 创建日志目录
    os.makedirs("logs", exist_ok=True)
    
    results = {}
    
    if args.mode in ["fifo", "baseline", "all"]:
        fifo_scheduler = FIFOScheduler()
        results["FIFO"] = fifo_scheduler.schedule(BASE_ORDERS)
    
    if args.mode in ["spt", "baseline", "all"]:
        spt_scheduler = SPTScheduler()
        results["SPT"] = spt_scheduler.schedule(BASE_ORDERS)
    
    if args.mode in ["baseline", "all"]:
        baseline_results = run_baseline_comparison()
        results.update(baseline_results)
    
    if args.mode in ["train", "all"]:
        # 更新训练配置
        TRAINING_CONFIG["num_workers"] = args.workers
        STOP_CONFIG["training_iteration"] = args.episodes
        
        try:
            best_result = run_marl_training({}, args.checkpoint)
            results["MARL"] = {
                "episode_reward_mean": best_result.metrics["episode_reward_mean"],
                "checkpoint_path": str(best_result.checkpoint)
            }
        except Exception as e:
            print(f"训练过程中出现错误: {e}")
            return
    
    # 保存结果
    if results:
        results_file = f"logs/results_{int(time.time())}.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n结果已保存到: {results_file}")
    
    # 显示结果摘要
    if results:
        print("\n" + "=" * 60)
        print("结果摘要")
        print("=" * 60)
        for method, stats in results.items():
            print(f"\n{method}:")
            if isinstance(stats, dict):
                for key, value in stats.items():
                    if isinstance(value, (int, float)):
                        print(f"  {key}: {value:.2f}")
    
    print("\n训练完成！")

if __name__ == "__main__":
    main() 