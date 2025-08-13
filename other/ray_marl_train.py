"""
真正的Ray RLlib多智能体强化学习训练脚本
解决Windows兼容性问题，使用正确的Ray配置
"""
#这个脚本作为从marl训练从windows的不兼容过渡到wsl的版本
import os
import sys
import time
import json
import tempfile
from typing import Dict, Any

# 设置环境变量解决Windows兼容性
os.environ['RAY_DISABLE_IMPORT_WARNING'] = '1'
os.environ['RAY_DEDUP_LOGS'] = '0'

import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env import PettingZooEnv
from ray.tune.registry import register_env
import numpy as np
import gymnasium as gym

# 添加环境路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from environments.w_factory_env import make_parallel_env
from environments.w_factory_config import *

def env_creator(config):
    """环境创建函数"""
    return PettingZooEnv(make_parallel_env(config))

# 注册环境
register_env("w_factory", env_creator)

def get_ray_config():
    """获取Ray初始化配置"""
    # 创建临时目录作为Ray的工作目录
    temp_dir = tempfile.mkdtemp()
    
    return {
        "local_mode": False,  # 尝试使用分布式模式
        "ignore_reinit_error": True,
        "include_dashboard": False,  # 禁用dashboard减少资源占用
        "_temp_dir": temp_dir,
        "object_store_memory": 100000000,  # 100MB对象存储
        "num_cpus": 4,  # 限制CPU使用
    }

def get_training_config():
    """获取训练配置"""
    
    config = (
        PPOConfig()
        .environment(
            env="w_factory",
            env_config={},
            disable_env_checking=True
        )
        .framework("tf2")
        .env_runners(
            # Windows兼容性配置 (Ray 2.48+ API)
            num_env_runners=2,  # 减少runner数量
            num_envs_per_env_runner=1,
            rollout_fragment_length=100,  # 减少片段长度
            batch_mode="complete_episodes"
        )
        .training(
            # PPO训练参数
            train_batch_size=1000,  # 减小批次大小
            sgd_minibatch_size=64,
            num_sgd_iter=5,
            lr=3e-4,
            gamma=0.99,
            lambda_=0.95,
            clip_param=0.2,
            vf_clip_param=10.0,
            entropy_coeff=0.01,
            vf_loss_coeff=0.5,
            # 梯度裁剪
            grad_clip=0.5,
        )
        .multi_agent(
            # 多智能体配置
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
            # 资源配置
            num_gpus=0,  # 使用CPU
            num_cpus_per_env_runner=1,
            num_gpus_per_env_runner=0
        )
        .evaluation(
            # 评估配置
            evaluation_interval=20,
            evaluation_duration=5,
            evaluation_num_env_runners=1,
            evaluation_config={
                "explore": False,
                "render_env": False,
            }
        )
        .debugging(
            # 调试配置
            log_level="ERROR",  # 减少日志输出
        )
        .experimental(
            # 实验性配置
            _disable_preprocessor_api=True,
        )
    )
    
    return config

def run_ray_training(num_iterations: int = 50):
    """运行Ray RLlib训练"""
    print("=" * 60)
    print("W工厂多智能体强化学习训练 - Ray RLlib版本")
    print("=" * 60)
    print("框架: Ray RLlib")
    print("算法: PPO (Proximal Policy Optimization)")
    print("多智能体: 策略共享MAPPO")
    print("=" * 60)
    
    # 验证配置
    if not validate_config():
        print("配置验证失败")
        return None
    
    try:
        # 初始化Ray
        ray_config = get_ray_config()
        print("初始化Ray...")
        
        if ray.is_initialized():
            ray.shutdown()
        
        ray.init(**ray_config)
        print("✓ Ray初始化成功")
        
        # 获取训练配置
        training_config = get_training_config()
        
        # 设置停止条件
        stop_config = {
            "training_iteration": num_iterations,
            "timesteps_total": num_iterations * 1000,
        }
        
        # 创建结果目录
        results_dir = os.path.join(os.getcwd(), "ray_results")
        os.makedirs(results_dir, exist_ok=True)
        
        print(f"开始训练 ({num_iterations} 轮)...")
        start_time = time.time()
        
        # 运行训练
        tuner = tune.Tuner(
            "PPO",
            param_space=training_config.to_dict(),
            run_config=tune.RunConfig(
                name="w_factory_ray_marl",
                local_dir=results_dir,
                stop=stop_config,
                checkpoint_config=tune.CheckpointConfig(
                    checkpoint_frequency=10,
                    num_to_keep=3
                ),
                verbose=1  # 减少输出
            )
        )
        
        results = tuner.fit()
        
        # 获取最佳结果
        best_result = results.get_best_result(
            metric="episode_reward_mean", 
            mode="max"
        )
        
        training_time = time.time() - start_time
        
        print("\n" + "=" * 60)
        print("Ray RLlib训练完成！")
        print("=" * 60)
        print(f"训练时间: {training_time/60:.2f} 分钟")
        print(f"最佳平均奖励: {best_result.metrics['episode_reward_mean']:.2f}")
        print(f"最佳检查点: {best_result.checkpoint}")
        
        # 保存结果摘要
        summary = {
            "framework": "Ray RLlib",
            "algorithm": "PPO/MAPPO",
            "training_time_minutes": training_time / 60,
            "best_reward": best_result.metrics['episode_reward_mean'],
            "checkpoint_path": str(best_result.checkpoint),
            "iterations": num_iterations,
            "agents": list(WORKSTATIONS.keys()),
            "config": "Windows兼容模式"
        }
        
        summary_file = os.path.join(results_dir, f"ray_training_summary_{int(time.time())}.json")
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        print(f"训练摘要已保存: {summary_file}")
        
        return best_result
        
    except Exception as e:
        print(f"Ray训练过程中出现错误: {e}")
        print("错误详情:")
        import traceback
        traceback.print_exc()
        
        # 尝试fallback到本地模式
        print("\n尝试fallback到本地模式...")
        return run_ray_training_local_mode(num_iterations)
    
    finally:
        # 清理Ray
        if ray.is_initialized():
            ray.shutdown()

def run_ray_training_local_mode(num_iterations: int = 50):
    """使用本地模式运行Ray训练（fallback方案）"""
    print("=" * 60)
    print("Ray RLlib训练 - 本地模式 (Fallback)")
    print("=" * 60)
    
    try:
        # 本地模式初始化Ray
        if ray.is_initialized():
            ray.shutdown()
        
        ray.init(local_mode=True, ignore_reinit_error=True)
        print("✓ Ray本地模式初始化成功")
        
        # 本地模式配置（简化）
        config = (
            PPOConfig()
            .environment(
                env="w_factory",
                env_config={},
                disable_env_checking=True
            )
            .framework("tf2")
            .rollouts(
                num_rollout_workers=0,  # 本地模式不使用worker
                rollout_fragment_length=200
            )
            .training(
                train_batch_size=500,
                sgd_minibatch_size=32,
                num_sgd_iter=3,
                lr=3e-4
            )
            .multi_agent(
                policies={
                    "shared_policy": (
                        None,
                        gym.spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32),
                        gym.spaces.Discrete(2),
                        {}
                    )
                },
                policy_mapping_fn=lambda agent_id, episode, worker, **kwargs: "shared_policy",
                policies_to_train=["shared_policy"]
            )
            .resources(num_gpus=0)
        )
        
        # 运行训练
        results_dir = os.path.join(os.getcwd(), "ray_results_local")
        os.makedirs(results_dir, exist_ok=True)
        
        start_time = time.time()
        
        tuner = tune.Tuner(
            "PPO",
            param_space=config.to_dict(),
            run_config=tune.RunConfig(
                name="w_factory_ray_local",
                local_dir=results_dir,
                stop={"training_iteration": num_iterations},
                checkpoint_config=tune.CheckpointConfig(
                    checkpoint_frequency=10,
                    num_to_keep=2
                )
            )
        )
        
        results = tuner.fit()
        best_result = results.get_best_result(metric="episode_reward_mean", mode="max")
        
        training_time = time.time() - start_time
        
        print(f"\n本地模式训练完成！")
        print(f"训练时间: {training_time/60:.2f} 分钟")
        print(f"最佳平均奖励: {best_result.metrics['episode_reward_mean']:.2f}")
        
        return best_result
        
    except Exception as e:
        print(f"本地模式训练也失败: {e}")
        return None
    
    finally:
        if ray.is_initialized():
            ray.shutdown()

def compare_with_baselines():
    """与基准算法对比"""
    print("\n运行基准算法对比...")
    
    try:
        from main import FIFOScheduler, SPTScheduler
        
        algorithms = {
            "FIFO": FIFOScheduler(),
            "SPT": SPTScheduler()
        }
        
        results = {}
        
        for name, scheduler in algorithms.items():
            print(f"运行 {name} 算法...")
            start_time = time.time()
            stats = scheduler.schedule(BASE_ORDERS)
            end_time = time.time()
            
            stats['computation_time'] = end_time - start_time
            results[name] = stats
            
            print(f"  {name} - Makespan: {stats['makespan']:.2f}, "
                  f"延期: {stats['total_tardiness']:.2f}")
        
        return results
        
    except Exception as e:
        print(f"基准算法对比失败: {e}")
        return {}

def main():
    """主函数"""
    print("W工厂多智能体强化学习系统")
    print("使用Ray RLlib框架的正式MARL训练")
    print("=" * 60)
    
    try:
        # 运行Ray训练
        ray_result = run_ray_training(num_iterations=30)  # 减少迭代次数用于测试
        
        if ray_result:
            print("\n🎉 Ray RLlib训练成功完成！")
            
            # 运行基准对比
            baseline_results = compare_with_baselines()
            
            if baseline_results:
                print("\n" + "=" * 60)
                print("性能对比")
                print("=" * 60)
                print(f"Ray MARL最佳奖励: {ray_result.metrics['episode_reward_mean']:.2f}")
                
                for name, stats in baseline_results.items():
                    makespan = stats.get('makespan', 0)
                    tardiness = stats.get('total_tardiness', 0)
                    print(f"{name:10} - Makespan: {makespan:6.1f}, 延期: {tardiness:6.1f}")
            
            print("\n✅ 这是真正的Ray RLlib MARL训练！")
            print("✅ 使用PPO/MAPPO算法")
            print("✅ 多智能体策略共享")
            print("✅ 神经网络策略学习")
            
        else:
            print("\n❌ Ray训练失败")
            print("建议检查系统环境或使用简化训练脚本")
            
    except Exception as e:
        print(f"主程序执行失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 