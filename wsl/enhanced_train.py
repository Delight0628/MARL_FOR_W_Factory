#!/usr/bin/env python3
"""
增强版WSL Ray RLlib训练脚本
- 完整的时间统计和预测
- 简洁的检查点信息
- 用户友好的进度显示
- 实时性能监控
"""

import os
import sys
import time
import json
from pathlib import Path
from datetime import datetime, timedelta

# 添加项目路径
current_dir = Path(__file__).parent.absolute()
project_root = current_dir.parent
sys.path.append(str(project_root))

# WSL环境优化
os.environ['RAY_DISABLE_IMPORT_WARNING'] = '1'
os.environ['RAY_USAGE_STATS_ENABLED'] = '0'
os.environ['RAY_DEDUP_LOGS'] = '0'

import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env
from environments.w_factory_env import WFactoryGymEnv

class TrainingTimer:
    """训练时间管理器"""
    
    def __init__(self, total_iterations):
        self.start_time = time.time()
        self.total_iterations = total_iterations
        self.iteration_times = []
        self.current_iteration = 0
    
    def start_iteration(self):
        """开始一轮训练"""
        self.iteration_start = time.time()
    
    def end_iteration(self):
        """结束一轮训练"""
        duration = time.time() - self.iteration_start
        self.iteration_times.append(duration)
        self.current_iteration += 1
        return duration
    
    def get_stats(self):
        """获取时间统计"""
        elapsed = time.time() - self.start_time
        
        if not self.iteration_times:
            return {
                'elapsed_time': elapsed,
                'avg_iteration_time': 0,
                'estimated_remaining': 0,
                'estimated_finish': None
            }
        
        avg_time = sum(self.iteration_times) / len(self.iteration_times)
        remaining_iterations = self.total_iterations - self.current_iteration
        estimated_remaining = remaining_iterations * avg_time
        estimated_finish = time.time() + estimated_remaining
        
        return {
            'elapsed_time': elapsed,
            'avg_iteration_time': avg_time,
            'estimated_remaining': estimated_remaining,
            'estimated_finish': estimated_finish,
            'fastest_iteration': min(self.iteration_times),
            'slowest_iteration': max(self.iteration_times),
            'current_iteration_time': self.iteration_times[-1] if self.iteration_times else 0
        }

def format_time(seconds):
    """格式化时间显示"""
    if seconds < 60:
        return f"{seconds:.1f}秒"
    elif seconds < 3600:
        return f"{seconds/60:.1f}分钟"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}小时{minutes}分钟"

def create_enhanced_config():
    """创建增强的训练配置"""
    config = (
        PPOConfig()
        .environment(
            env="w_factory",
            env_config={"debug_level": "WARNING"}
        )
        .framework("torch")
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False,
        )
        .training(
            train_batch_size=2000,
            minibatch_size=128,
            num_epochs=5,
            lr=5e-4,
            gamma=0.99,
            lambda_=0.95,
            clip_param=0.2,
            vf_clip_param=10.0,
            entropy_coeff=0.01,
            vf_loss_coeff=0.5,
        )
        .env_runners(
            num_env_runners=0,
            rollout_fragment_length=200,
        )
        .resources(
            num_gpus=0,
        )
        .multi_agent(
            policies={"shared_policy": (None, None, None, None)},
            policy_mapping_fn=lambda agent_id, *args, **kwargs: "shared_policy",
        )
        .debugging(
            log_level="WARNING",
        )
    )
    return config

def run_enhanced_training(num_iterations=10):
    """运行增强版训练"""
    print("🚀 增强版MARL训练系统")
    print("=" * 80)
    
    # 显示训练开始信息
    start_datetime = datetime.now()
    print(f"📅 开始时间: {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🎯 训练轮次: {num_iterations}")
    print(f"📁 结果目录: enhanced_results")
    
    # 初始化计时器
    timer = TrainingTimer(num_iterations)
    
    try:
        # 初始化Ray
        print("\n🔧 初始化Ray环境...")
        if not ray.is_initialized():
            ray.init(
                num_cpus=4,
                num_gpus=0,
                object_store_memory=1000000000,
                ignore_reinit_error=True,
                log_to_driver=False,
            )
        print("✅ Ray初始化完成")
        
        # 注册环境
        register_env("w_factory", lambda config: WFactoryGymEnv(config))
        print("✅ 环境注册完成")
        
        # 创建算法
        print("🧠 构建训练算法...")
        config = create_enhanced_config()
        algo = config.build()
        print("✅ 算法构建完成")
        
        # 创建结果目录
        results_dir = r"D:\MPU\毕业论文\MARL_FOR_W_Factory\wsl\ray_result"
        os.makedirs(results_dir, exist_ok=True)
        
        # 训练变量
        best_reward = float('-inf')
        best_checkpoint = None
        total_episodes = 0
        training_history = []
        
        print(f"\n🎯 开始训练 ({start_datetime.strftime('%H:%M:%S')})")
        print("=" * 80)
        
        # 训练循环
        for iteration in range(1, num_iterations + 1):
            timer.start_iteration()
            
            # 执行训练
            result = algo.train()
            
            # 记录时间
            iter_duration = timer.end_iteration()
            stats = timer.get_stats()
            
            # 提取指标
            reward_mean = result.get("episode_reward_mean", 0)
            episodes_this_iter = result.get("episodes_this_iter", 0)
            episode_len_mean = result.get("episode_len_mean", 0)
            
            # 兼容Ray 2.48.0
            if episodes_this_iter == 0 and 'env_runners' in result:
                env_stats = result['env_runners']
                reward_mean = env_stats.get("episode_reward_mean", reward_mean)
                episodes_this_iter = env_stats.get("episodes_this_iter", episodes_this_iter)
                episode_len_mean = env_stats.get("episode_len_mean", episode_len_mean)
            
            total_episodes += episodes_this_iter
            
            # 记录训练历史
            training_history.append({
                'iteration': iteration,
                'reward': reward_mean,
                'episodes': episodes_this_iter,
                'duration': iter_duration,
                'timestamp': time.time()
            })
            
            # 显示进度
            progress = iteration / num_iterations * 100
            progress_bar = "█" * int(progress // 5) + "░" * (20 - int(progress // 5))
            
            print(f"\n轮次 {iteration:2d}/{num_iterations} [{progress_bar}] {progress:5.1f}%")
            print(f"奖励: {reward_mean:8.2f} | Episodes: {episodes_this_iter:2d} | 长度: {episode_len_mean:5.1f}")
            
            # 检查是否有改进
            is_best = reward_mean > best_reward
            if is_best:
                best_reward = reward_mean
                best_checkpoint = algo.save(results_dir)
                print(f"🎉 新最佳! 奖励: {best_reward:.2f} | 已保存检查点")
            else:
                improvement = reward_mean - best_reward
                print(f"📊 当前奖励与最佳差距: {improvement:+6.2f}")
            
            # 显示学习统计
            if "info" in result and "learner" in result["info"]:
                learner_info = result["info"]["learner"]["shared_policy"]
                if "learner_stats" in learner_info:
                    stats_info = learner_info["learner_stats"]
                    policy_loss = stats_info.get("policy_loss", 0)
                    vf_loss = stats_info.get("vf_loss", 0)
                    entropy = stats_info.get("entropy", 0)
                    print(f"损失 - 策略: {policy_loss:8.4f} | 价值: {vf_loss:6.4f} | 熵: {entropy:6.4f}")
            
            # 时间统计
            print(f"时间 - 本轮: {format_time(iter_duration)} | 平均: {format_time(stats['avg_iteration_time'])}")
            print(f"进度 - 已用: {format_time(stats['elapsed_time'])} | 剩余: {format_time(stats['estimated_remaining'])}")
            
            if stats['estimated_finish']:
                finish_time = datetime.fromtimestamp(stats['estimated_finish'])
                print(f"预计完成时间: {finish_time.strftime('%H:%M:%S')}")
            
            print("-" * 80)
        
        # 训练完成统计
        end_datetime = datetime.now()
        total_duration = (end_datetime - start_datetime).total_seconds()
        final_stats = timer.get_stats()
        
        print(f"\n🏁 训练完成! ({end_datetime.strftime('%H:%M:%S')})")
        print("=" * 80)
        print(f"📊 最终统计:")
        print(f"   最佳奖励: {best_reward:.2f}")
        print(f"   总Episodes: {total_episodes}")
        print(f"   总用时: {format_time(total_duration)}")
        print(f"   平均每轮: {format_time(final_stats['avg_iteration_time'])}")
        print(f"   最快单轮: {format_time(final_stats['fastest_iteration'])}")
        print(f"   最慢单轮: {format_time(final_stats['slowest_iteration'])}")
        
        # 保存训练摘要
        summary = {
            "training_info": {
                "start_time": start_datetime.isoformat(),
                "end_time": end_datetime.isoformat(),
                "total_duration": total_duration,
                "iterations": num_iterations
            },
            "results": {
                "best_reward": best_reward,
                "total_episodes": total_episodes,
                "avg_iteration_time": final_stats['avg_iteration_time'],
                "fastest_iteration": final_stats['fastest_iteration'],
                "slowest_iteration": final_stats['slowest_iteration']
            },
            "training_history": training_history,
            "checkpoint": str(best_checkpoint) if best_checkpoint else None
        }
        
        summary_file = f"{results_dir}/training_summary.json"
        with open(summary_file, "w", encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"📄 训练摘要已保存: {summary_file}")
        
        if best_checkpoint:
            print(f"💾 最佳模型检查点: {results_dir}/")
        
        # 清理
        algo.stop()
        
        return summary
        
    except Exception as e:
        print(f"❌ 训练出错: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    # 记录脚本开始时间
    script_start_time = time.time()
    script_start_datetime = datetime.now()
    
    print("🐧 WSL增强版MARL训练系统")
    print("🔧 特性: 时间统计 | 进度预测 | 简洁输出 | 性能监控")
    print(f"🕐 脚本启动时间: {script_start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 询问用户训练轮次
    try:
        iterations = input("\n请输入训练轮次 (默认10): ").strip()
        iterations = int(iterations) if iterations else 10
    except ValueError:
        iterations = 10
    
    print(f"\n🚀 开始 {iterations} 轮训练...")
    
    # 运行训练
    result = run_enhanced_training(num_iterations=iterations)
    
    # 计算脚本总运行时间
    script_end_time = time.time()
    script_end_datetime = datetime.now()
    total_script_time = script_end_time - script_start_time
    
    if result:
        print("\n✅ 训练成功完成!")
        
        # 显示详细时间统计
        print(f"\n⏰ 完整时间统计:")
        print(f"   脚本开始: {script_start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"   脚本结束: {script_end_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"   脚本总运行时间: {total_script_time/60:.1f}分钟 ({total_script_time:.1f}秒)")
        
        # 从结果中获取纯训练时间
        if 'training_info' in result and 'total_duration' in result['training_info']:
            training_time = result['training_info']['total_duration']
            setup_time = total_script_time - training_time
            print(f"   纯训练时间: {training_time/60:.1f}分钟 ({training_time:.1f}秒)")
            print(f"   环境初始化时间: {setup_time/60:.1f}分钟 ({setup_time:.1f}秒)")
            print(f"   训练效率: {training_time/total_script_time*100:.1f}%")
        
        print("\n📋 后续操作:")
        print("1. 查看结果: ls D:\\MPU\\毕业论文\\MARL_FOR_W_Factory\\wsl\\ray_result\\")
        print("2. 分析摘要: cat D:\\MPU\\毕业论文\\MARL_FOR_W_Factory\\wsl\\ray_result\\training_summary.json")
        print("3. 运行可视化: python wsl/analyze_results.py")
    else:
        print("\n❌ 训练失败!")
        print(f"⏰ 脚本运行时间: {total_script_time/60:.1f}分钟") 