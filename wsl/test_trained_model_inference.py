#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练模型推理测试脚本 - 验证训练好的MARL智能体性能
"""

import os
import sys
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any

# 添加环境路径
current_dir = Path(__file__).parent.absolute()
parent_dir = current_dir.parent
sys.path.append(str(current_dir))
sys.path.append(str(parent_dir))

try:
    import ray
    from ray.rllib.algorithms.ppo import PPO
    print("✅ Ray库导入成功")
except ImportError as e:
    print(f"❌ Ray库导入失败: {e}")
    sys.exit(1)

try:
    from environments.w_factory_env import WFactoryGymEnv
    from environments.w_factory_config import *
    print("✅ 工厂环境导入成功")
except ImportError as e:
    print(f"❌ 工厂环境导入失败: {e}")
    sys.exit(1)

try:
    from wsl_ray_marl_train import OptimizedWFactoryWrapper, env_creator
    print("✅ 主训练脚本包装器导入成功")
except ImportError as e:
    print(f"❌ 主训练脚本包装器导入失败: {e}")
    sys.exit(1)

def find_latest_checkpoint(checkpoint_dir: str) -> str:
    """查找最新的检查点"""
    if not os.path.exists(checkpoint_dir):
        raise FileNotFoundError(f"检查点目录不存在: {checkpoint_dir}")
    
    # 查找所有检查点目录
    checkpoint_dirs = []
    for item in os.listdir(checkpoint_dir):
        item_path = os.path.join(checkpoint_dir, item)
        if os.path.isdir(item_path) and item.startswith("PPO_"):
            checkpoint_dirs.append(item_path)
    
    if not checkpoint_dirs:
        raise FileNotFoundError(f"在 {checkpoint_dir} 中未找到PPO检查点")
    
    # 找到最新的检查点
    latest_checkpoint = max(checkpoint_dirs, key=os.path.getmtime)
    
    # 查找检查点文件
    for item in os.listdir(latest_checkpoint):
        if item.startswith("checkpoint_") and not item.endswith(".tmp"):
            checkpoint_path = os.path.join(latest_checkpoint, item)
            print(f"✅ 找到最新检查点: {checkpoint_path}")
            return checkpoint_path
    
    raise FileNotFoundError(f"在 {latest_checkpoint} 中未找到有效检查点文件")

def test_trained_model(checkpoint_path: str, num_episodes: int = 5) -> Dict[str, Any]:
    """测试训练好的模型"""
    print(f"\n🤖 加载训练好的MARL模型...")
    print(f"检查点路径: {checkpoint_path}")
    
    # 初始化Ray - 优化版本，避免重复初始化
    if not ray.is_initialized():
        print("🚀 初始化Ray (本地模式)...")
        ray.init(local_mode=True, ignore_reinit_error=True, log_to_driver=False)
    else:
        print("✅ Ray已初始化，跳过重复初始化")
    
    # 注册环境
    ray.tune.register_env("w_factory", env_creator)
    
    # 创建配置
    config = {
        'debug_level': 'INFO',
        'training_mode': False,
        'use_fixed_rewards': True,
        'show_completion_stats': True
    }
    
    try:
        # 加载训练好的算法
        trainer = PPO.from_checkpoint(checkpoint_path)
        print("✅ 模型加载成功")
        
        # 创建测试环境
        test_env = OptimizedWFactoryWrapper(config)
        
        results = []
        
        print(f"\n🎯 开始推理测试 ({num_episodes} episodes)...")
        print("=" * 60)
        
        for episode in range(num_episodes):
            print(f"\n📊 Episode {episode + 1}/{num_episodes}")
            print("-" * 40)
            
            obs, info = test_env.reset()
            episode_reward = 0
            step_count = 0
            completion_events = 0
            
            start_time = time.time()
            
            while step_count < 480:  # 最大步数限制
                # 使用训练好的策略获取动作
                actions = {}
                for agent in test_env.agents:
                    if agent in obs:
                        action = trainer.compute_single_action(obs[agent], policy_id="default_policy")
                        actions[agent] = action
                    else:
                        actions[agent] = 0  # 默认动作
                
                # 执行动作
                obs, rewards, terminated, truncated, info = test_env.step(actions)
                
                step_reward = sum(rewards.values())
                episode_reward += step_reward
                step_count += 1
                
                # 检查零件完成
                if hasattr(test_env, 'base_env') and hasattr(test_env.base_env, 'pz_env'):
                    sim = test_env.base_env.pz_env.sim
                    if sim:
                        current_completed = len(sim.completed_parts)
                        if current_completed > completion_events:
                            completion_events = current_completed
                            if completion_events <= 5 or completion_events % 5 == 0:  # 显示前5个和每5个
                                print(f"   🎉 第{completion_events}个零件完成 (步骤{step_count})")
                
                # 检查终止条件
                if terminated.get('__all__', False):
                    print(f"   🏁 Episode自然终止于第{step_count}步")
                    break
            
            episode_time = time.time() - start_time
            
            # 获取最终统计
            final_stats = {}
            if hasattr(test_env, 'base_env') and hasattr(test_env.base_env, 'pz_env'):
                sim = test_env.base_env.pz_env.sim
                if sim and hasattr(sim, 'get_completion_stats'):
                    final_stats = sim.get_completion_stats()
            
            # 计算关键指标
            makespan = final_stats.get('makespan', step_count)
            completion_rate = final_stats.get('completion_rate', 0)
            
            # 设备利用率
            avg_utilization = 0
            if 'utilization_stats' in final_stats and final_stats['utilization_stats']:
                utils = list(final_stats['utilization_stats'].values())
                avg_utilization = np.mean(utils)
            
            # 延期分析
            tardiness_info = final_stats.get('tardiness_info', {})
            late_orders = tardiness_info.get('late_orders', 0)
            max_tardiness = tardiness_info.get('max_tardiness', 0)
            total_orders = final_stats.get('total_orders', 1)
            tardiness_rate = (late_orders / total_orders) * 100 if total_orders > 0 else 0
            
            # 自然终止判断
            natural_termination = step_count < 480
            
            episode_result = {
                'episode': episode + 1,
                'total_reward': episode_reward,
                'steps': step_count,
                'makespan': makespan,
                'completion_rate': completion_rate,
                'avg_utilization': avg_utilization,
                'tardiness_rate': tardiness_rate,
                'max_tardiness': max_tardiness,
                'natural_termination': natural_termination,
                'episode_time': episode_time,
                'completed_parts': completion_events
            }
            
            results.append(episode_result)
            
            # 显示episode结果
            print(f"   📈 Episode结果:")
            print(f"     总奖励: {episode_reward:.1f}")
            print(f"     步数: {step_count} ({'自然终止' if natural_termination else '时间截断'})")
            print(f"     完成零件: {completion_events}/29 ({completion_rate:.1f}%)")
            print(f"     Makespan: {makespan:.1f}分钟")
            print(f"     平均设备利用率: {avg_utilization:.1%}")
            print(f"     延期率: {tardiness_rate:.1f}% ({late_orders}个订单)")
            print(f"     用时: {episode_time:.1f}秒")
        
        # 计算总体统计
        avg_results = {
            'avg_reward': np.mean([r['total_reward'] for r in results]),
            'avg_steps': np.mean([r['steps'] for r in results]),
            'avg_makespan': np.mean([r['makespan'] for r in results]),
            'avg_completion_rate': np.mean([r['completion_rate'] for r in results]),
            'avg_utilization': np.mean([r['avg_utilization'] for r in results]),
            'avg_tardiness_rate': np.mean([r['tardiness_rate'] for r in results]),
            'avg_max_tardiness': np.mean([r['max_tardiness'] for r in results]),
            'natural_termination_rate': np.mean([r['natural_termination'] for r in results]) * 100,
            'avg_completed_parts': np.mean([r['completed_parts'] for r in results])
        }
        
        return {
            'episodes': results,
            'summary': avg_results,
            'checkpoint_path': checkpoint_path
        }
        
    except Exception as e:
        print(f"❌ 模型测试失败: {e}")
        raise
    finally:
        ray.shutdown()

def analyze_model_performance(test_results: Dict[str, Any]):
    """分析模型性能"""
    print(f"\n📊 训练模型性能分析")
    print("=" * 60)
    
    summary = test_results['summary']
    episodes = test_results['episodes']
    
    print(f"🎯 总体性能指标:")
    print(f"   平均奖励: {summary['avg_reward']:.1f}")
    print(f"   平均步数: {summary['avg_steps']:.1f}")
    print(f"   平均完成零件: {summary['avg_completed_parts']:.1f}/29")
    print(f"   自然终止率: {summary['natural_termination_rate']:.1f}%")
    
    print(f"\n🏭 项目核心目标达成情况:")
    
    # 目标1: 最小化Makespan
    makespan = summary['avg_makespan']
    if makespan < 400:
        makespan_grade = "优秀 ✅"
    elif makespan < 450:
        makespan_grade = "良好 ⚠️"
    else:
        makespan_grade = "需改进 ❌"
    print(f"   1. Makespan: {makespan:.1f}分钟 ({makespan_grade})")
    
    # 目标2: 最大化设备利用率
    utilization = summary['avg_utilization']
    if utilization > 0.6:
        util_grade = "优秀 ✅"
    elif utilization > 0.4:
        util_grade = "良好 ⚠️"
    else:
        util_grade = "需改进 ❌"
    print(f"   2. 设备利用率: {utilization:.1%} ({util_grade})")
    
    # 目标3: 最小化延期
    tardiness = summary['avg_tardiness_rate']
    if tardiness < 10:
        tardiness_grade = "优秀 ✅"
    elif tardiness < 25:
        tardiness_grade = "良好 ⚠️"
    else:
        tardiness_grade = "需改进 ❌"
    print(f"   3. 延期率: {tardiness:.1f}% ({tardiness_grade})")
    
    # 学习效果分析
    print(f"\n🧠 学习效果分析:")
    completion_rate = summary['avg_completion_rate']
    if completion_rate > 80:
        print(f"   ✅ 任务完成能力: 优秀 ({completion_rate:.1f}%)")
    elif completion_rate > 60:
        print(f"   ⚠️  任务完成能力: 良好 ({completion_rate:.1f}%)")
    else:
        print(f"   ❌ 任务完成能力: 需改进 ({completion_rate:.1f}%)")
    
    natural_rate = summary['natural_termination_rate']
    if natural_rate > 50:
        print(f"   ✅ 效率优化: 优秀 ({natural_rate:.1f}%自然终止)")
    elif natural_rate > 20:
        print(f"   ⚠️  效率优化: 良好 ({natural_rate:.1f}%自然终止)")
    else:
        print(f"   ❌ 效率优化: 需改进 ({natural_rate:.1f}%自然终止)")
    
    # 稳定性分析
    reward_std = np.std([r['total_reward'] for r in episodes])
    makespan_std = np.std([r['makespan'] for r in episodes])
    
    print(f"\n📈 稳定性分析:")
    print(f"   奖励标准差: {reward_std:.1f}")
    print(f"   Makespan标准差: {makespan_std:.1f}分钟")
    
    if reward_std < summary['avg_reward'] * 0.1:
        print(f"   ✅ 性能稳定性: 优秀")
    elif reward_std < summary['avg_reward'] * 0.2:
        print(f"   ⚠️  性能稳定性: 良好")
    else:
        print(f"   ❌ 性能稳定性: 需改进")

def main():
    """主函数"""
    print("🤖 W工厂MARL训练模型推理测试")
    print("=" * 60)
    
    # 查找检查点 - WSL路径兼容
    checkpoint_dir = os.path.join(current_dir, "ray_result")
    
    # 如果在WSL环境中，检查Windows路径
    if not os.path.exists(checkpoint_dir):
        # 尝试WSL到Windows的路径映射
        windows_path = "/mnt/d/MPU/毕业论文/MARL_FOR_W_Factory/wsl/ray_result"
        if os.path.exists(windows_path):
            checkpoint_dir = windows_path
            print(f"🔍 使用WSL路径: {checkpoint_dir}")
        else:
            print(f"🔍 检查路径: {checkpoint_dir}")
            print(f"🔍 WSL路径: {windows_path}")
    
    try:
        checkpoint_path = find_latest_checkpoint(checkpoint_dir)
    except FileNotFoundError as e:
        print(f"❌ {e}")
        print(f"💡 请先运行训练脚本: python wsl/wsl_ray_marl_train.py")
        return
    
    # 测试模型
    try:
        test_results = test_trained_model(checkpoint_path, num_episodes=3)
        
        # 分析性能
        analyze_model_performance(test_results)
        
        print(f"\n🎯 推理测试完成!")
        print(f"💡 下一步建议:")
        
        summary = test_results['summary']
        if (summary['avg_makespan'] < 450 and 
            summary['avg_utilization'] > 0.5 and 
            summary['avg_tardiness_rate'] < 20):
            print("   ✅ 模型性能良好，可以进行基准对比测试")
            print("   🔧 运行: python wsl/test_performance_benchmark.py")
        else:
            print("   ⚠️  模型性能需要改进")
            print("   🔧 建议增加训练轮次或调整超参数")
            
    except Exception as e:
        print(f"❌ 推理测试失败: {e}")

if __name__ == "__main__":
    main()