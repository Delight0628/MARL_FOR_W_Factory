"""
基于Stable-Baselines3的多智能体强化学习训练脚本
替代Ray RLlib，解决Windows兼容性问题
使用SuperSuit包装器实现真正的MARL
"""

import os
import sys
import time
import json
import numpy as np
from typing import Dict, Any, List
import warnings
warnings.filterwarnings('ignore')

# 添加环境路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)  # 回到MARL_FOR_W_Factory目录
sys.path.append(parent_dir)

from environments.w_factory_env import make_parallel_env
from environments.w_factory_config import *

# 导入强化学习库
try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
    from stable_baselines3.common.monitor import Monitor
    import supersuit as ss
    print("✓ Stable-Baselines3 和 SuperSuit 导入成功")
except ImportError as e:
    print(f"❌ 缺少依赖库: {e}")
    print("请安装: pip install stable-baselines3[extra] supersuit")
    sys.exit(1)

class MARLTrainingCallback(BaseCallback):
    """MARL训练回调函数"""
    
    def __init__(self, eval_freq: int = 1000, verbose: int = 1):
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.episode_rewards = []
        self.episode_lengths = []
        
    def _on_step(self) -> bool:
        # 记录训练统计
        if len(self.model.ep_info_buffer) > 0:
            for info in self.model.ep_info_buffer:
                if 'r' in info:
                    self.episode_rewards.append(info['r'])
                if 'l' in info:
                    self.episode_lengths.append(info['l'])
        
        # 定期输出训练进度
        if self.num_timesteps % self.eval_freq == 0:
            if len(self.episode_rewards) > 0:
                recent_rewards = self.episode_rewards[-10:]
                avg_reward = np.mean(recent_rewards)
                print(f"步数: {self.num_timesteps:8d} | "
                      f"平均奖励: {avg_reward:8.2f} | "
                      f"回合数: {len(self.episode_rewards):4d}")
        
        return True

def create_env():
    """创建并包装环境"""
    print("创建PettingZoo环境...")
    
    # 创建原始环境
    env = make_parallel_env()
    
    # 修复SuperSuit兼容性问题：添加render_mode属性
    if not hasattr(env, 'render_mode'):
        env.render_mode = None
    
    # 使用SuperSuit包装器转换为单智能体环境
    # 这是实现MARL的关键步骤
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, 1, num_cpus=1, base_class='stable_baselines3')
    
    print(f"✓ 环境创建成功")
    print(f"  观测空间: {env.observation_space}")
    print(f"  动作空间: {env.action_space}")
    
    return env

def train_sb3_marl(total_timesteps: int = 100000, learning_rate: float = 3e-4):
    """使用Stable-Baselines3训练MARL"""
    
    print("=" * 60)
    print("W工厂多智能体强化学习训练 - Stable-Baselines3版本")
    print("=" * 60)
    print("框架: Stable-Baselines3 + SuperSuit")
    print("算法: PPO (Proximal Policy Optimization)")
    print("多智能体: 向量化环境MARL")
    print("=" * 60)
    
    # 验证配置
    if not validate_config():
        print("配置验证失败")
        return None
    
    try:
        # 创建环境
        env = create_env()
        
        # 创建PPO模型
        print("创建PPO模型...")
        model = PPO(
            "MlpPolicy",  # 多层感知机策略
            env,
            learning_rate=learning_rate,
            n_steps=2048,  # 每次更新收集的步数
            batch_size=64,  # 小批量大小
            n_epochs=10,    # 每次更新的训练轮数
            gamma=0.99,     # 折扣因子
            gae_lambda=0.95, # GAE参数
            clip_range=0.2,  # PPO裁剪参数
            ent_coef=0.01,   # 熵系数
            vf_coef=0.5,     # 价值函数系数
            max_grad_norm=0.5, # 梯度裁剪
            verbose=1,
            device='cpu'  # 使用CPU，避免GPU兼容性问题
        )
        
        print("✓ PPO模型创建成功")
        print(f"  策略网络: {model.policy}")
        print(f"  学习率: {learning_rate}")
        print(f"  总训练步数: {total_timesteps}")
        
        # 创建回调函数
        callback = MARLTrainingCallback(eval_freq=2000)
        
        # 开始训练
        print("\n开始训练...")
        start_time = time.time()
        
        model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            progress_bar=True
        )
        
        training_time = time.time() - start_time
        
        print("\n" + "=" * 60)
        print("Stable-Baselines3 MARL训练完成！")
        print("=" * 60)
        print(f"训练时间: {training_time/60:.2f} 分钟")
        print(f"总步数: {total_timesteps}")
        
        # 安全地计算平均奖励
        if len(callback.episode_rewards) > 0:
            recent_rewards = callback.episode_rewards[-10:]
            if len(recent_rewards) > 0:
                avg_reward = np.mean(recent_rewards)
                print(f"平均奖励: {avg_reward:.2f}")
            else:
                print("平均奖励: 无数据")
        else:
            print("平均奖励: 无数据")
        
        # 保存模型
        os.makedirs("models", exist_ok=True)
        model_path = "models/sb3_marl_model"
        model.save(model_path)
        print(f"✓ 模型已保存: {model_path}")
        
        # 保存训练统计
        training_stats = {
            "framework": "Stable-Baselines3",
            "algorithm": "PPO",
            "total_timesteps": total_timesteps,
            "training_time_minutes": training_time / 60,
            "episode_rewards": callback.episode_rewards,
            "episode_lengths": callback.episode_lengths,
            "final_avg_reward": float(np.mean(callback.episode_rewards[-10:])) if len(callback.episode_rewards) > 0 else 0.0,
            "agents": list(WORKSTATIONS.keys()),
            "config": {
                "learning_rate": learning_rate,
                "n_steps": 2048,
                "batch_size": 64,
                "n_epochs": 10
            }
        }
        
        os.makedirs("results", exist_ok=True)
        stats_file = f"results/sb3_training_stats_{int(time.time())}.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(training_stats, f, ensure_ascii=False, indent=2)
        
        print(f"✓ 训练统计已保存: {stats_file}")
        
        return model, training_stats
        
    except Exception as e:
        print(f"训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def evaluate_model(model, num_episodes: int = 10):
    """评估训练好的模型"""
    print(f"\n评估模型 ({num_episodes} 回合)...")
    
    try:
        env = create_env()
        
        eval_rewards = []
        eval_lengths = []
        
        for episode in range(num_episodes):
            obs = env.reset()
            episode_reward = 0
            episode_length = 0
            done = False
            
            while not done:
                # 使用训练好的策略
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action)
                
                # 处理奖励（可能是数组）
                if isinstance(reward, np.ndarray):
                    reward = float(reward.sum())
                elif isinstance(reward, (list, tuple)):
                    reward = float(sum(reward))
                else:
                    reward = float(reward)
                
                # 处理done（可能是数组）
                if isinstance(done, np.ndarray):
                    done = bool(done.any())
                elif isinstance(done, (list, tuple)):
                    done = bool(any(done))
                else:
                    done = bool(done)
                
                episode_reward += reward
                episode_length += 1
                
                if episode_length > 1000:  # 防止无限循环
                    break
            
            eval_rewards.append(episode_reward)
            eval_lengths.append(episode_length)
            
            print(f"  回合 {episode+1}: 奖励={episode_reward:.2f}, 长度={episode_length}")
        
        eval_results = {
            "mean_reward": float(np.mean(eval_rewards)),
            "std_reward": float(np.std(eval_rewards)),
            "mean_length": float(np.mean(eval_lengths)),
            "eval_rewards": eval_rewards,
            "eval_lengths": eval_lengths
        }
        
        print(f"\n评估结果:")
        print(f"  平均奖励: {eval_results['mean_reward']:.2f} ± {eval_results['std_reward']:.2f}")
        print(f"  平均长度: {eval_results['mean_length']:.1f}")
        
        return eval_results
        
    except Exception as e:
        print(f"评估过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return None

def compare_with_simple_baseline():
    """与简单基准算法对比"""
    print("\n运行基准算法对比...")
    
    try:
        # 尝试导入并运行简单训练脚本
        try:
            from simple_train import SimpleTrainer
            simple_trainer = SimpleTrainer()
            simple_results = simple_trainer.train(num_episodes=10)
            
            # 检查结果格式
            if isinstance(simple_results, dict) and 'episode_rewards' in simple_results:
                print(f"简单基准算法:")
                print(f"  平均奖励: {np.mean(simple_results['episode_rewards']):.2f}")
                print(f"  训练时间: {simple_results.get('training_time', 0)/60:.2f} 分钟")
                return simple_results
            else:
                print("简单基准算法结果格式不正确")
                return None
                
        except ImportError:
            print("未找到simple_train模块，跳过基准对比")
            return None
            
    except Exception as e:
        print(f"基准对比失败: {e}")
        return None

def main():
    """主函数"""
    print("W工厂多智能体强化学习系统")
    print("基于Stable-Baselines3的MARL实现")
    print("=" * 60)
    
    try:
        # 训练模型
        model, training_stats = train_sb3_marl(
            total_timesteps=50000,  # 适中的训练步数
            learning_rate=3e-4
        )
        
        if model is not None:
            print("\n🎉 SB3 MARL训练成功完成！")
            
            # 评估模型
            eval_results = evaluate_model(model, num_episodes=5)
            
            # 基准对比
            baseline_results = compare_with_simple_baseline()
            
            # 最终总结
            print("\n" + "=" * 60)
            print("最终结果总结")
            print("=" * 60)
            
            if eval_results:
                print(f"SB3 MARL平均奖励: {eval_results['mean_reward']:.2f}")
            
            if baseline_results:
                baseline_avg = np.mean(baseline_results['episode_rewards'])
                print(f"简单基准平均奖励: {baseline_avg:.2f}")
                
                if eval_results:
                    improvement = eval_results['mean_reward'] - baseline_avg
                    print(f"性能提升: {improvement:.2f} ({improvement/baseline_avg*100:.1f}%)")
            
            print("\n✅ 这是真正的MARL训练！")
            print("✅ 使用工业级PPO算法")
            print("✅ 多智能体协同学习")
            print("✅ Windows完全兼容")
            print("✅ 无Ray依赖问题")
            
        else:
            print("\n❌ SB3训练失败")
            
    except Exception as e:
        print(f"主程序执行失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 